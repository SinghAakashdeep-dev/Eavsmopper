"""
Audio sources. Each source pushes (label, block) tuples into a shared queue.
A block is a 1-D float32 numpy array at 16 kHz mono (what Whisper wants).

  MicSource          -> your microphone             (label "Me")
  SystemAudioSource  -> whatever your PC is playing (label "Them"), via WASAPI loopback

Both use PyAudioWPatch (one PortAudio library for everything). Mixing it with
sounddevice in the same process meant two separate PortAudio copies fighting over
Windows audio, which is why the mic went silent in "Call + mic" mode.
"""

from math import gcd

import numpy as np

SAMPLE_RATE = 16000
BLOCK_SECONDS = 0.1


class _WasapiSource:
    """Opens one WASAPI device at its own native rate, then downmixes + resamples to 16 kHz mono."""

    loopback = False

    def __init__(self, label, out_q):
        self.label = label
        self.out_q = out_q
        self.pa = None
        self.stream = None
        self.device_name = "?"

    def _pick_device(self, pyaudio, wasapi):
        raise NotImplementedError

    def start(self):
        import pyaudiowpatch as pyaudio          # pip name: PyAudioWPatch (Windows only)
        from scipy.signal import resample_poly

        self.pa = pyaudio.PyAudio()
        wasapi = self.pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        device = self._pick_device(pyaudio, wasapi)
        self.device_name = device["name"]

        rate = int(device["defaultSampleRate"])
        channels = max(1, int(device["maxInputChannels"]))
        g = gcd(SAMPLE_RATE, rate)
        up, down = SAMPLE_RATE // g, rate // g

        def callback(in_data, frame_count, time_info, status):
            data = np.frombuffer(in_data, dtype=np.float32).reshape(-1, channels).mean(axis=1)
            if rate != SAMPLE_RATE:
                data = resample_poly(data, up, down)
            self.out_q.put((self.label, data.astype(np.float32)))
            return (None, pyaudio.paContinue)

        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=channels,
            rate=rate,
            input=True,
            input_device_index=device["index"],
            frames_per_buffer=int(rate * BLOCK_SECONDS),
            stream_callback=callback,
        )
        self.stream.start_stream()

    def stop(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.pa:
            self.pa.terminate()


class MicSource(_WasapiSource):
    """Your default Windows recording device (Settings > Sound > Input)."""

    def _pick_device(self, pyaudio, wasapi):
        idx = wasapi["defaultInputDevice"]
        if idx < 0:
            raise RuntimeError("No default microphone found in Windows sound settings.")
        return self.pa.get_device_info_by_index(idx)


class SystemAudioSource(_WasapiSource):
    """
    Records what your default speakers/headphones are playing (the other people in the call).
    Note: loopback sends NO blocks while nothing is playing; the transcriber handles that.
    """

    def _pick_device(self, pyaudio, wasapi):
        speakers = self.pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        if speakers.get("isLoopbackDevice"):
            return speakers
        for dev in self.pa.get_loopback_device_info_generator():
            if speakers["name"] in dev["name"]:
                return dev
        raise RuntimeError("No loopback device found for your default speakers.")
