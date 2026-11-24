import sounddevice as sd
import soundfile as sf
import queue
fs = 16000  # Sample rate for whisper model
q = queue.Queue()
def callback(indata, frames, time, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, flush=True)
    q.put(indata.copy())    

try:
    with sf.SoundFile('output.wav', mode='w', samplerate=fs, channels=1) as file:
        with sd.InputStream(samplerate=fs, channels=1, callback=callback):
            print('#' * 80)
            print('Press Ctrl+C to stop the recording')
            print('#' * 80)
            while True:
                file.write(q.get())
except KeyboardInterrupt:
    print('\nRecording finished: output.wav')