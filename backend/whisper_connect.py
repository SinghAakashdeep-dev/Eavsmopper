from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import soundfile as sf
import torch

# float32 keeps the loaded audio small and clean (sf.read defaults to float64)
audio, sr = sf.read("output.wav", dtype="float32")

# FIX: GPU -> float16 (fast, half the memory), CPU -> float32 (CPUs are bad at float16).
# Use real torch types, not strings.
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
# FIX: the model itself must be loaded in the same dtype as the inputs
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    "openai/whisper-large-v3-turbo",
    torch_dtype=torch_dtype,
    device_map="auto",
)

inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
# FIX: move features to wherever the model lives AND match its dtype, read from the model itself
features = inputs.input_features.to(model.device, dtype=model.dtype)

with torch.no_grad():
    predicted_ids = model.generate(features)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(transcription)
