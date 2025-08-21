import torch
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf

# 1. Automatically select GPU if available, otherwise fall back to CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Initialize the pipeline on the selected device
synthesiser = pipeline("text-to-speech", "MBZUAI/speecht5_tts_clartts_ar", device=device)

embeddings_dataset = load_dataset("herwoww/arabic_xvector_embeddings", split="validation")

# 3. Move the speaker embedding tensor to the same device as the pipeline
speaker_embedding = torch.tensor(embeddings_dataset[105]["speaker_embeddings"]).unsqueeze(0).to(device)

# The rest of your code remains the same
speech = synthesiser("1. **عدادات الكهرباء السكنية (المنزلية)**", forward_params={"speaker_embeddings": speaker_embedding})

# The audio output from the pipeline is on the CPU, so no changes are needed here
sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])

print("Speech saved to speech.wav")