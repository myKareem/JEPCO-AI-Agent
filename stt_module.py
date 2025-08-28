import nemo.collections.asr as nemo_asr
import os
from pydub import AudioSegment # Import pydub

# --- 1. DEFINE FILE PATHS ---
input_audio_path = 'my_audio.wav' # Your original stereo file
mono_audio_path = 'my_audio_mono.wav' # The new mono file we will create

# --- 2. CONVERT STEREO TO MONO (NEW STEP) ---
print(f"Loading '{input_audio_path}' and converting to mono...")
sound = AudioSegment.from_wav(input_audio_path)
sound = sound.set_channels(1)
sound.export(mono_audio_path, format="wav")
print(f"Mono file saved at '{mono_audio_path}'")

# --- 3. LOAD THE ASR MODEL ---
print("Loading the ASR model...")
# Using the model name that worked for you before
asr_model = nemo_asr.models.EncDecHybridRNNTCTCModel.from_pretrained(
    model_name="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
)

# --- 4. TRANSCRIBE THE MONO FILE ---
# We now pass the path to the new mono audio file
print(f"Transcribing '{mono_audio_path}'...")
transcriptions = asr_model.transcribe([mono_audio_path])

# --- 5. PRINT THE RESULT ---
print("\n--- Transcription Result ---")
if transcriptions:
    print(f"Text: {transcriptions[0]}")
else:
    print("Could not transcribe the audio.")
print("--------------------------")

# Clean up the temporary mono file
os.remove(mono_audio_path)