import os
import warnings
import logging

# --- 0. SILENCE LOGGING ---
os.environ["NEMO_LOG_LEVEL"] = "ERROR"  # must be set before importing NeMo
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from pydub import AudioSegment
import nemo.collections.asr as nemo_asr

# --- 1. DEFINE FILE PATHS ---
input_audio_path = "my_audio.wav"
mono_audio_path = "my_audio_mono.wav"

# --- 2. CONVERT AUDIO TO MONO ---
sound = AudioSegment.from_wav(input_audio_path)
sound = sound.set_channels(1)
sound.export(mono_audio_path, format="wav")

# --- 3. LOAD THE ASR MODEL ---
asr_model = nemo_asr.models.EncDecHybridRNNTCTCModel.from_pretrained(
    model_name="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
)

# --- 4. TRANSCRIBE THE MONO FILE (suppress progress bar) ---
results = asr_model.transcribe(
    [mono_audio_path],
    return_hypotheses=True,   # ensures we can access .text
    verbose=False             # disables tqdm progress bar
)

# --- 5. PRINT ONLY THE TRANSCRIPT ---
if results:
    print(results[0].text)   # extract just the transcript
else:
    print("Could not transcribe the audio.")

# --- 6. CLEAN UP ---
os.remove(mono_audio_path)
