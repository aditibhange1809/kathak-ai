from faster_whisper import WhisperModel
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python test_whisper.py <audio_file_path>")
    sys.exit(1)

audio_path = Path(sys.argv[1])

if not audio_path.is_file():
    print(f"Audio file not found: {audio_path}")
    sys.exit(1)

print("Loading Whisper model...")

model = WhisperModel("small", device="cpu", compute_type="int8")

print("Transcribing audio...")

segments, info = model.transcribe(
    str(audio_path),
    beam_size=5
)

print(f"\nDetected language: {info.language}")
print("\n--- TRANSCRIPTION ---")

for segment in segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}")

print("\nTranscription complete!")