from faster_whisper import WhisperModel

# Use CPU instead of GPU
model = WhisperModel("base", device="cpu", compute_type="int8")
segments, info = model.transcribe("luluslibraryvolume2_01_alcott_64kb.mp3")

# Save output to a file
with open("transcription.txt", "w", encoding="utf-8") as f:
    for segment in segments:
        line = f"[{segment.start:.2f} - {segment.text}]\n"
        # print(line, end="")  # Print to console
        f.write(line)  # Save to file

print(segments)
