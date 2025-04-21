import numpy as np
import whisper
import re

# Load the Whisper model
model = whisper.load_model("base")

IDEAL = {
    "wpm": [120, 150],
    "min_grammar_score": 80,
}

def analyze_wpm(transcript: str, duration: float) -> float:
    words = len(transcript.split())
    wpm = words / (duration / 60)
    return wpm

def analyze_fillers(transcript: str) -> list:
    fillers = re.findall(r'\b(uh|um|like|you know)\b', transcript, re.IGNORECASE)
    return fillers

def basic_grammar_score(transcript: str) -> int:
    score = 100 if re.search(r'[.!?]', transcript) else 60
    return score

def process_audio_bytes(audio_data: bytes) -> dict:
    audio_np = np.frombuffer(audio_data, dtype=np.int16)
    audio_np = audio_np.astype(np.float32)
    audio_np /= np.max(np.abs(audio_np))

    sr = 16000
    duration = len(audio_np) / sr

    try:
        result = model.transcribe(audio_np)
    except RuntimeError as e:
        return {}

    transcript = result['text']

    wpm = analyze_wpm(transcript, duration)
    fillers = analyze_fillers(transcript)
    grammar = basic_grammar_score(transcript)
    fluency = max(60, 95 - len(fillers) * 2)
    if grammar > 95 and len(fillers) < 2:
        confidence = 90
    elif grammar > 80 and len(fillers) < 5:
        confidence = 80
    else:
        confidence = 65


    tips = []
    # Analyze speaking speed
    if wpm < IDEAL["wpm"][0]:
        tips.append("You're speaking a bit slowly. Try to increase your pace to sound more engaging.")
        speed_feedback = "Too slow"
    elif wpm > IDEAL["wpm"][1]:
        tips.append("You're speaking too fast. Slow down a little to ensure clarity and understanding.")
        speed_feedback = "Too fast"
    else:
        speed_feedback = "Good pace"

    if len(fillers) > 2:
        tips.append("Reduce filler words.")
    if grammar < IDEAL["min_grammar_score"]:
        tips.append("Improve sentence structure.")

    feedback = {
        "transcript": transcript,
        "duration_sec": round(duration, 2),
        "wpm": wpm,
        "filler_words": fillers,
        "grammar_score": grammar,
        "fluency_score": fluency,
        "confidence_score": confidence,
        "speed_feedback": speed_feedback,
        "overall_rating": round((grammar + fluency + confidence) / 3, 1),
        "tips": tips
    }

    return feedback

# import whisper
# import numpy as np
# import io
# import soundfile as sf
# import tempfile

# def process_audio_bytes(audio_bytes):
#     try:
#         print("Loading Whisper model...")
#         model = whisper.load_model("base")

#         print("Reading audio bytes...")
#         audio_np, sr = sf.read(io.BytesIO(audio_bytes))

#         if len(audio_np.shape) > 1:
#             print("Converting stereo to mono...")
#             audio_np = np.mean(audio_np, axis=1)

#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
#             print("Writing temporary WAV file...")
#             sf.write(tmp.name, audio_np, sr)

#             print("Transcribing audio...")
#             result = model.transcribe(tmp.name)

#         transcript = result.get("text", "").strip()
#         segments = result.get("segments", [])
#         duration_sec = segments[-1]["end"] if segments else 0.0
#         num_words = len(transcript.split())
#         wpm = num_words / (duration_sec / 60) if duration_sec > 0 else 0

#         print("Generating feedback...")
#         feedback = generate_feedback(transcript, wpm)

#         return {
#             "transcript": transcript,
#             "duration_sec": duration_sec,
#             "wpm": wpm,
#             **feedback
#         }

#     except Exception as e:
#         return {"error": str(e)}

# def generate_feedback(transcript: str, wpm: float):
#     tips = []
#     filler_words_list = ["um", "uh", "like", "you know", "so"]
#     fillers_used = [word for word in filler_words_list if word in transcript.lower()]

#     if wpm > 160:
#         tips.append("Try to speak a little slower.")
#     elif wpm < 100:
#         tips.append("Try to speak a little faster.")

#     if fillers_used:
#         tips.append("Reduce usage of filler words like: " + ", ".join(fillers_used))

#     return {
#         "filler_words": fillers_used,
#         "grammar_score": 100,
#         "fluency_score": 95,
#         "confidence_score": 85,
#         "overall_rating": 93.3,
#         "tips": tips
#     }


