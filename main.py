from fastapi import FastAPI, UploadFile, File, HTTPException
from feedback_engine import process_audio_bytes

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the audio analysis API!"}

@app.post("/analyze/")
async def analyze_audio(file: UploadFile = File(...)):
    if file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Only WAV files are supported")

    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

    result = process_audio_bytes(audio_bytes)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    return result
