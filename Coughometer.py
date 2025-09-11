from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil
import uuid
import os
from pydub import AudioSegment
from inference import predict_from_audio

app = FastAPI()

# (Optional) serve static files if you later add CSS/JS
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# --------- HTML FRONTEND ----------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Coughometer</title>
    </head>
    <body>
        <h1>Coughometer 🎤</h1>
        <p>Upload a WAV file or record live.</p>

        <!-- File Upload -->
        <form id="upload-form" enctype="multipart/form-data">
            <input type="file" name="file" accept=".wav,.webm,.ogg" />
            <button type="submit">Upload & Predict</button>
        </form>

        <hr/>

        <!-- Recorder -->
        <button id="record-btn">🎙️ Start Recording</button>
        <button id="stop-btn" disabled>⏹ Stop</button>
        <audio id="audio-playback" controls></audio>
        <button id="send-btn" disabled>🚀 Send to Coughometer</button>

        <h3 id="result"></h3>

        <script>
        // ---- Upload ----
        document.getElementById("upload-form").onsubmit = async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const res = await fetch("/predict", { method: "POST", body: formData });
            const data = await res.json();
            document.getElementById("result").innerText =
                "Prediction: " + (data.prediction || data.detail);
        };

        // ---- Recorder ----
        let mediaRecorder;
        let audioChunks = [];

        const recordBtn = document.getElementById("record-btn");
        const stopBtn = document.getElementById("stop-btn");
        const sendBtn = document.getElementById("send-btn");
        const audioPlayback = document.getElementById("audio-playback");

        recordBtn.onclick = async () => {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);

            mediaRecorder.onstop = () => {
                const blob = new Blob(audioChunks, { type: "audio/webm" });
                audioPlayback.src = URL.createObjectURL(blob);

                sendBtn.onclick = async () => {
                    const formData = new FormData();
                    formData.append("file", blob, "recording.webm");

                    const res = await fetch("/predict", {
                        method: "POST",
                        body: formData
                    });
                    const data = await res.json();
                    document.getElementById("result").innerText =
                        "Prediction: " + (data.prediction || data.detail);
                };

                sendBtn.disabled = false;
            };

            mediaRecorder.start();
            recordBtn.disabled = true;
            stopBtn.disabled = false;
        };

        stopBtn.onclick = () => {
            mediaRecorder.stop();
            recordBtn.disabled = false;
            stopBtn.disabled = true;
        };
        </script>
    </body>
    </html>
    """

# --------- API ENDPOINT ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # accept wav, webm, ogg
    if not (file.filename.endswith(".wav") or file.filename.endswith(".webm") or file.filename.endswith(".ogg")):
        raise HTTPException(status_code=400, detail="Only WAV, WebM, or OGG audio files are accepted.")

    temp_path = f"temp_{uuid.uuid4()}_{file.filename}"
    wav_path = temp_path.rsplit(".", 1)[0] + ".wav"

    try:
        # save raw upload
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # if not wav → convert
        if not temp_path.endswith(".wav"):
            audio = AudioSegment.from_file(temp_path)
            audio.export(wav_path, format="wav")
            os.remove(temp_path)
        else:
            wav_path = temp_path

        prediction = predict_from_audio(wav_path)
        return {"prediction": prediction}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)
