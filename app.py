import os, tempfile, uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from starlette.background import BackgroundTask
from faster_whisper import WhisperModel
import yt_dlp

MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
DOWNLOAD_DIR = "/tmp/downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

app = FastAPI(title="CLIPFORGE AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"Загружаю модель {MODEL_SIZE}...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=2)
print("Модель готова.")

@app.get("/", response_class=HTMLResponse)
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/worker.js")
def worker():
    return FileResponse("worker.js", media_type="text/javascript")

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_SIZE}

def download_video(url: str) -> str:
    file_id = str(uuid.uuid4())
    out_tmpl = os.path.join(DOWNLOAD_DIR, file_id + ".%(ext)s")
    ydl_opts = {
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "outtmpl": out_tmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": {
            "youtube": {
                "player_client": ["tv", "mweb"],
            }
        },
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        },
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
        except Exception as e:
            raise HTTPException(status_code=400, detail="yt-dlp: " + str(e))

@app.post("/api/download")
async def download_from_url(url: str = Form(...)):
    path = download_video(url)
    if not os.path.exists(path):
        raise HTTPException(status_code=500, detail="Файл не скачан")
    def cleanup():
        try:
            os.unlink(path)
        except:
            pass
    return FileResponse(
        path,
        filename=os.path.basename(path),
        media_type="video/mp4",
        background=BackgroundTask(cleanup)
    )

@app.post("/api/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form("ru")):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        segments, info = model.transcribe(
            path,
            language=language or None,
            vad_filter=True,
            beam_size=1,
        )
        cues = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
        return {"language": info.language, "duration": info.duration, "cues": cues}
    finally:
        os.unlink(path)
