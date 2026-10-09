import os, tempfile, uuid, subprocess, threading, time, json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from faster_whisper import WhisperModel
import yt_dlp

MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
WORK_DIR = "/tmp/clipforge"
os.makedirs(WORK_DIR, exist_ok=True)

app = FastAPI(title="CLIPFORGE AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

print(f"Загружаю модель {MODEL_SIZE}...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=2)
print("Модель готова.")

jobs = {}
tasks = {}

def to_ass_time(t):
    h=int(t//3600); m=int((t%3600)//60); s=int(t%60); cs=int((t-int(t))*100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def hex_to_ass(hx, alpha=0):
    h = hx.lstrip("#"); r,g,b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}"

def wrap_lines(text, maxc, maxl):
    words = text.split(); lines, cur = [], ""
    for w in words:
        if not cur: cur = w
        elif len(cur)+1+len(w) <= maxc: cur += " " + w
        else: lines.append(cur); cur = w
    if cur: lines.append(cur)
    return "\\N".join(lines[:maxl])

STYLES = {
    "bold":    {"fs":64,"color":"#ffffff","oc":"#000000","ow":5,"sh":1,"mv":260,"mc":22,"ml":2},
    "minimal": {"fs":56,"color":"#ffffff","oc":"#000000","ow":2,"sh":1,"mv":220,"mc":26,"ml":2},
    "yellow":  {"fs":72,"color":"#ffea00","oc":"#12002b","ow":5,"sh":1,"mv":260,"mc":20,"ml":2},
}

def build_ass(cues, style, W, H, clip_start):
    s = STYLES.get(style, STYLES["bold"])
    pri = hex_to_ass(s["color"]); out = hex_to_ass(s["oc"])
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,{s["fs"]},{pri},{pri},{out},&H80000000,-1,0,0,0,100,100,0,0,1,{s["ow"]},{s["sh"]},2,40,40,{s["mv"]},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [head]
    for c in cues:
        a = max(0, c["start"] - clip_start); b = max(0, c["end"] - clip_start)
        if b <= 0: continue
        txt = wrap_lines(c["text"], s["mc"], s["ml"]).replace("'", "\\'")
        lines.append(f"Dialogue: 0,{to_ass_time(a)},{to_ass_time(b)},Default,,0,0,0,,{txt}")
    return "\n".join(lines)

def download_video(url, out_dir):
    fid = uuid.uuid4().hex[:8]
    tmpl = os.path.join(out_dir, f"{fid}.%(ext)s")
    opts = {"format":"bestvideo[height<=720]+bestaudio/best[height<=720]/best",
            "outtmpl":tmpl,"noplaylist":True,"quiet":True,"no_warnings":True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info), info.get("title","video"), info.get("duration",0)

def score_text(t):
    l = t.lower(); sc = 0
    if "?" in t or "!" in t: sc += 0.6
    for w in ["почему","как ","что ","когда","зачем","главное","представь","интересн","смешн","удивительн","секрет","шок"]:
        if w in l: sc += 0.3
    return sc

def suggest_clips(cues, dur, min_len, max_len, count):
    wins = []; n = len(cues)
    for i in range(n):
        start = cues[i]["start"]; end = start; text = ""; j = i
        while j < n and (end-start) < max_len:
            text = (text + " " + cues[j]["text"]).strip()
            end = cues[j]["end"]
            if (end-start) >= min_len and cues[j]["text"].rstrip().endswith((".","!","?","…")): break
            j += 1
        ln = end-start
        if ln < min_len or ln > max_len+3: continue
        wins.append({"start":max(0,start-0.15),"end":min(dur,end+0.15),
                     "text":text[:250],
                     "reason":"Эмоция/вопрос" if ("?" in text or "!" in text) else "Законченная мысль",
                     "score":score_text(text)+ln/100})
    if not wins:
        step = max(min_len, dur/(count*2)); t = 0
        while t + min_len <= dur:
            wins.append({"start":t,"end":min(dur,t+min_len),"text":"","reason":"Сетка","score":0.1})
            t += step
    wins.sort(key=lambda w:-w["score"])
    picked = []
    for w in wins:
        if len(picked) >= count: break
        bad = False
        for p in picked:
            o = max(0, min(p["end"], w["end"]) - max(p["start"], w["start"]))
            u = min(p["end"]-p["start"], w["end"]-w["start"])
            if u and o/u > 0.4: bad = True; break
        if not bad: picked.append(w)
    picked.sort(key=lambda w:w["start"])
    return picked

def prepare_task(task_id, url, language):
    t = tasks[task_id]
    try:
        t["status"] = "downloading"; t["progress"] = 5
        job_id = task_id
        job_dir = os.path.join(WORK_DIR, job_id); os.makedirs(job_dir, exist_ok=True)
        path, title, dur = download_video(url, job_dir)
        t["status"] = "transcribing"; t["progress"] = 30
        segments, info = model.transcribe(path, language=(language or None), vad_filter=True, beam_size=1)
        cues = [{"start":s.start,"end":s.end,"text":s.text.strip()} for s in segments]
        merged = []
        for c in cues:
            if merged and (c["start"]-merged[-1]["end"] < 0.25) and (merged[-1]["end"]-merged[-1]["start"]+c["end"]-c["start"] < 6):
                merged[-1]["end"] = c["end"]; merged[-1]["text"] += " " + c["text"]
            else:
                merged.append(dict(c))
        total_dur = info.duration or dur
        jobs[job_id] = {"path":path,"title":title,"duration":total_dur,"cues":merged,"language":info.language,"dir":job_dir}
        plans = suggest_clips(merged, total_dur, 15, 35, 8)
        t["data"] = {"job_id":job_id,"title":title,"duration":total_dur,"language":info.language,"cues":merged,"plans":plans}
        t["status"] = "done"; t["progress"] = 100
    except Exception as e:
        t["status"] = "error"; t["error"] = str(e)

def render_task(task_id, job_id, clips, style, W, H):
    t = tasks[task_id]
    try:
        job = jobs[job_id]
        out_dir = os.path.join(job["dir"], "clips"); os.makedirs(out_dir, exist_ok=True)
        files = []; total = len(clips)
        for i, clip in enumerate(clips):
            t["progress"] = int(i/total*95)
            t["status"] = f"рендер {i+1}/{total}"
            start = float(clip["start"]); dur = float(clip["end"]) - start
            if dur <= 0: continue
            ass = build_ass(job["cues"], style, W, H, start)
            ass_path = os.path.join(out_dir, f"c{i}.ass")
            with open(ass_path,"w",encoding="utf-8") as f: f.write(ass)
            out_path = os.path.join(out_dir, f"CLIPFORGE_{i+1:02d}.mp4")
            vf = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},ass={ass_path}"
            cmd = ["ffmpeg","-hide_banner","-y","-ss",str(start),"-t",str(dur),"-i",job["path"],
                   "-vf",vf,"-c:v","libx264","-preset","veryfast","-crf","23","-pix_fmt","yuv420p",
                   "-c:a","aac","-b:a","128k","-movflags","+faststart",out_path]
            subprocess.run(cmd, check=True, capture_output=True, timeout=900)
            files.append({"name":f"CLIPFORGE_{i+1:02d}.mp4","path":out_path,
                          "size":os.path.getsize(out_path),"start":start,"end":start+dur})
        t["files"] = files; t["status"] = "done"; t["progress"] = 100
    except Exception as e:
        t["status"] = "error"; t["error"] = str(e)

@app.get("/", response_class=HTMLResponse)
def index():
    return open("index.html", encoding="utf-8").read()

@app.get("/health")
def health():
    return {"ok": True, "model": MODEL_SIZE}

@app.post("/api/prepare")
def prepare_start(url: str = Form(...), language: str = Form("ru")):
    task_id = uuid.uuid4().hex[:10]
    tasks[task_id] = {"status":"pending","progress":0,"error":None,"data":None}
    threading.Thread(target=prepare_task, args=(task_id,url,language), daemon=True).start()
    return {"task_id": task_id}

@app.post("/api/render")
def render_start(job_id: str = Form(...), clips_json: str = Form(...),
                 style: str = Form("bold"), width: int = Form(720), height: int = Form(1280)):
    if job_id not in jobs: raise HTTPException(404, "job not found")
    clips = json.loads(clips_json)
    task_id = uuid.uuid4().hex[:10]
    tasks[task_id] = {"status":"pending","progress":0,"error":None,"files":[]}
    threading.Thread(target=render_task, args=(task_id,job_id,clips,style,width,height), daemon=True).start()
    return {"task_id": task_id}

@app.get("/api/job/{task_id}")
def job_status(task_id):
    if task_id not in tasks: raise HTTPException(404)
    t = tasks[task_id]
    files = [{"name":f["name"],"url":f"/api/clip/{task_id}/{i}","size":f["size"],
              "start":f["start"],"end":f["end"]} for i,f in enumerate(t.get("files",[]))]
    return {"status":t["status"],"progress":t["progress"],"error":t.get("error"),
            "data":t.get("data"),"files":files}

@app.get("/api/clip/{task_id}/{idx}")
def serve_clip(task_id, idx):
    if task_id not in tasks: raise HTTPException(404)
    files = tasks[task_id].get("files",[])
    i = int(idx)
    if i < 0 or i >= len(files): raise HTTPException(404)
    f = files[i]
    return FileResponse(f["path"], filename=f["name"], media_type="video/mp4")
