import os, uuid, shutil, asyncio, threading, logging, random, json, subprocess
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from telegram.constants import ChatAction
from faster_whisper import WhisperModel
from groq import Groq
import yt_dlp
from fastapi import FastAPI
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
DOWNLOAD_DIR = "/tmp/downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN не задан")

log.info(f"Загружаю Whisper {MODEL_SIZE}...")
whisper = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=2)
log.info("Whisper готов.")

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
if not groq_client:
    log.warning("GROQ_API_KEY не задан — эвристика")

health_app = FastAPI()

@health_app.get("/")
def health():
    return {"ok": True, "model": MODEL_SIZE, "llm": bool(groq_client)}

def run_http():
    uvicorn.run(health_app, host="0.0.0.0", port=7860, log_level="warning")

# ---------------- Скачивание ----------------
def download_video(url: str) -> str:
    file_id = str(uuid.uuid4())
    out = os.path.join(DOWNLOAD_DIR, f"{file_id}.%(ext)s")
    cookies_src = "/app/youtube-cookies.txt"
    cookies_tmp = f"/tmp/cookies_{file_id}.txt"
    cookies = None
    if os.path.exists(cookies_src):
        shutil.copy(cookies_src, cookies_tmp)
        cookies = cookies_tmp
    opts = {
        "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "outtmpl": out,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "js_runtimes": {"node": {}},
        "remote_components": ["ejs:github"],
        "extractor_args": {"youtube": {"player_client": ["tv", "mweb", "android_vr"]}},
    }
    if cookies:
        opts["cookiefile"] = cookies
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)
    finally:
        if cookies and os.path.exists(cookies):
            try: os.unlink(cookies)
            except: pass

# ---------------- Транскрипция ----------------
def transcribe(path: str, language: str = "ru"):
    segments, info = whisper.transcribe(path, language=language or None, vad_filter=True, beam_size=1)
    cues = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]
    return cues, info.duration

# ---------------- ИИ-анализ ----------------
def llm_analyze_moments(cues, duration, clip_len, count):
    if not groq_client:
        return heuristic_analyze(cues, duration, clip_len, count)
    transcript = "\n".join(f"[{c['start']:.1f}-{c['end']:.1f}] {c['text']}" for c in cues)
    if len(transcript) > 12000:
        transcript = transcript[:12000] + "\n...(обрезано)"
    prompt = f"""Ты — эксперт по вирусному контенту для TikTok, Reels, YouTube Shorts.
Проанализируй транскрипт и найди {count} самых вирусных моментов.

Длительность видео: {duration:.0f} сек. Целевая длина клипа: {clip_len} сек (±10).

Транскрипт:
{transcript}

Критерии: хук в начале, эмоциональные пики, конфликт, цитата, завершённая мысль,
практическая ценность, интрига, неожиданный поворот.

Верни ТОЛЬКО JSON-объект: {{"moments": [{{"start": число, "end": число, "text": "описание", "score": 0-100}}]}}
Отсортируй по score. Не перекрывай моменты."""
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "Возвращаешь только валидный JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4, max_tokens=4000,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        arr = data.get("moments", []) if isinstance(data, dict) else data
        moments = []
        for m in arr:
            s = float(m.get("start", 0)); e = float(m.get("end", 0))
            if e <= s or s < 0 or e > duration + 1: continue
            moments.append({"start": s, "end": e, "text": m.get("text", ""), "score": float(m.get("score", 0))})
        moments.sort(key=lambda x: -x["score"])
        log.info(f"LLM вернул {len(moments)} моментов")
        return moments[:count] if moments else heuristic_analyze(cues, duration, clip_len, count)
    except Exception as e:
        log.warning(f"LLM упал: {e}")
        return heuristic_analyze(cues, duration, clip_len, count)

def heuristic_analyze(cues, duration, clip_len, count):
    wins = []
    if cues:
        for i in range(len(cues)):
            start = cues[i]["start"]; end = start; text = ""; j = i
            while j < len(cues) and (end - start) < clip_len + 8:
                text += (" " if text else "") + cues[j]["text"]
                end = cues[j]["end"]
                if (end - start) >= clip_len - 8 and cues[j]["text"].rstrip().endswith((".", "!", "?", "…")):
                    break
                j += 1
            L = end - start
            if L < clip_len - 8 or L > clip_len + 10: continue
            wins.append({"start": max(0, start - 0.15), "end": min(duration, end + 0.15), "text": text})
    if not wins:
        step = max(clip_len, duration / (count * 2))
        t = 0
        while t + clip_len <= duration:
            wins.append({"start": t, "end": min(duration, t + clip_len), "text": ""})
            t += step
    return wins[:count]

# ---------------- Общий энкодер: качество без потерь ----------------
def _encode_args(src, out_path, vf, af=None, clip_start=None, clip_dur=None):
    """Собирает команду ffmpeg с libx264 CRF 18 preset veryfast.
    Если libx264 недоступен — фолбэк на mpeg4 qscale 2 (максимальное качество)."""
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if clip_start is not None:
        base += ["-ss", str(clip_start)]
    if clip_dur is not None:
        base += ["-t", str(clip_dur)]
    base += ["-i", src]
    if vf:
        base += ["-vf", vf]
    if af:
        base += ["-af", af]

    enc_x264 = base + ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                       "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                       "-movflags", "+faststart", "-y", out_path]
    r = subprocess.run(enc_x264, capture_output=True, text=True)
    if r.returncode == 0:
        return
    # Фолбэк на mpeg4 с максимальным качеством
    log.warning("libx264 недоступен, фолбэк на mpeg4 qscale=2")
    enc_mp4 = base + ["-c:v", "mpeg4", "-qscale:v", "2", "-pix_fmt", "yuv420p",
                      "-c:a", "aac", "-b:a", "192k",
                      "-movflags", "+faststart", "-y", out_path]
    r2 = subprocess.run(enc_mp4, capture_output=True, text=True)
    if r2.returncode != 0:
        raise RuntimeError(f"ffmpeg: {r2.stderr[:500]}")

# ---------------- Рендер клипа (нарезка) ----------------
def render_clip(src, clip, out_path, cues):
    W, H = 1080, 1920  # HD вертикаль
    dur = clip["end"] - clip["start"]
    base = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}"
    draws = []
    def esc(t):
        return t.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:").replace("%", "\\%").replace(",", "\\,")
    for c in cues:
        s = max(0, c["start"] - clip["start"]); e = min(dur, c["end"] - clip["start"])
        if e <= 0 or s >= dur or not c["text"]: continue
        words = c["text"].split(); lines = []; cur = ""
        for w in words:
            if len((cur + " " + w).strip()) <= 22: cur = (cur + " " + w).strip()
            else:
                lines.append(cur); cur = w
        if cur: lines.append(cur)
        text = "\\n".join(lines[:2])
        draws.append(
            f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:"
            f"text='{esc(text)}':fontsize=64:fontcolor=white:borderw=6:bordercolor=black:"
            f"x=(w-text_w)/2:y=h-th-260:enable='between(t\\,{s:.2f}\\,{e:.2f})'"
        )
    vf = base + ("," + ",".join(draws) if draws else "")
    _encode_args(src, out_path, vf, clip_start=clip["start"], clip_dur=dur)

# ---------------- Уникализация (без скорости и громкости) ----------------
def uniquify_video(src, out_path, intensity="medium"):
    """
    Генерирует уникальную версию видео.
    Меняются ТОЛЬКО: яркость, контраст, насыщенность, гамма, шум, кроп.
    Скорость и громкость НЕ трогаем — качество и звук сохраняются.
    """
    if intensity == "light":
        b = random.uniform(0.02, 0.05); c = random.uniform(1.02, 1.05)
        s = random.uniform(1.02, 1.05); g = random.uniform(1.01, 1.03)
        noise = random.randint(1, 2)
    elif intensity == "strong":
        b = random.uniform(0.06, 0.10); c = random.uniform(1.08, 1.12)
        s = random.uniform(1.06, 1.12); g = random.uniform(1.05, 1.09)
        noise = random.randint(3, 5)
    else:  # medium
        b = random.uniform(0.04, 0.07); c = random.uniform(1.04, 1.08)
        s = random.uniform(1.04, 1.08); g = random.uniform(1.02, 1.05)
        noise = random.randint(2, 3)

    # Лёгкий кроп 2-4 px по краям — незаметен глазу, но меняет пиксели
    crop_x = random.randint(2, 4); crop_y = random.randint(2, 4)

    vf_parts = [
        f"crop=iw-{crop_x*2}:ih-{crop_y*2}:{crop_x}:{crop_y}",
        f"eq=brightness={b:.4f}:contrast={c:.4f}:saturation={s:.4f}:gamma={g:.4f}",
        f"noise=alls={noise}:allf=t",
    ]
    vf = ",".join(vf_parts)

    _encode_args(src, out_path, vf, af=None)
    return {"brightness": round(b,4), "contrast": round(c,4), "saturation": round(s,4),
            "gamma": round(g,4), "noise": noise, "crop": crop_x}

# ---------------- Бот ----------------
processing_lock = asyncio.Lock()

async def cmd_start(update, ctx):
    await update.message.reply_text(
        "🎬 *CLIPFORGE AI*\n\n"
        "Пришли ссылку на видео (YouTube/VK/Instagram/TikTok) "
        "или файл до 20 МБ.\n\n"
        "*Режимы:*\n"
        "✂️ /clip — нарезка на вирусные клипы (ИИ)\n"
        "🎨 /unique — уникализация (много версий)\n"
        "⚙️ /settings — настройки\n"
        "/help — помощь",
        parse_mode="Markdown",
    )

async def cmd_help(update, ctx):
    await update.message.reply_text(
        "ℹ️ *Как пользоваться*\n\n"
        "1. Пришли ссылку или видео\n"
        "2. Дождись обработки (3–10 мин)\n"
        "3. Получи клипы\n"
        "4. Тапни по видео → «Сохранить в галерею»\n\n"
        "*Нарезка:* ИИ (Llama 3.1) ищет вирусные моменты\n"
        "*Уникализация:* N разных версий (цвет, шум, кроп)\n"
        "Скорость и громкость НЕ меняются.",
        parse_mode="Markdown",
    )

user_settings = {}
def get_settings(uid):
    return user_settings.setdefault(uid, {
        "mode": "clip", "clip_len": 25, "count": 5, "language": "ru",
        "unique_count": 5, "unique_intensity": "medium",
    })

async def cmd_clip(update, ctx):
    s = get_settings(update.effective_user.id); s["mode"] = "clip"
    await update.message.reply_text("✂️ Режим: *нарезка*", parse_mode="Markdown")

async def cmd_unique(update, ctx):
    s = get_settings(update.effective_user.id); s["mode"] = "unique"
    await update.message.reply_text("🎨 Режим: *уникализация*", parse_mode="Markdown")

async def cmd_settings(update, ctx):
    uid = update.effective_user.id
    s = get_settings(uid)
    if s["mode"] == "clip":
        kb = [
            [InlineKeyboardButton(f"Длина: {s['clip_len']} сек", callback_data="set_len")],
            [InlineKeyboardButton(f"Клипов: {s['count']}", callback_data="set_count")],
            [InlineKeyboardButton(f"Язык: {s['language']}", callback_data="set_lang")],
            [InlineKeyboardButton("🎨 На уникализацию", callback_data="switch_unique")],
        ]
    else:
        kb = [
            [InlineKeyboardButton(f"Копий: {s['unique_count']}", callback_data="set_ucount")],
            [InlineKeyboardButton(f"Интенсивность: {s['unique_intensity']}", callback_data="set_uintensity")],
            [InlineKeyboardButton("✂️ На нарезку", callback_data="switch_clip")],
        ]
    await update.message.reply_text("⚙️ Настройки", reply_markup=InlineKeyboardMarkup(kb))

async def on_callback(update, ctx):
    q = update.callback_query; await q.answer()
    uid = q.from_user.id; s = get_settings(uid); d = q.data
    if d == "set_len":
        opts = [15, 25, 35, 50]
        i = opts.index(s["clip_len"]) if s["clip_len"] in opts else 1
        s["clip_len"] = opts[(i + 1) % len(opts)]
    elif d == "set_count":
        opts = [2, 3, 5, 10, 15, 20]
        i = opts.index(s["count"]) if s["count"] in opts else 2
        s["count"] = opts[(i + 1) % len(opts)]
    elif d == "set_lang":
        opts = ["ru", "en", "auto"]
        i = opts.index(s["language"]) if s["language"] in opts else 0
        s["language"] = opts[(i + 1) % len(opts)]
    elif d == "set_ucount":
        opts = [2, 3, 5, 10, 15, 20]
        i = opts.index(s["unique_count"]) if s["unique_count"] in opts else 2
        s["unique_count"] = opts[(i + 1) % len(opts)]
    elif d == "set_uintensity":
        opts = ["light", "medium", "strong"]
        i = opts.index(s["unique_intensity"]) if s["unique_intensity"] in opts else 1
        s["unique_intensity"] = opts[(i + 1) % len(opts)]
    elif d == "switch_unique":
        s["mode"] = "unique"
    elif d == "switch_clip":
        s["mode"] = "clip"
    if s["mode"] == "clip":
        kb = [
            [InlineKeyboardButton(f"Длина: {s['clip_len']} сек", callback_data="set_len")],
            [InlineKeyboardButton(f"Клипов: {s['count']}", callback_data="set_count")],
            [InlineKeyboardButton(f"Язык: {s['language']}", callback_data="set_lang")],
            [InlineKeyboardButton("🎨 На уникализацию", callback_data="switch_unique")],
        ]
    else:
        kb = [
            [InlineKeyboardButton(f"Копий: {s['unique_count']}", callback_data="set_ucount")],
            [InlineKeyboardButton(f"Интенсивность: {s['unique_intensity']}", callback_data="set_uintensity")],
            [InlineKeyboardButton("✂️ На нарезку", callback_data="switch_clip")],
        ]
    await q.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup(kb))

async def process_clips(update, ctx, path, label):
    uid = update.effective_user.id; s = get_settings(uid); chat = update.effective_chat
    async with processing_lock:
        status = await ctx.bot.send_message(chat.id, "🎙 Распознаю речь...")
        await ctx.bot.send_chat_action(chat.id, ChatAction.TYPING)
        try:
            cues, duration = await asyncio.to_thread(transcribe, path, s["language"])
        except Exception as e:
            await status.edit_text(f"❌ Ошибка распознавания: {e}"); return
        await status.edit_text(f"✅ {len(cues)} фраз. 🧠 ИИ ищет вирусные моменты...")
        clips = await asyncio.to_thread(llm_analyze_moments, cues, duration, s["clip_len"], s["count"])
        if not clips:
            await status.edit_text("❌ Моменты не найдены"); return
        await status.edit_text(f"🎬 Рендерю {len(clips)} клипов (HD 1080×1920)...")
        for i, clip in enumerate(clips, 1):
            out = os.path.join(DOWNLOAD_DIR, f"out_{uuid.uuid4()}.mp4")
            try:
                await asyncio.to_thread(render_clip, path, clip, out, cues)
                size = os.path.getsize(out) / 1048576
                score = clip.get("score", "")
                cap = f"CLIPFORGE {i}/{len(clips)}"
                if score: cap += f" · 🔥 {int(score)}/100"
                if clip.get("text"): cap += f"\n{clip['text'][:120]}"
                cap += "\nТапни → «Сохранить в галерею»"
                await ctx.bot.send_video(
                    chat_id=chat.id,
                    video=InputFile(open(out, "rb"), filename=f"CLIPFORGE_{i:02d}.mp4"),
                    supports_streaming=True, caption=cap, width=1080, height=1920,
                )
                await status.edit_text(f"🎬 {i}/{len(clips)} готово")
            except Exception as e:
                await ctx.bot.send_message(chat.id, f"⚠️ Клип {i}: {e}")
            finally:
                try: os.unlink(out)
                except: pass
        await status.edit_text(f"🎉 Готово! {len(clips)} клипов.")

async def process_unique(update, ctx, path, label):
    uid = update.effective_user.id; s = get_settings(uid); chat = update.effective_chat
    n = s["unique_count"]; intensity = s["unique_intensity"]
    async with processing_lock:
        await ctx.bot.send_message(chat.id, f"🎨 Генерирую {n} версий ({intensity})... Скорость и громкость не меняю.")
        status = await ctx.bot.send_message(chat.id, f"0/{n}")
        for i in range(1, n + 1):
            out = os.path.join(DOWNLOAD_DIR, f"uniq_{uuid.uuid4()}.mp4")
            try:
                params = await asyncio.to_thread(uniquify_video, path, out, intensity)
                size = os.path.getsize(out) / 1048576
                cap = (f"UNIQUE {i}/{n} · {intensity}\n"
                       f"Ярк {params['brightness']}, конт {params['contrast']}, "
                       f"насыщ {params['saturation']}, гамма {params['gamma']}, шум {params['noise']}\n"
                       f"Тапни → «Сохранить в галерею»")
                await ctx.bot.send_video(
                    chat_id=chat.id,
                    video=InputFile(open(out, "rb"), filename=f"UNIQUE_{i:02d}.mp4"),
                    supports_streaming=True, caption=cap,
                )
                await status.edit_text(f"🎨 {i}/{n} готово")
            except Exception as e:
                await ctx.bot.send_message(chat.id, f"⚠️ Версия {i}: {e}")
            finally:
                try: os.unlink(out)
                except: pass
        await status.edit_text(f"🎉 Готово! {n} уникальных версий.")

async def process_source(update, ctx, path, label):
    s = get_settings(update.effective_user.id)
    if s["mode"] == "unique":
        await process_unique(update, ctx, path, label)
    else:
        await process_clips(update, ctx, path, label)

async def on_link(update, ctx):
    url = update.message.text.strip()
    msg = await update.message.reply_text("📥 Скачиваю...")
    path = None
    try:
        path = await asyncio.to_thread(download_video, url)
        size = os.path.getsize(path) / 1048576
        await msg.edit_text(f"✅ Скачано ({size:.1f} МБ)")
        await process_source(update, ctx, path, "видео")
    except Exception as e:
        await msg.edit_text(f"❌ {str(e)[:300]}")
    finally:
        if path:
            try: os.unlink(path)
            except: pass

async def on_video(update, ctx):
    v = update.message.video or update.message.document
    if not v: return
    if v.file_size and v.file_size > 20 * 1024 * 1024:
        await update.message.reply_text("⚠️ Файл > 20 МБ. Bot API не примет — пришли ссылку."); return
    msg = await update.message.reply_text("📥 Скачиваю файл...")
    path = os.path.join(DOWNLOAD_DIR, f"in_{uuid.uuid4()}.mp4")
    try:
        f = await ctx.bot.get_file(v.file_id)
        await f.download_to_drive(path)
        await msg.edit_text("✅ Файл получен")
        await process_source(update, ctx, path, "видео")
    except Exception as e:
        await msg.edit_text(f"❌ {e}")
    finally:
        try: os.unlink(path)
        except: pass

async def on_text(update, ctx):
    t = update.message.text or ""
    if t.startswith(("http://", "https://")):
        await on_link(update, ctx)
    else:
        await update.message.reply_text("Пришли ссылку или видео. /help")

def main():
    threading.Thread(target=run_http, daemon=True).start()
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("settings", cmd_settings))
    app.add_handler(CommandHandler("clip", cmd_clip))
    app.add_handler(CommandHandler("unique", cmd_unique))
    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_handler(MessageHandler(filters.VIDEO | filters.Document.VIDEO, on_video))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    log.info("Бот запущен.")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
