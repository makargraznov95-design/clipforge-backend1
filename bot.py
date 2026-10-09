import os, asyncio, subprocess, tempfile, random
from aiogram import Bot, Dispatcher, F
from aiogram.types import (
    Message, FSInputFile, InlineKeyboardMarkup, InlineKeyboardButton,
    CallbackQuery, BotCommand
)
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from faster_whisper import WhisperModel
import yt_dlp

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise SystemExit("Нужен TELEGRAM_BOT_TOKEN в переменных окружения")

MODEL_SIZE = os.getenv("WHISPER_MODEL", "tiny")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

bot = Bot(BOT_TOKEN)
dp = Dispatcher()

print(f"Загружаю Whisper {MODEL_SIZE}...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8", cpu_threads=2)
print("Whisper готов")


class Setup(StatesGroup):
    clip_count = State()
    clip_length = State()
    uniq_count = State()
    uniq_intensity = State()


# ---------- Скачивание ----------

def download_video(url: str, outdir: str) -> str:
    out = os.path.join(outdir, "src.%(ext)s")
    opts = {
        "format": "best[height<=720]/best/bv*+ba/b",
        "outtmpl": out,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "js_runtimes": ["node"],
        "extractor_args": {"youtube": {"player_client": ["tv", "mweb", "android_vr"]}},
    }
    cookies = "/app/youtube-cookies.txt"
    if os.path.exists(cookies):
        opts["cookiefile"] = cookies
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


# ---------- Транскрипция ----------

def transcribe(path: str):
    segments, _ = model.transcribe(path, vad_filter=True, beam_size=1)
    return [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments]


def score_text(t: str) -> float:
    """Насколько фраза подходит на 'цепляющий' момент."""
    l = t.lower()
    s = 0.0
    if "!" in t or "?" in t:
        s += 0.6
    if any(w in l for w in ["почему", "как", "что", "когда", "зачем", "кто", "где"]):
        s += 0.5
    if any(w in l for w in ["главное", "представь", "интересн", "смешн", "удивительн", "секрет", "шок", "правда"]):
        s += 0.8
    if any(w in l for w in ["ха-ха", "хаха", "смех", "лол", "прикол", "ржу"]):
        s += 0.6
    if any(w in l for w in ["смотри", "внимание", "запомни", "важно", "сейчас будет"]):
        s += 0.4
    return s


def pick_clips(cues, total, count, min_len, max_len):
    """Выбор моментов без перекрытия."""
    wins = []

    # 1) По транскрипту — законченные фразы
    if cues:
        for i in range(len(cues)):
            start = cues[i]["start"]
            end = start
            text = ""
            j = i
            while j < len(cues) and (end - start) < max_len:
                text += (" " if text else "") + cues[j]["text"]
                end = cues[j]["end"]
                if (end - start) >= min_len and cues[j]["text"].strip().endswith((".", "!", "?", "…")):
                    break
                j += 1
            length = end - start
            if min_len <= length <= max_len + 3:
                wins.append({
                    "start": max(0.0, start - 0.15),
                    "end": min(total, end + 0.15),
                    "score": score_text(text),
                    "text": text,
                })

    # 2) Если транскрипт пустой — равномерная сетка
    if not wins:
        step = max(min_len, total / (count * 2))
        t = 0.0
        while t + min_len <= total:
            wins.append({
                "start": t,
                "end": min(total, t + min_len),
                "score": 0.1,
                "text": "",
            })
            t += step

    # 3) Выбираем топ без перекрытия > 40%
    wins.sort(key=lambda w: -w["score"])
    picked = []
    for w in wins:
        if len(picked) >= count:
            break
        ok = True
        for p in picked:
            o = max(0.0, min(p["end"], w["end"]) - max(p["start"], w["start"]))
            u = min(p["end"] - p["start"], w["end"] - w["start"])
            if u and o / u > 0.4:
                ok = False
                break
        if ok:
            picked.append(w)

    picked.sort(key=lambda w: w["start"])
    return picked


# ---------- Рендер клипа с субтитрами ----------

def esc(t: str) -> str:
    for a, b in [("\\", "\\\\"), ("'", "\\'"), (":", "\\:"), (",", "\\,"), ("%", "\\%")]:
        t = t.replace(a, b)
    return t


SUBTITLE_STYLES = {
    "bold":    {"fs": 0.062, "color": "white",   "bw": 5, "bc": "black",       "y": 0.15},
    "yellow":  {"fs": 0.067, "color": "#ffea00", "bw": 5, "bc": "#12002b",     "y": 0.14},
    "minimal": {"fs": 0.052, "color": "white",   "bw": 2, "bc": "black@0.6",   "y": 0.12},
}


def render_clip(src, out, clip, cues, style="bold", W=1080, H=1920):
    dur = clip["end"] - clip["start"]
    st = SUBTITLE_STYLES.get(style, SUBTITLE_STYLES["bold"])
    draws = []

    for c in cues:
        s = max(0.0, c["start"] - clip["start"])
        e = min(dur, c["end"] - clip["start"])
        if e <= 0 or s >= dur or not c["text"]:
            continue
        words = c["text"].split()
        lines, cur = [], ""
        for w in words:
            if len((cur + " " + w).strip()) <= 22:
                cur = (cur + " " + w).strip()
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        text = esc("\\n".join(lines[:2]))
        fs = int(W * st["fs"])
        y = f"h-th-{int(H * st['y'])}"
        draws.append(
            f"drawtext=fontfile={FONT}:text='{text}':fontsize={fs}:"
            f"fontcolor={st['color']}:borderw={st['bw']}:bordercolor={st['bc']}:"
            f"x=(w-text_w)/2:y={y}:"
            f"enable='between(t\\,{s:.2f}\\,{e:.2f})'"
        )

    vf = f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H}"
    if draws:
        vf += "," + ",".join(draws)

    subprocess.run([
        "ffmpeg", "-hide_banner", "-y",
        "-ss", f"{clip['start']}", "-t", f"{dur}", "-i", src,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        out,
    ], check=True, capture_output=True)


# ---------- Уникализатор ----------

def render_unique(src, out, intensity):
    """
    Только цветокоррекция. Не меняет: звук, скорость, зеркало, кроп.
    Качество сохраняется: crf 18, preset medium.
    intensity: 1 (лёгкая) / 2 (средняя) / 3 (заметная)
    """
    if intensity == 1:
        ranges = {"b": 0.03, "c": 0.06, "s": 0.06, "g": 0.03}
    elif intensity == 3:
        ranges = {"b": 0.10, "c": 0.18, "s": 0.18, "g": 0.10}
    else:
        ranges = {"b": 0.06, "c": 0.12, "s": 0.12, "g": 0.06}

    brightness = round(random.uniform(-ranges["b"], ranges["b"]), 3)
    contrast   = round(1.0 + random.uniform(-ranges["c"], ranges["c"]), 3)
    saturation = round(1.0 + random.uniform(-ranges["s"], ranges["s"]), 3)
    gamma      = round(1.0 + random.uniform(-ranges["g"], ranges["g"]), 3)

    eq = (f"eq=brightness={brightness}:"
          f"contrast={contrast}:"
          f"saturation={saturation}:"
          f"gamma={gamma}")

    subprocess.run([
        "ffmpeg", "-hide_banner", "-y", "-i", src,
        "-vf", eq,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        out,
    ], check=True, capture_output=True)

    return {
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "gamma": gamma,
    }


# ---------- Утилиты ----------

async def get_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", path],
        check=True, capture_output=True, text=True
    )
    return float(r.stdout.strip())


# ---------- Хендлеры ----------

@dp.message(Command("start"))
async def cmd_start(msg: Message):
    await msg.answer(
        "👋 CLIPFORGE AI\n\n"
        "Что умею:\n\n"
        "🎬 <b>AI-нарезка</b> — распознаю речь, найду лучшие моменты, "
        "сделаю вертикальные клипы 9:16 со встроенными субтитрами.\n\n"
        "🎨 <b>Уникализатор</b> — сделаю N разных версий одного видео, "
        "меняя только яркость/контраст/насыщенность. Звук, скорость, "
        "зеркало и качество остаются прежними.\n\n"
        "Команды:\n"
        "/clip — AI-нарезка\n"
        "/uniq — Уникализатор\n"
        "/help — справка\n\n"
        "Работаю со ссылками YouTube/VK/Instagram/TikTok и с видео из галереи.",
        parse_mode="HTML",
    )


@dp.message(Command("help"))
async def cmd_help(msg: Message):
    await msg.answer(
        "📖 <b>Как пользоваться</b>\n\n"
        "1. /clip — нарезка клипов\n"
        "2. Выбери количество (2–20)\n"
        "3. Длину клипа (10–180 сек)\n"
        "4. Стиль субтитров\n"
        "5. Пришли ссылку или видео\n\n"
        "Или /uniq — уникализация:\n"
        "1. Количество версий (2–20)\n"
        "2. Интенсивность эффекта\n"
        "3. Пришли ссылку или видео\n\n"
        "⚠️ Telegram ограничивает видео 50 МБ. "
        "Если клип больше — он будет пропущен с предупреждением.",
        parse_mode="HTML",
    )


@dp.message(Command("clip"))
async def cmd_clip(msg: Message, state: FSMContext):
    await state.update_data(mode="clip")
    await msg.answer("🎬 Сколько клипов сделать? Введи число от 2 до 20:")
    await state.set_state(Setup.clip_count)


@dp.message(Command("uniq"))
async def cmd_uniq(msg: Message, state: FSMContext):
    await state.update_data(mode="uniq")
    await msg.answer("🎨 Сколько уникальных версий? Введи число от 2 до 20:")
    await state.set_state(Setup.uniq_count)


@dp.message(Setup.clip_count)
async def set_clip_count(msg: Message, state: FSMContext):
    try:
        n = int(msg.text.strip())
        if not 2 <= n <= 20:
            raise ValueError
    except Exception:
        await msg.answer("❌ Введи целое число от 2 до 20.")
        return
    await state.update_data(clip_count=n)
    await msg.answer("📏 Длина клипа в секундах (10–180):")
    await state.set_state(Setup.clip_length)


@dp.message(Setup.clip_length)
async def set_clip_length(msg: Message, state: FSMContext):
    try:
        n = int(msg.text.strip())
        if not 10 <= n <= 180:
            raise ValueError
    except Exception:
        await msg.answer("❌ Введи целое число от 10 до 180.")
        return
    await state.update_data(clip_len=n)
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="Bold — жирный с обводкой", callback_data="style_bold")],
        [InlineKeyboardButton(text="Yellow — жёлтый контрастный", callback_data="style_yellow")],
        [InlineKeyboardButton(text="Minimal — тонкий", callback_data="style_minimal")],
    ])
    await msg.answer("✨ Стиль субтитров:", reply_markup=kb)


@dp.callback_query(F.data.startswith("style_"))
async def set_style(cb: CallbackQuery, state: FSMContext):
    style = cb.data.replace("style_", "")
    await state.update_data(style=style)
    d = await state.get_data()
    await cb.message.edit_text(
        f"✅ <b>Настройки AI-нарезки</b>\n"
        f"• Клипов: {d.get('clip_count')}\n"
        f"• Длина: {d.get('clip_len')} сек\n"
        f"• Стиль: {style}\n\n"
        f"Теперь пришли ссылку на YouTube/VK или видео из галереи.",
        parse_mode="HTML",
    )
    await state.set_state(None)
    await cb.answer()


@dp.message(Setup.uniq_count)
async def set_uniq_count(msg: Message, state: FSMContext):
    try:
        n = int(msg.text.strip())
        if not 2 <= n <= 20:
            raise ValueError
    except Exception:
        await msg.answer("❌ Введи целое число от 2 до 20.")
        return
    await state.update_data(uniq_count=n)
    kb = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="1 — лёгкая", callback_data="int_1")],
        [InlineKeyboardButton(text="2 — средняя", callback_data="int_2")],
        [InlineKeyboardButton(text="3 — заметная", callback_data="int_3")],
    ])
    await msg.answer("🎛 Интенсивность эффекта:", reply_markup=kb)


@dp.callback_query(F.data.startswith("int_"))
async def set_intensity(cb: CallbackQuery, state: FSMContext):
    intensity = int(cb.data.replace("int_", ""))
    await state.update_data(uniq_intensity=intensity)
    d = await state.get_data()
    await cb.message.edit_text(
        f"✅ <b>Настройки Уникализатора</b>\n"
        f"• Версий: {d.get('uniq_count')}\n"
        f"• Интенсивность: {intensity}\n\n"
        f"Каждая версия получит свою комбинацию цветокоррекции. "
        f"Звук, скорость, зеркало и качество не меняются.\n\n"
        f"Пришли ссылку или видео.",
        parse_mode="HTML",
    )
    await state.set_state(None)
    await cb.answer()


# ---------- Приём ссылки ----------

@dp.message(F.text.startswith("http"))
async def handle_link(msg: Message, state: FSMContext):
    d = await state.get_data()
    if not d.get("mode"):
        await msg.answer("Сначала выбери режим: /clip или /uniq")
        return
    status = await msg.answer("⏳ Скачиваю видео...")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            src = download_video(msg.text.strip(), tmp)
            await _process(msg, status, src, tmp, d)
        except Exception as e:
            await status.edit_text(f"❌ Ошибка скачивания: {e}")


# ---------- Приём файла из галереи ----------

@dp.message(F.video | F.document)
async def handle_file(msg: Message, state: FSMContext):
    d = await state.get_data()
    if not d.get("mode"):
        await msg.answer("Сначала выбери режим: /clip или /uniq")
        return
    status = await msg.answer("⏳ Получаю файл...")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            file_obj = msg.video or msg.document
            path = os.path.join(tmp, "src.mp4")
            await bot.download(file_obj, destination=path)
            await _process(msg, status, path, tmp, d)
        except Exception as e:
            await status.edit_text(f"❌ Ошибка: {e}")


# ---------- Основная обработка ----------

async def _process(msg: Message, status, src, tmp, d):
    mode = d.get("mode")

    if mode == "clip":
        count = d.get("clip_count", 5)
        clip_len = d.get("clip_len", 25)
        style = d.get("style", "bold")

        await status.edit_text("🎙 Распознаю речь (это может занять несколько минут)...")
        cues = transcribe(src)

        total = await get_duration(src)
        min_len = max(8, clip_len - 8)
        max_len = clip_len + 8
        clips = pick_clips(cues, total, count, min_len, max_len)

        await status.edit_text(
            f"🎬 Нашёл {len(clips)} моментов. Рендерю..."
        )

        done = 0
        for i, clip in enumerate(clips, 1):
            out = os.path.join(tmp, f"CLIPFORGE_{i:02d}.mp4")
            try:
                await status.edit_text(
                    f"🎬 Рендерю {i}/{len(clips)}...\n"
                    f"Фрагмент {clip['start']:.0f}–{clip['end']:.0f} сек"
                )
                render_clip(src, out, clip, cues, style)
            except subprocess.CalledProcessError as e:
                await msg.answer(f"❌ Клип {i} не удалось отрендерить: ffmpeg error")
                continue

            size_mb = os.path.getsize(out) / 1048576
            if size_mb > 49:
                await msg.answer(
                    f"⚠️ Клип {i} ({clip['start']:.0f}–{clip['end']:.0f} сек) весит "
                    f"{size_mb:.1f} МБ — превышает лимит Telegram 50 МБ. Пропускаю."
                )
                continue

            await msg.answer_video(
                FSInputFile(out, filename=f"CLIPFORGE_{i:02d}.mp4"),
                caption=(
                    f"🎬 CLIPFORGE #{i:02d}\n"
                    f"Фрагмент: {clip['start']:.0f}–{clip['end']:.0f} сек "
                    f"({clip['end'] - clip['start']:.0f} сек)\n"
                    f"Размер: {size_mb:.1f} МБ · Стиль: {style}"
                ),
            )
            done += 1
            await status.edit_text(f"✅ Готово {done}/{len(clips)} клипов...")

        if done == 0:
            await status.edit_text("❌ Ни один клип не удалось создать.")
        else:
            await status.edit_text(f"🎉 Готово! Создано {done} клипов.")

    elif mode == "uniq":
        count = d.get("uniq_count", 5)
        intensity = d.get("uniq_intensity", 2)

        await status.edit_text(f"🎨 Создаю {count} уникальных версий...")
        done = 0
        seen = set()

        for i in range(1, count + 1):
            out = os.path.join(tmp, f"UNIQUE_{i:02d}.mp4")

            # пробуем подобрать непохожие параметры
            params = None
            for _try in range(5):
                p = render_unique(src, out, intensity) if False else None
                # сначала сгенерируем значения, потом рендерим
                if intensity == 1:
                    ranges = {"b": 0.03, "c": 0.06, "s": 0.06, "g": 0.03}
                elif intensity == 3:
                    ranges = {"b": 0.10, "c": 0.18, "s": 0.18, "g": 0.10}
                else:
                    ranges = {"b": 0.06, "c": 0.12, "s": 0.12, "g": 0.06}

                cand = (
                    round(random.uniform(-ranges["b"], ranges["b"]), 3),
                    round(1.0 + random.uniform(-ranges["c"], ranges["c"]), 3),
                    round(1.0 + random.uniform(-ranges["s"], ranges["s"]), 3),
                    round(1.0 + random.uniform(-ranges["g"], ranges["g"]), 3),
                )
                # минимум 0.03 разброса от уже сделанных
                if all(sum(abs(a - b) for a, b in zip(cand, s)) > 0.05 for s in seen):
                    params = cand
                    break
            if params is None:
                params = cand
            seen.add(params)

            brightness, contrast, saturation, gamma = params
            eq = (f"eq=brightness={brightness}:contrast={contrast}:"
                  f"saturation={saturation}:gamma={gamma}")

            await status.edit_text(
                f"🎨 Рендерю версию {i}/{count}...\n"
                f"яркость {brightness:+.3f} · контраст {contrast:.3f}"
            )

            try:
                subprocess.run([
                    "ffmpeg", "-hide_banner", "-y", "-i", src,
                    "-vf", eq,
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p",
                    "-c:a", "copy",
                    "-movflags", "+faststart",
                    out,
                ], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                await msg.answer(f"❌ Версия {i} не удалась.")
                continue

            size_mb = os.path.getsize(out) / 1048576
            if size_mb > 49:
                await msg.answer(
                    f"⚠️ Версия {i} весит {size_mb:.1f} МБ — превышает лимит Telegram 50 МБ. Пропускаю."
                )
                continue

            await msg.answer_video(
                FSInputFile(out, filename=f"UNIQUE_{i:02d}.mp4"),
                caption=(
                    f"🎨 UNIQUE #{i:02d}\n"
                    f"Яркость: {brightness:+.3f}\n"
                    f"Контраст: {contrast:.3f}\n"
                    f"Насыщенность: {saturation:.3f}\n"
                    f"Гамма: {gamma:.3f}\n"
                    f"Размер: {size_mb:.1f} МБ"
                ),
            )
            done += 1
            await status.edit_text(f"✅ Готово {done}/{count} версий...")

        if done == 0:
            await status.edit_text("❌ Ни одна версия не создана.")
        else:
            await status.edit_text(f"🎉 Готово! Создано {done} уникальных версий.")


# ---------- Запуск ----------

async def set_commands():
    await bot.set_my_commands([
        BotCommand(command="start", description="Главное меню"),
        BotCommand(command="clip", description="AI-нарезка клипов"),
        BotCommand(command="uniq", description="Уникализатор видео"),
        BotCommand(command="help", description="Справка"),
    ])


async def main():
    await set_commands()
    print("Бот запущен. Жду сообщений.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
