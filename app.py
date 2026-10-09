def download_video(url: str) -> str:
    file_id = str(uuid.uuid4())
    out_tmpl = os.path.join(DOWNLOAD_DIR, f"{file_id}.%(ext)s")
    ydl_opts = {
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]",
        "outtmpl": out_tmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        # --- Новые параметры для обхода защиты ---
        "extractor_args": {
            "youtube": {
                "player_client": ["tv", "mweb", "android_vr"],
                "player_skip": ["webpage", "configs"],
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
            raise HTTPException(status_code=400, detail=f"yt-dlp: {e}")
