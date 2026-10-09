FROM python:3.11-slim

# Устанавливаем ffmpeg, git и Node.js для работы POT-провайдера
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Клонируем и собираем POT-провайдер
RUN git clone --single-branch --branch 2.0.1 https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git /opt/bgutil-pot \
    && cd /opt/bgutil-pot/server \
    && npm ci \
    && npx tsc

WORKDIR /app

COPY requirements.txt .
# Добавляем плагин для yt-dlp, который будет использовать наш POT-провайдер
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir bgutil-ytdlp-pot-provider

COPY app.py .
COPY index.html .
COPY worker.js .

# Запускаем и POT-провайдер, и основной сервер
CMD node /opt/bgutil-pot/server/build/main.js --port 4416 & \
    uvicorn app:app --host 0.0.0.0 --port 7860
