FROM node:22-bookworm-slim

ENV NODE_ENV=production \
    PYTHONUNBUFFERED=1 \
    PYTHON_BIN=/opt/venv/bin/python \
    MODEL_CACHE_DIR=/tmp/spamshield-model

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir "numpy==1.26.4" "tflite-runtime==2.14.0"

WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev

COPY . .

EXPOSE 10000
CMD ["npm", "start"]