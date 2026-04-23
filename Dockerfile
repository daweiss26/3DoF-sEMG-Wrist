FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/src:/app/src/controller:/app/src/util

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cargo \
    git \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libusb-1.0-0 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
COPY orbita3d_control ./orbita3d_control

RUN pip install --upgrade pip setuptools wheel maturin && \
    pip install -r requirements.txt && \
    pip install ./orbita3d_control/orbita3d_c_api

COPY . .

CMD ["python", "src/orbita_abh_camera.py"]
