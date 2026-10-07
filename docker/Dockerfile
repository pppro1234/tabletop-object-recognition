FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN cmake -S . -B build && cmake --build build
RUN pip3 install --no-cache-dir fastapi uvicorn pydantic

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "service.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
