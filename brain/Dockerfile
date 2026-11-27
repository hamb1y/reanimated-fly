FROM python:3.12-slim
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 DATA_DIR=/data CACHE_DIR=/cache TORCH_THREADS=8 OMP_NUM_THREADS=8 \
    BRAIN_DEVICE=auto TRITON_CACHE_DIR=/cache/triton
WORKDIR /srv
# gcc: Triton compiles its small launcher stub with the system C compiler on first use
RUN apt-get update && apt-get install -y --no-install-recommends gcc libc6-dev && rm -rf /var/lib/apt/lists/*
# One image for GPU and CPU hosts: the CUDA wheel (nvidia-* libs and triton come with it) falls back to the
# CPU path when no GPU is visible. CPU-only tag: --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install torch --index-url ${TORCH_INDEX}
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app app
COPY tests tests
COPY pytest.ini .
EXPOSE 5031
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5031/health',timeout=4).status==200 else 1)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5031", "--log-level", "warning"]
