FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libegl1 libglib2.0-0 libxcb1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt \
    && pip uninstall -y opencv-python && pip install opencv-python-headless \
    && pip cache purge || true
COPY app ./app
COPY tests ./tests
COPY pytest.ini .
EXPOSE 5032
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s CMD curl -fs http://localhost:5032/health || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5032", "--log-level", "info"]
