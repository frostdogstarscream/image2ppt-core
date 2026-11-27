FROM image2ppt-layerd:20261104-r4
USER root
ENV PIP_NO_INDEX=0 PIP_CONSTRAINT="" PIP_FIND_LINKS="" PIP_CONFIG_FILE=/dev/null \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN python -m pip install --index-url https://mirrors.aliyun.com/pypi/simple \
      diffusers==0.37.0 transformers==4.57.3 accelerate==1.12.0 peft==0.18.0 \
      'gguf>=0.10,<0.20' sentencepiece \
    && python -m pip check \
    && python -m pip freeze > /opt/layer-experiments-environment.lock
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    DIFFUSERS_GGUF_CUDA_KERNELS=false OMP_NUM_THREADS=8
WORKDIR /workspace
USER 10001:10001
ENTRYPOINT ["python", "-u"]
