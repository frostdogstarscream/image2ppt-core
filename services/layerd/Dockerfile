ARG BASE_IMAGE=image2ppt-vision:0.2.1
FROM ${BASE_IMAGE}
USER root
ENV PIP_NO_INDEX=0 PIP_CONSTRAINT="" PIP_FIND_LINKS="" PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY constraints.txt /opt/layerd/constraints.txt
COPY upstream /opt/layerd/upstream
RUN python -m pip install --index-url https://mirrors.aliyun.com/pypi/simple hatchling \
    && python -m pip install --no-build-isolation --index-url https://mirrors.aliyun.com/pypi/simple --constraint /opt/layerd/constraints.txt /opt/layerd/upstream \
    && python -m pip freeze > /opt/layerd/environment.lock
RUN python -m pip uninstall -y image2ppt-vision \
    && python -m pip check \
    && python -m pip freeze > /opt/layerd/environment.lock
COPY app.py /opt/layerd/app.py
COPY source-provenance.json /opt/layerd/source-provenance.json
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    LAMA_MODEL=/models/big-lama.pt LAYERD_MODELS=/models LAYERD_DATA=/data \
    LAYERD_TOKEN_FILE=/run/secrets/client-token LAYERD_PROCESS_SIZE=1024 \
    LAYERD_REVISION=21aef937a0371614adb4d961f52d02409cb8ecc7 OMP_NUM_THREADS=4
WORKDIR /opt/layerd
USER 10001:10001
ENTRYPOINT ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "18840", "--workers", "1", "--no-access-log"]
