FROM image2ppt-vision:0.2.1
ENV PIP_NO_INDEX=0 PIP_FIND_LINKS="" PIP_CONSTRAINT=""
RUN python -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
COPY sam3 /opt/sam3
RUN python -m pip install --no-cache-dir 'setuptools==70.3.0' /opt/sam3 einops pycocotools psutil
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
WORKDIR /src
ENTRYPOINT ["python", "-m", "image2ppt_vision.layered.sam3_worker"]
