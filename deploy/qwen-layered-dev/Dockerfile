FROM image2ppt-sam3:20261030
ENV PIP_NO_INDEX=0 PIP_FIND_LINKS="" PIP_CONSTRAINT=""
RUN python -m pip install --no-cache-dir --index-url https://pypi.org/simple 'diffusers==0.36.0' 'transformers==4.57.3' 'accelerate==1.12.0' 'huggingface-hub==0.36.0' 'fastapi==0.115.12' 'uvicorn==0.34.2' 'python-multipart==0.0.20' 'sentencepiece==0.2.1' 'protobuf==6.33.2'
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
WORKDIR /app
ENTRYPOINT ["python", "-u", "/app/server.py"]
