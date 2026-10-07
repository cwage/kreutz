FROM python:3.12-slim

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# The GPU search (search._stack). The CUDA 12.6 build, because the host's
# NVIDIA driver supports CUDA 12.9 and PyPI's default wheel wants 13.
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126

COPY kreutz ./kreutz

ENV KREUTZ_DATA=/data

ENTRYPOINT ["python", "-u", "-m"]
