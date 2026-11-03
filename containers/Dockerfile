# vbPET BIDS App, CPU image
#   docker build -f containers/Dockerfile -t vbpet:cpu .
#   docker run --rm -v /data/bids:/bids:ro -v /data/out:/out vbpet:cpu /bids /out participant ...
FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu
WORKDIR /opt/vbpet
COPY pyproject.toml README.md LICENSE ./
COPY vbpet ./vbpet
RUN pip install --upgrade pip && pip install .
ENTRYPOINT ["vbpet"]
