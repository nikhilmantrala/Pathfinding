FROM python:3.10

RUN pip install --no-cache-dir tensorflow tensorflowjs

WORKDIR /workspace

ENTRYPOINT ["tensorflowjs_converter"]
