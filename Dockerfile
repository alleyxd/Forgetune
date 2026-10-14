FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
WORKDIR /app
COPY pyproject.toml README.md ./
COPY forgetune ./forgetune
RUN pip install --no-cache-dir .
COPY examples ./examples
ENTRYPOINT ["forgetune"]
