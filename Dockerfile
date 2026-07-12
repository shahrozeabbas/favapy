FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

CMD ["python", "-c", "import favapy; print(favapy.__version__)"]
