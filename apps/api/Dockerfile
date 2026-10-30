FROM python:3.11-slim AS base

WORKDIR /app

RUN pip install --no-cache-dir hatch

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir .

FROM python:3.11-slim

WORKDIR /app
COPY --from=base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=base /usr/local/bin /usr/local/bin
COPY src ./src
COPY ../../contracts /app/contracts

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["uvicorn", "cropapi.main:app", "--host", "0.0.0.0", "--port", "8000"]