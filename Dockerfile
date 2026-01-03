FROM python:3.13.5-slim-bookworm
WORKDIR /code

RUN pip install uv

COPY ["pyproject.toml", "uv.lock", "scripts/predict.py", "scripts/preprocess.py", "model", "./"]

RUN uv sync --frozen --no-dev
ENV PATH="/code/.venv/bin:$PATH"

EXPOSE 9696
ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]