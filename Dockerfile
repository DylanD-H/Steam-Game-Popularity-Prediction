FROM python:3.13.5-slim-bookworm
WORKDIR /code

RUN pip install uv

COPY ["pyproject.toml", "uv.lock", "./"]
COPY ["scripts", "./scripts"]
COPY ["model", "./model"]

RUN python -m venv .venv && \
    .venv/bin/python -m ensurepip --upgrade

RUN uv sync
RUN ./.venv/bin/python -m pip install --upgrade pip
RUN ./.venv/bin/python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

ENV PATH="/code/.venv/bin:$PATH"

EXPOSE 9696
ENTRYPOINT ["uvicorn", "scripts.predict:app", "--host", "0.0.0.0", "--port", "9696"]