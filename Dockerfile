# syntax=docker/dockerfile:1
#
# Multi-arch image built with uv. Works on linux/arm64 (Raspberry Pi 4/5, 64-bit
# OS) as well as amd64. numpy/pandas ship aarch64 wheels, so nothing is compiled
# here — which keeps a Pi build to a couple of minutes instead of half an hour.
#
#   docker build -t tqt .
#   docker compose up -d
#
# Requires a 64-bit Raspberry Pi OS. On 32-bit (armv7) there are no pandas wheels
# and the build would compile from source; use the systemd unit in deploy/ instead.

FROM python:3.12-slim-bookworm

# uv comes from its official distroless image — no curl-pipe-sh in the build.
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /usr/local/bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Seoul \
    # Copy rather than hardlink: the cache and the venv can be on different
    # layers/filesystems, where hardlinking fails.
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    # Pin the venv location so it is predictable on PATH below.
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Put the project venv on PATH. This is what lets `python ...`, `pytest` and
# `tqt` run directly inside the container with no `uv run` prefix — so
# docker-compose commands stay plain.
ENV PATH="/app/.venv/bin:$PATH"

# tzdata matters: every schedule in this bot is expressed in KST. curl is used
# by the container healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata curl \
 && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- dependency layer -------------------------------------------------------
# Install dependencies before copying source, so editing code doesn't invalidate
# the (slow) dependency layer. `--locked` fails the build if uv.lock is stale
# rather than silently resolving something different from what was tested.
# `--no-default-groups` leaves pytest/ruff out of the runtime image.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-default-groups

# --- project layer ----------------------------------------------------------
COPY README.md ./
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups

COPY config/ ./config/

# Run as a non-root user; the data volume is chowned to it.
RUN useradd --create-home --uid 10001 tqt \
 && mkdir -p /app/data \
 && chown -R tqt:tqt /app
USER tqt

VOLUME ["/app/data"]
EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8000/healthz || exit 1

# Paper mode by default. Live requires TQT_MODE=live *and*
# TQT_LIVE_CONFIRM=I_UNDERSTAND_REAL_MONEY, both set explicitly by you.
ENV TQT_MODE=paper \
    TQT_DB_PATH=/app/data/tqt.db

ENTRYPOINT ["tqt"]
CMD ["run", "--dashboard"]
