# syntax=docker/dockerfile:1
#
# Multi-arch image. Builds for linux/arm64 (Raspberry Pi 4/5, 64-bit OS) as well
# as amd64. numpy/pandas ship aarch64 wheels, so nothing is compiled here — which
# is what keeps a Pi build to a couple of minutes instead of half an hour.
#
#   docker build -t tqt .
#   docker compose up -d
#
# Requires a 64-bit Raspberry Pi OS. On 32-bit (armv7) there are no pandas wheels
# and the build will try to compile from source; use the systemd unit in deploy/
# with the system Python instead.

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Seoul

# tzdata matters: every schedule in this bot is expressed in KST, and curl is
# used by the container healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata curl \
 && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the dependency manifest first so `pip install` is cached across code edits.
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

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
