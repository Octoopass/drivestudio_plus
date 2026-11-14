# ─────────────────────────────────────────────────────────────────────────────
# Lệnh hay dùng khi phát triển drive2sim (AGENTS.md mục 5). Gõ `make` để xem danh sách.
#
#   make setup        # tạo .venv nhẹ (không có PyTorch) và cài node_modules của Studio
#   make check        # format + lint + test, chạy trước khi commit
#   make studio       # bật Studio từ bản đã build (dist/)
#   make dev          # bật Studio + Vite tự tải lại khi sửa giao diện
#   make dev DEMO=1   # như trên nhưng trên dữ liệu mẫu (giả) ở .demo/; bỏ DEMO=1 là về dữ liệu thật
#   make studio CONDA_ENV=drive2sim  # Studio chạy bằng môi trường GPU (mode Render GPU, run trên máy)
#   make down         # dừng tiến trình đang giữ cổng của lần `studio`/`dev` gần nhất
#
# Quy tắc của file này:
#   1. Chỉ nghe ở cổng TRỐNG: dò lên từ API_PORT/WEB_PORT, in ra cổng chọn được và tiến trình
#      đang chiếm cổng nếu phải nhảy.
#   2. Ctrl+C hoặc `make down` chỉ dừng tiến trình đang giữ cổng của Studio, không đụng cổng khác.
#   3. Cổng của lần chạy gần nhất lưu ở .d2s-studio-ports.env (đã git-ignore) để `status`/`down`
#      biết cần dọn cổng nào.
#   4. Môi trường nặng (PyTorch, gsplat, ...) chưa có ở đây; cài theo README mục Installation.
# ─────────────────────────────────────────────────────────────────────────────

SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c
.ONESHELL:
.DEFAULT_GOAL := help

ROOT       := $(CURDIR)
VENV       := $(ROOT)/.venv
PY         := $(VENV)/bin/python
D2S        := $(VENV)/bin/d2s
# CONDA_ENV=drive2sim: chạy Studio (studio, dev, demo) bằng d2s của conda env có môi trường GPU
# (envs/main.yml, envs/install_main.sh), để mode Render GPU và run trên máy này dùng được GPU.
CONDA_ENV ?=
ifneq ($(CONDA_ENV),)
  CONDA_BASE := $(shell conda info --base 2>/dev/null)
  D2S := $(CONDA_BASE)/envs/$(CONDA_ENV)/bin/d2s
endif
FRONTEND   := $(ROOT)/drive2sim/studio/frontend
PORT_STATE := $(ROOT)/.d2s-studio-ports.env
LINT_PATHS := drive2sim tests
# Gói cho lần chạy test bằng Python 3.9 (bản thấp nhất mà pyproject.toml hỗ trợ).
PY39_DEPS  := --with "pydantic>=2.5,<3" --with "pyyaml>=6" --with "pytest>=7" \
              --with "fastapi>=0.110" --with "httpx>=0.24"

API_PORT  ?= 8000
WEB_PORT  ?= 5173
PORT_SCAN ?= 20
# Thư mục làm việc của Studio; để trống thì dùng D2S_WORKDIR hoặc gốc repo (AGENTS.md mục 6).
WORKDIR   ?=
# DEMO=1: chạy studio/dev trên dữ liệu mẫu ở DEMO_DIR (tạo một lần, giữ lại giữa các lần bật).
DEMO      ?= 0
DEMO_DIR  ?= $(ROOT)/.demo
DEMO_DEPS :=
ifeq ($(DEMO),1)
  override WORKDIR := $(DEMO_DIR)
  # Lệnh kaggle, colab giả đứng trước PATH: tab Kaggle, Colab của run mẫu không gửi gì đi.
  export PATH := $(DEMO_DIR)/.bin:$(PATH)
  # Khoá API nhập thử trên Studio mẫu được ghi vào đây, không đụng ~/.kaggle thật.
  export KAGGLE_CONFIG_DIR := $(DEMO_DIR)/.kaggle
  DEMO_DEPS := demo-data
endif

# Hàm shell dùng chung cho các target (make mở rộng $ thành $$).
define PORT_HELPERS
port_busy() { ss -ltnH "sport = :$$1" 2>/dev/null | grep -q .; }
port_owner() { fuser -n tcp "$$1" 2>/dev/null | xargs || true; }
port_free_from() {
  local port="$$1" tried=0
  while [ "$$tried" -lt "$(PORT_SCAN)" ]; do
    if ! port_busy "$$port"; then echo "$$port"; return 0; fi
    tried=$$((tried + 1)); port=$$((port + 1))
  done
  echo ""
}
free_port_or_die() {
  local label="$$1" start="$$2" chosen
  chosen=$$(port_free_from "$$start")
  if [ -z "$$chosen" ]; then
    echo "[!] Không còn cổng trống cho $$label trong $$start..$$((start + $(PORT_SCAN) - 1))." >&2
    echo "    Thử: make $(MAKECMDGOALS) API_PORT=9000 WEB_PORT=6000" >&2
    exit 1
  fi
  if [ "$$chosen" != "$$start" ]; then
    echo "[!] $$label: cổng $$start đang bị PID $$(port_owner "$$start") chiếm, dùng $$chosen." >&2
  fi
  echo "$$chosen"
}
kill_port() {
  local port="$$1"
  if ! port_busy "$$port"; then echo "[stop] cổng $$port đã trống."; return 0; fi
  echo "[stop] dừng cổng $$port (PID $$(port_owner "$$port"))"
  fuser -k -TERM -n tcp "$$port" >/dev/null 2>&1 || true
}
load_ports() {
  if [ -f "$(PORT_STATE)" ]; then . "$(PORT_STATE)"; else API_PORT="$(API_PORT)"; WEB_PORT="$(WEB_PORT)"; fi
}
endef

.PHONY: help setup test test-py39 lint format check types build studio dev demo demo-data down status clean

help:
	@echo "drive2sim — lệnh có sẵn:"
	@echo "  make setup       # tạo .venv và cài drive2sim[dev,studio,kaggle]; npm ci cho Studio"
	@echo "  make test        # chạy pytest bằng .venv"
	@echo "  make test-py39   # chạy pytest bằng Python 3.9 (qua uv)"
	@echo "  make lint        # ruff check"
	@echo "  make format      # ruff format"
	@echo "  make check       # format + lint + test (trước khi commit)"
	@echo "  make types       # sinh lại kiểu TypeScript từ drive2sim/core/schema.py"
	@echo "  make build       # kiểm tra kiểu và build giao diện vào dist/ (nhớ commit dist/)"
	@echo "  make studio      # bật Studio (tự tìm cổng trống)"
	@echo "  make dev         # bật Studio + Vite dev server (tự tải lại khi sửa giao diện)"
	@echo "  make demo        # = make studio DEMO=1: Studio trên dữ liệu mẫu (giả) ở .demo/"
	@echo "                   # DEMO=1 dùng được với studio và dev; FORCE=1 để tạo lại dữ liệu mẫu"
	@echo "                   # CONDA_ENV=drive2sim: chạy Studio bằng môi trường GPU (envs/install_main.sh)"
	@echo "  make down        # dừng tiến trình đang giữ cổng của Studio"
	@echo "  make status      # xem cổng nào đang chạy"
	@echo "  make clean       # xoá __pycache__, .pytest_cache, .ruff_cache"
	@echo ""
	@echo "  Tham số: API_PORT=$(API_PORT) WEB_PORT=$(WEB_PORT) WORKDIR=$(if $(WORKDIR),$(WORKDIR),<mặc định>) DEMO=$(DEMO)"

## setup: môi trường nhẹ để test và chạy Studio; đã có .venv thì chỉ cài lại gói.
setup:
	@[ -x "$(PY)" ] || uv venv "$(VENV)"
	uv pip install --python "$(PY)" -e ".[dev,studio,kaggle]"
	cd "$(FRONTEND)" && npm ci

test:
	@"$(PY)" -m pytest -q

test-py39:
	@uv run --python 3.9 --no-project $(PY39_DEPS) python -m pytest -q

lint:
	@uvx ruff check $(LINT_PATHS)

format:
	@uvx ruff format $(LINT_PATHS)

check: format lint test

## types: `npm run types` gọi lệnh `d2s` nên đưa .venv/bin lên PATH.
types:
	@cd "$(FRONTEND)" && PATH="$(VENV)/bin:$$PATH" npm run types

build:
	@cd "$(FRONTEND)" && npm run build

## studio: chạy bản đã build; Ctrl+C dừng server và giải phóng cổng.
studio: $(DEMO_DEPS)
	@$(PORT_HELPERS)
	[ -x "$(D2S)" ] || { echo "[!] Thiếu $(D2S); chạy: make setup"; exit 1; }
	api=$$(free_port_or_die "Studio" "$(API_PORT)")
	printf 'API_PORT=%s\nWEB_PORT=\n' "$$api" > "$(PORT_STATE)"
	echo ""
	echo "  Studio : http://127.0.0.1:$$api"
	echo "  Dừng   : Ctrl+C"
	echo ""
	exec "$(D2S)" studio --port "$$api" $(if $(WORKDIR),--workdir "$(WORKDIR)",)

## dev: Studio (API) + Vite; Vite chuyển /api và /files sang cổng Studio qua D2S_STUDIO_PORT.
dev: $(DEMO_DEPS)
	@$(PORT_HELPERS)
	[ -x "$(D2S)" ] || { echo "[!] Thiếu $(D2S); chạy: make setup"; exit 1; }
	[ -d "$(FRONTEND)/node_modules" ] || { echo "[!] Thiếu node_modules; chạy: make setup"; exit 1; }
	api=$$(free_port_or_die "Studio" "$(API_PORT)")
	web=$$(free_port_or_die "Vite" "$(WEB_PORT)")
	printf 'API_PORT=%s\nWEB_PORT=%s\n' "$$api" "$$web" > "$(PORT_STATE)"
	echo ""
	echo "  Giao diện : http://127.0.0.1:$$web   (tự tải lại khi sửa src/)"
	echo "  API       : http://127.0.0.1:$$api/api"
	echo "  Dừng      : Ctrl+C (dừng cả hai)"
	echo ""
	cleanup() {
	  trap - INT TERM EXIT
	  echo ""; echo "[stop] dừng Studio + Vite…"
	  kill -TERM $${API_PID:-} $${WEB_PID:-} 2>/dev/null || true
	  sleep 1
	  kill_port "$$web"; kill_port "$$api"
	  # Một `make dev` khác mở sau có thể đã ghi đè file cổng; chỉ xoá khi file còn là của lần này.
	  grep -qx "API_PORT=$$api" "$(PORT_STATE)" 2>/dev/null && rm -f "$(PORT_STATE)" || true
	  echo "[stop] xong."
	}
	trap cleanup INT TERM EXIT
	"$(D2S)" studio --port "$$api" $(if $(WORKDIR),--workdir "$(WORKDIR)",) &
	API_PID=$$!
	(cd "$(FRONTEND)" && D2S_STUDIO_PORT="$$api" exec npx vite --host 127.0.0.1 --port "$$web" --strictPort) &
	WEB_PID=$$!
	# Một trong hai dừng (lỗi hoặc Ctrl+C) thì dọn cả hai.
	wait -n "$$API_PID" "$$WEB_PID" || true

## demo-data: dữ liệu mẫu ở DEMO_DIR; chỉ tạo khi chưa có (FORCE=1 để tạo lại), kèm lệnh kaggle, colab giả.
demo-data:
	@[ -x "$(PY)" ] || { echo "[!] Thiếu $(PY); chạy: make setup"; exit 1; }
	if [ ! -f "$(DEMO_DIR)/DEMO" ] || [ "$(FORCE)" = "1" ]; then
	  "$(PY)" -m drive2sim.scripts.make_demo "$(DEMO_DIR)" --force
	fi
	mkdir -p "$(DEMO_DIR)/.bin"
	printf '#!/bin/sh\necho "dữ liệu mẫu: không gọi Kaggle thật" >&2\nexit 1\n' > "$(DEMO_DIR)/.bin/kaggle"
	chmod +x "$(DEMO_DIR)/.bin/kaggle"
	printf '#!/bin/sh\necho "dữ liệu mẫu: không gọi Colab thật" >&2\nexit 1\n' > "$(DEMO_DIR)/.bin/colab"
	chmod +x "$(DEMO_DIR)/.bin/colab"

demo:
	@$(MAKE) --no-print-directory studio DEMO=1

down:
	@$(PORT_HELPERS)
	load_ports
	[ -n "$${WEB_PORT:-}" ] && kill_port "$$WEB_PORT"
	kill_port "$$API_PORT"
	rm -f "$(PORT_STATE)"

status:
	@$(PORT_HELPERS)
	load_ports
	for entry in "Studio:$$API_PORT" "Vite:$${WEB_PORT:-}"; do
	  label=$${entry%%:*}; port=$${entry##*:}
	  [ -n "$$port" ] || continue
	  if port_busy "$$port"; then echo "[status] $$label :$$port ĐANG CHẠY (PID $$(port_owner "$$port"))"
	  else echo "[status] $$label :$$port trống"; fi
	done

clean:
	@find drive2sim tests -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
