# ChangeForge —— 变化点检测工程框架
# 作者: 晨星
#
# Windows 若没有 make，直接照抄命令即可：
#   python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt
#   .venv/Scripts/python -m pytest tests -q
#   .venv/Scripts/python examples/run_demo.py

PY ?= python
VENV ?= .venv
BIN := $(VENV)/bin/$(PY)
ifeq ($(OS),Windows_NT)
	BIN := $(VENV)/Scripts/python.exe
endif

.PHONY: help venv install lint format check test demo clean docker-build

help: ## 显示可用目标
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "%-14s %s\n", $$1, $$2}'

venv: ## 创建虚拟环境
	$(PY) -m venv $(VENV)

install: ## 安装依赖（含 dev）
	$(BIN) -m pip install --upgrade pip
	$(BIN) -m pip install -r requirements.txt
	$(BIN) -m pip install -e .

lint: ## ruff 静态检查
	$(BIN) -m ruff check .

format: ## ruff 格式化
	$(BIN) -m ruff format .

check: lint ## 交付前收口：lint + 格式检查
	$(BIN) -m ruff format --check .

test: ## 跑测试
	$(BIN) -m pytest tests -q

demo: ## 端到端演示（落盘 artifacts/benchmark.json）
	$(BIN) examples/run_demo.py

docker-build: ## 构建镜像
	docker build -t changeforge:0.1.0 .

clean: ## 清理缓存与产物
	rm -rf .pytest_cache .ruff_cache artifacts build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
