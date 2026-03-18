#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR=".venv"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Не найден интерпретатор $PYTHON_BIN. Установите Python 3."
  read -r
  exit 1
fi

echo "Введите запрос:"
read -r QUERY

if [ -z "${QUERY}" ]; then
  echo "Пустой запрос. Нажмите Enter для выхода."
  read -r
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

if ! python -c "import yaml, trafilatura, sentence_transformers" >/dev/null 2>&1; then
  python -m pip install --upgrade pip
  python -m pip install -e '.[full]'
fi

PYTHONPATH=src python -m production_rag_pipeline.cli "$QUERY" --mode llm

echo
echo "Готово. Нажмите Enter для выхода."
read -r
