#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d ".venv" ]; then
  source ".venv/bin/activate"
fi

echo "Введите запрос:"
read -r QUERY

if [ -z "${QUERY}" ]; then
  echo "Пустой запрос. Нажмите Enter для выхода."
  read -r
  exit 1
fi

PYTHONPATH=src python -m production_rag_pipeline.cli "$QUERY" --mode llm

echo
echo "Готово. Нажмите Enter для выхода."
read -r
