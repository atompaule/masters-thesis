#!/bin/sh

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <script_path> [script_args...]"
  exit 1
fi

SCRIPT_PATH="$1"
shift

python -m debugpy --connect n-hpc-login1:5678 --wait-for-client "$SCRIPT_PATH" "$@"
