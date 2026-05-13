#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="dataset/EPVME-Dataset"

if [[ -d "$TARGET_DIR" ]]; then
  echo "Target already exists: $TARGET_DIR"
  exit 0
fi

git clone https://github.com/sunknighteric/EPVME-Dataset "$TARGET_DIR"
