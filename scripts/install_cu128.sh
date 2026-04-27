#!/bin/bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel
"${PYTHON_BIN}" -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
"${PYTHON_BIN}" -m pip install -r requirements.txt
"${PYTHON_BIN}" -m deepspeed.env_report || true
