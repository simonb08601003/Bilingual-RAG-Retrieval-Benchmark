#!/usr/bin/env bash
# DRCD (Traditional Chinese) and SQuAD v1.1 (English) into ./data, plus two spaCy models.
set -euo pipefail
DATA_DIR="${DATA_DIR:-$(dirname "$0")/../data}"
mkdir -p "$DATA_DIR"
for f in DRCD_training.json DRCD_test.json; do
  curl -fsSL -o "$DATA_DIR/$f" "https://raw.githubusercontent.com/DRCKnowledgeTeam/DRCD/master/$f"
done
curl -fsSL -o "$DATA_DIR/squad-train-v1.1.json" https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json
curl -fsSL -o "$DATA_DIR/squad-dev-v1.1.json" https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/dev-v1.1.json
python -m pip install "https://github.com/explosion/spacy-models/releases/download/zh_core_web_md-3.8.0/zh_core_web_md-3.8.0-py3-none-any.whl" \
                      "https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl"
echo "Data saved to $DATA_DIR"
