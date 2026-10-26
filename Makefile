export PYTHONPATH := src:app
.PHONY: data eval analyze test serve
data:
	bash scripts/download_data.sh
eval:
	python scripts/run_eval.py zh en
analyze:
	python scripts/analyze.py
test:
	python -m pytest -q tests
serve:
	uvicorn main:app --port 8000
