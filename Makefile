check:
	ruff check . && pytest
data:
	./scripts/download_data.sh
train: data
	PYTHONPATH=src python -m roadscan.train
evaluate:
	PYTHONPATH=src python -m roadscan.evaluate
serve:
	PYTHONPATH=src uvicorn roadscan.api:app --reload
