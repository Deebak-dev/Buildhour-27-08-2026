.PHONY: help setup ingest inspect ask retrieve recall run eval test clean

PYTHON ?= python
Q ?=

help:
	$(PYTHON) -c "print('setup   - install dependencies\ningest  - run Flow A: load, chunk, embed, store\ninspect - write readable dumps of chunks and embeddings to data/inspect\nask     - ask one question end to end: make ask Q=\"exit load on HDFC Large Cap\"\nretrieve- show the chunks retrieval returns for one query: make retrieve Q=\"...\"\nrecall  - measure recall@k over eval/golden_set.jsonl\nrun      - launch the Streamlit UI (Phase 8)\neval     - run golden + adversarial evaluation (Phase 9)\ntest     - run the test suite\nclean    - delete generated corpus and vector store')"

setup:
	$(PYTHON) -m pip install -r requirements.txt

ingest:
	$(PYTHON) -m ingest.run_ingest

inspect:
	$(PYTHON) -m ingest.inspect_chunks

ask:
	$(PYTHON) -m rag.pipeline --q "$(Q)"

retrieve:
	$(PYTHON) -m eval.inspect_query -q "$(Q)"

recall:
	$(PYTHON) -m eval.recall --verbose

run:
	$(PYTHON) -m streamlit run app.py

eval:
	$(PYTHON) -m eval.run_eval

test:
	$(PYTHON) -m pytest tests/ -q

clean:
	$(PYTHON) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ['data/chroma','data/raw','data/clean','data/chunks','data/inspect','logs']]"
