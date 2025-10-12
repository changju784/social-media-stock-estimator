# Makefile
preprocess:
	python src/preprocessing/merge_raw_json.py
	python src/preprocessing/clean_texts.py
	python src/preprocessing/tokenize_normalize.py
	python src/preprocessing/finbert_labeler.py
	python src/preprocessing/prepare_dataset.py
