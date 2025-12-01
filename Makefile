# ==========================================
# SOCIAL MEDIA STOCK ESTIMATOR PIPELINE
# ==========================================

RAW_DIR = data/raw/v2
INTERIM_DIR = data/interim
PROCESSED_DIR = data/processed

# Final output check files
MERGED = $(INTERIM_DIR)/merged_reddit.csv
CLEANED = $(INTERIM_DIR)/cleaned_texts.csv
TOKENIZED = $(INTERIM_DIR)/tokenized_texts.csv
LABELED = $(INTERIM_DIR)/labeled_texts.csv
AUGMENTED = $(INTERIM_DIR)/labeled_texts_with_prices.csv

# ==========================================
# Run everything
# ==========================================

all: merge_raw clean_text tokenize sentiment augment prepare
	@echo "\n🎉 ALL PIPELINE STEPS COMPLETED SUCCESSFULLY!"

# ==========================================
# Steps
# ==========================================

merge_raw:
	@echo "\n[1/6] 🔄 Merging raw Reddit files..."
	python -m src.preprocessing.merge_raw_json

clean_text:
	@echo "\n[2/6] 🧹 Cleaning Reddit texts..."
	python -m src.preprocessing.clean_texts
	python -m src.preprocessing.patch_labeled_tickers

tokenize:
	@echo "\n[3/6] ✂ Tokenizing texts..."
	python -m src.preprocessing.tokenize_normalize

sentiment:
	@echo "\n[4/6] 🔍 Running FinBERT sentiment labeling..."
	python -m src.preprocessing.finbert_labeler

augment:
	@echo "\n[5/6] 💹 Augmenting with yfinance prices..."
	python -m src.preprocessing.augment_with_yfinance \
		--input $(INTERIM_DIR)/labeled_texts.csv \
		--output $(INTERIM_DIR)/labeled_texts_with_prices.csv

prepare:
	@echo "\n[6/6] 📦 Preparing final train/val/test splits..."
	python -m src.preprocessing.prepare_dataset

# ==========================================
# Cleanup
# ==========================================

clean:
	rm -f $(INTERIM_DIR)/*.csv
	rm -f $(PROCESSED_DIR)/*.csv

.PHONY: all merge_raw clean_text tokenize sentiment augment prepare clean
