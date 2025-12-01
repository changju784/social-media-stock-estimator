# ================================
# SOCIAL MEDIA STOCK ESTIMATOR PIPELINE
# ================================

# Directories
RAW_DIR = data/raw
INTERIM_DIR = data/interim
PROCESSED_DIR = data/processed

# Files
MERGED = $(INTERIM_DIR)/merged_reddit.csv
CLEANED = $(INTERIM_DIR)/cleaned_texts.csv
TOKENIZED = $(INTERIM_DIR)/tokenized_texts.csv
LABELED = $(INTERIM_DIR)/labeled_texts.csv
AUGMENTED = $(INTERIM_DIR)/labeled_with_prices.csv
TRAIN = $(PROCESSED_DIR)/reddit_train.csv
VAL = $(PROCESSED_DIR)/reddit_val.csv
TEST = $(PROCESSED_DIR)/reddit_test.csv

# ================================
# Main Targets
# ================================

# Run full end-to-end pipeline
all: merge_raw_json clean_texts tokenize sentiment augment_prices prepare_dataset
	@echo "\n🎉 FULL PIPELINE COMPLETED SUCCESSFULLY!"

# ================================
# Steps
# ================================

merge_raw_json: $(MERGED)
$(MERGED):
	@echo "\n[1/6] 🔄 Merging raw Reddit JSON files..."
	mkdir -p $(INTERIM_DIR)
	python src/preprocessing/merge_raw_json.py

clean_texts: $(CLEANED)
$(CLEANED): $(MERGED)
	@echo "\n[2/6] 🧹 Cleaning & normalizing text..."
	python src/preprocessing/clean_texts.py

tokenize: $(TOKENIZED)
$(TOKENIZED): $(CLEANED)
	@echo "\n[3/6] ✂ Tokenizing texts..."
	python src/preprocessing/tokenize_normalize.py

sentiment: $(LABELED)
$(LABELED): $(CLEANED)
	@echo "\n[4/6] 🔍 Generating FinBERT sentiment labels..."
	python src/preprocessing/finbert_labeler.py

augment_prices: $(AUGMENTED)
$(AUGMENTED): $(LABELED)
	@echo "\n[5/6] 💹 Augmenting with Yahoo Finance prices..."
	python src/preprocessing/augment_with_yfinance.py \
		--input $(LABELED) \
		--output $(AUGMENTED)

prepare_dataset: $(TRAIN) $(VAL) $(TEST)
$(TRAIN) $(VAL) $(TEST): $(AUGMENTED)
	@echo "\n[6/6] 📦 Preparing final train/val/test splits..."
	mkdir -p $(PROCESSED_DIR)
	python src/preprocessing/prepare_dataset.py

# ================================
# Utility Targets
# ================================

clean:
	@echo "Cleaning intermediate files..."
	rm -f $(INTERIM_DIR)/*.csv
	rm -f $(PROCESSED_DIR)/*.csv

.PHONY: all merge_raw_json clean_texts tokenize sentiment augment_prices prepare_dataset clean
   