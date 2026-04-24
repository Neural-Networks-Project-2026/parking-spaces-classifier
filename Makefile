.PHONY: fetch-data

fetch-data:
	mkdir -p data/raw
	kaggle datasets download -d ammarnassanalhajali/pklot-dataset -p data/raw --unzip
	@echo "data are now in data/raw"