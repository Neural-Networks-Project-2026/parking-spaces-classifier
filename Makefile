.PHONY: fetch-data

fetch-data:
	@echo "Pobieranie datasetu PKLot z Kaggle..."
	mkdir -p data/raw
	kaggle datasets download -d ammarnassanalhajali/pklot-dataset -p data/raw --unzip
	@echo "Dane pobrane i rozpakowane w data/raw!"