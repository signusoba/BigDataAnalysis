.PHONY: serve test corpus train
serve:
	bash scripts/run.sh
test:
	python -m pytest
corpus:
	python -m persian_ocr.corpus --output data/letters
train:
	python -m persian_ocr.training --data data/letters --output artifacts --epochs 12 --models cnn hog
