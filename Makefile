.PHONY: summarize smoke clean

summarize:           ## Summarize all 120 evaluated tech news articles; write predictions + metrics
	python3 summarize.py

smoke:               ## CI target: summarize 3 articles via fixture; verifies pipeline runs
	ARTICLES_PATH=data/tiny_articles_smoke.csv \
	  REFERENCES_PATH=data/tiny_refs_smoke.csv \
	  OUTPUT_PATH=summary_predictions_smoke.csv \
	  python3 summarize.py

clean:               ## Remove generated outputs
	rm -f summary_predictions.csv summary_predictions_smoke.csv summary_metrics.json
