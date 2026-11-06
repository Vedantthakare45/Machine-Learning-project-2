## Day 14 - Sentiment Analysis with Naïve Bayes (Streamlit)

**Goal:** classify IMDb movie reviews as positive/negative using a Multinomial Naïve Bayes model
with TF-IDF features, exposed through a Streamlit app.

- Implementation: [`DAY - 14/sentiment_nb.py`](DAY%20-%2014/sentiment_nb.py) (training + `predict`)
- App: [`DAY - 14/app.py`](DAY%20-%2014/app.py)
- Details and run instructions: [`DAY - 14/README.md`](DAY%20-%2014/README.md)

Quick start:
```bash
cd "DAY - 14"
pip install -r requirements.txt
python sentiment_nb.py train
streamlit run app.py
```
