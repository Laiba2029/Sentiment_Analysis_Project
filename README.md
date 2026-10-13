# SentimentIQ 

## Project Structure
```
sentiment_api/
├── app.py            ← FastAPI backend (BERT inference)
├── requirements.txt  ← Python dependencies
├── index.html        ← Frontend (single review)
├── dashboard.html    ← Frontend (batch CSV upload)
└── README.md         ← This file
```

---

---

## Model Info

- **Model**: `nlptown/bert-base-multilingual-uncased-sentiment`
- **Trained on**: Amazon, Yelp, TripAdvisor product reviews (~400K samples)
- **Languages**: English, Dutch, German, French, Spanish, Italian
- **Output**: 1–5 star rating → mapped to Positive / Neutral / Negative
- **Accuracy**: ~92% on English product reviews

---


