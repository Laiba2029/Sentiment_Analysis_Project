"""
Product Review Sentiment Analysis API
Uses a pre-trained BERT model (nlptown/bert-base-multilingual-uncased-sentiment)
fine-tuned on product reviews — perfect for e-commerce use case.
"""

import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import re
import time
import csv
import io

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

# ── CORS ─────────────────────────────────────────────────────────────────────
# In production, replace "*" with your actual frontend URL for security, e.g.:
#   allow_origins=["https://your-frontend.vercel.app"]
# For development and sharing/demos, "*" is fine.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
print("Loading BERT model... (first run downloads ~700MB)")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded successfully!")

class ReviewRequest(BaseModel):
    text: str

class SentimentResult(BaseModel):
    sentiment: str
    confidence: float
    star_rating: int
    probabilities: dict
    processing_time_ms: float

class BatchRequest(BaseModel):
    reviews: list[str]

def preprocess(text: str) -> str:
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = text.replace("n't", " not").replace("'re", " are").replace("'ve", " have")
    return text[:1000]

def stars_to_sentiment(star: int) -> str:
    if star <= 2:
        return "Negative"
    elif star == 3:
        return "Neutral"
    else:
        return "Positive"

def analyze_sentiment(text: str) -> SentimentResult:
    cleaned = preprocess(text)
    start = time.time()
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)[0]
    elapsed = (time.time() - start) * 1000
    star_probs = probs.tolist()
    predicted_star = int(torch.argmax(probs).item()) + 1
    neg_prob = star_probs[0] + star_probs[1]
    neu_prob = star_probs[2]
    pos_prob = star_probs[3] + star_probs[4]
    sentiment = stars_to_sentiment(predicted_star)
    conf_map = {"Negative": neg_prob, "Neutral": neu_prob, "Positive": pos_prob}
    confidence = conf_map[sentiment]
    return SentimentResult(
        sentiment=sentiment,
        confidence=round(confidence, 4),
        star_rating=predicted_star,
        probabilities={"Positive": round(pos_prob, 4), "Neutral": round(neu_prob, 4), "Negative": round(neg_prob, 4)},
        processing_time_ms=round(elapsed, 2)
    )

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running", "model": MODEL_NAME}

@app.post("/analyze", response_model=SentimentResult)
def analyze(req: ReviewRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
    if len(req.text) > 5000:
        raise HTTPException(status_code=400, detail="Review text too long (max 5000 chars).")
    return analyze_sentiment(req.text)

@app.post("/analyze/batch")
def analyze_batch(req: BatchRequest):
    if len(req.reviews) > 500:
        raise HTTPException(status_code=400, detail="Batch limit is 500 reviews.")
    results = []
    for review in req.reviews:
        if review.strip():
            results.append(analyze_sentiment(review).model_dump())
    return {"results": results, "count": len(results)}

def detect_review_col(headers):
    preferred = ["review", "text", "comment", "feedback", "description",
                 "reviews", "Review", "Text", "Comment", "Feedback"]
    for name in preferred:
        if name in headers:
            return name
    return headers[0]

@app.post("/analyze/csv")
async def analyze_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    content = await file.read()
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError:
        decoded = content.decode("latin-1")
    reader = csv.DictReader(io.StringIO(decoded))
    rows = list(reader)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV file is empty.")
    if len(rows) > 500:
        raise HTTPException(status_code=400, detail="CSV limit is 500 rows.")
    headers = list(rows[0].keys())
    review_col = detect_review_col(headers)
    results = []
    pos_count = neg_count = neu_count = 0
    total_conf = 0.0
    for i, row in enumerate(rows):
        text = row.get(review_col, "").strip()
        if not text:
            continue
        result = analyze_sentiment(text)
        r = result.model_dump()
        r["review_text"] = text
        r["row_number"] = i + 1
        for col in headers:
            if col != review_col:
                r[col] = row.get(col, "")
        results.append(r)
        if result.sentiment == "Positive": pos_count += 1
        elif result.sentiment == "Negative": neg_count += 1
        else: neu_count += 1
        total_conf += result.confidence
    total = len(results)
    avg_conf = round(total_conf / total, 4) if total > 0 else 0
    return {
        "results": results,
        "summary": {
            "total": total,
            "positive": pos_count,
            "negative": neg_count,
            "neutral": neu_count,
            "positive_pct": round(pos_count / total * 100, 1) if total else 0,
            "negative_pct": round(neg_count / total * 100, 1) if total else 0,
            "neutral_pct": round(neu_count / total * 100, 1) if total else 0,
            "avg_confidence": avg_conf,
            "detected_column": review_col,
            "filename": file.filename,
        }
    }

@app.post("/analyze/csv/download")
async def analyze_csv_download(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    content = await file.read()
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError:
        decoded = content.decode("latin-1")
    reader = csv.DictReader(io.StringIO(decoded))
    rows = list(reader)
    if not rows:
        raise HTTPException(status_code=400, detail="CSV file is empty.")
    if len(rows) > 500:
        raise HTTPException(status_code=400, detail="CSV limit is 500 rows.")
    headers = list(rows[0].keys())
    review_col = detect_review_col(headers)
    output = io.StringIO()
    out_headers = headers + ["sentiment", "confidence", "star_rating", "prob_positive", "prob_neutral", "prob_negative"]
    writer = csv.DictWriter(output, fieldnames=out_headers)
    writer.writeheader()
    for row in rows:
        text = row.get(review_col, "").strip()
        if not text:
            continue
        result = analyze_sentiment(text)
        row["sentiment"] = result.sentiment
        row["confidence"] = result.confidence
        row["star_rating"] = result.star_rating
        row["prob_positive"] = result.probabilities["Positive"]
        row["prob_neutral"] = result.probabilities["Neutral"]
        row["prob_negative"] = result.probabilities["Negative"]
        writer.writerow(row)
    output.seek(0)
    filename = file.filename.replace(".csv", "_analyzed.csv")
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}


# ── Entry point for deployment platforms ─────────────────────────────────────
# Railway and Render inject a PORT environment variable.
# Run locally:   uvicorn app:app --reload --port 8000
# Deployed auto: the platform uses this block via `python app.py`
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
