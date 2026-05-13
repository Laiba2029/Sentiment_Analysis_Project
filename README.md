# SentimentIQ — Setup & Deployment Guide

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

## Running Locally

### Step 1 — Install Python dependencies

Make sure you have Python 3.9+ installed, then:

```bash
pip install -r requirements.txt
```

> First run downloads the BERT model (~700MB). Subsequent runs load from cache.

### Step 2 — Start the backend

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
Loading BERT model...
Model loaded successfully!
INFO: Uvicorn running on http://0.0.0.0:8000
```

### Step 3 — Open the frontend

Open `index.html` in your browser (no server needed):

```bash
# Windows
start index.html

# Mac
open index.html

# Linux
xdg-open index.html
```

---

## Deploying to the Internet (Share a Link)

To share the project with your supervisor, you need to deploy the **backend** online.
The frontend HTML files can then be opened locally or also hosted for free.

---

### Option A — Railway (Recommended, Free, Easiest)

1. **Push your code to GitHub** (create a new repo and push all files)

2. **Go to [railway.app](https://railway.app)** → Sign up with GitHub → Click **New Project**

3. **Select "Deploy from GitHub repo"** → pick your repository

4. Railway auto-detects Python. Set the **Start Command** in Settings:
   ```
   python app.py
   ```

5. Railway will give you a public URL like:
   ```
   https://sentimentiq-production.up.railway.app
   ```

6. **Update `index.html`** — open it and change line near the top of the `<script>`:
   ```javascript
   const API_URL = "https://sentimentiq-production.up.railway.app";
   ```
   Do the same in `dashboard.html`.

7. Now share your `index.html` file directly, or host it on GitHub Pages (see below).

---

### Option B — Render (Free, Slightly Slower Cold Starts)

1. Push code to GitHub

2. Go to [render.com](https://render.com) → New → **Web Service**

3. Connect your GitHub repo

4. Fill in:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Environment**: Python 3

5. Click Deploy. You'll get a URL like:
   ```
   https://sentimentiq.onrender.com
   ```

6. Update `API_URL` in `index.html` and `dashboard.html` as above.

> **Note**: Render free tier spins down after 15 minutes of inactivity. First request after idle takes ~30 seconds. This is normal.

---

### Hosting the Frontend (So You Can Share a Full Link)

Once your backend is deployed, host the HTML files on **GitHub Pages** (free):

1. Push `index.html` and `dashboard.html` to a GitHub repo (make sure `API_URL` is updated first)
2. Go to repo **Settings → Pages**
3. Source: **Deploy from branch → main → / (root)**
4. Your site will be at: `https://yourusername.github.io/your-repo-name/`

Share that link — your supervisor can open it in any browser with no setup.

---

## API Reference

### POST /analyze
```json
Request:  { "text": "This product is amazing!" }
Response: {
  "sentiment": "Positive",
  "confidence": 0.9231,
  "star_rating": 5,
  "probabilities": { "Positive": 0.92, "Neutral": 0.05, "Negative": 0.03 },
  "processing_time_ms": 142.5
}
```

### POST /analyze/batch
```json
Request:  { "reviews": ["Great!", "Terrible.", "It's okay."] }
Response: { "results": [...], "count": 3 }
```

Used internally by the frontend for sentence-by-sentence mixed-sentiment breakdown.

---

## Model Info

- **Model**: `nlptown/bert-base-multilingual-uncased-sentiment`
- **Trained on**: Amazon, Yelp, TripAdvisor product reviews (~400K samples)
- **Languages**: English, Dutch, German, French, Spanish, Italian
- **Output**: 1–5 star rating → mapped to Positive / Neutral / Negative
- **Accuracy**: ~92% on English product reviews

---

## Troubleshooting

| Problem | Fix |
|---|---|
| CORS error in browser | Backend is not running or `API_URL` is wrong |
| Model download hangs | Check internet connection (~700MB needed) |
| Out of memory | Model needs ~1.5GB RAM minimum |
| Slow inference | Normal on CPU (~150ms). GPU reduces to ~20ms |
| Render cold start slow | Free tier idles — first request takes ~30s, then normal |
