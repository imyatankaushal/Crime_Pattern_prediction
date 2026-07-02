# 🚔 Crime Pattern Prediction System

An end-to-end **Data Science & Machine Learning** project that predicts high-risk crime incidents, identifies geographic hotspots, and uncovers temporal crime patterns. Built with Python, scikit-learn, Pandas, and Streamlit.

---

## 🎯 Project Highlights (for Viva)

- **Problem:** Predict whether a crime incident is likely to be **high-risk**, identify **hotspot zones**, and analyze **temporal patterns**.
- **Dataset:** Realistic synthetic dataset (15,000 records) with spatial, temporal, weather, and contextual features.
- **Techniques:** EDA, Feature Engineering, Label Encoding, Standardization, Train/Test Split, Cross-Validation.
- **Models Trained & Compared:**
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - KMeans Clustering (Hotspot Detection)
- **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix.
- **Deliverables:** Jupyter Notebook, Trained Models, Visualizations, **Interactive Streamlit Dashboard**.

---

## 📂 Project Structure

```
crime-prediction/
├── data/                 # Dataset (CSV)
├── notebooks/            # Jupyter notebook (EDA + ML pipeline)
├── src/
│   ├── generate_data.py  # Synthetic data generator
│   └── train_models.py   # Full ML training pipeline
├── models/               # Saved .pkl models
├── reports/              # Generated plots & metrics
├── app/
│   └── streamlit_app.py  # Interactive web dashboard
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the dataset
```bash
python src/generate_data.py
```

### 3. Train all models & generate plots
```bash
python src/train_models.py
```

### 4. Open the Jupyter Notebook (for viva walkthrough)
```bash
jupyter notebook notebooks/Crime_Pattern_Prediction.ipynb
```

### 5. Launch the interactive demo dashboard ⭐
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Features in the Streamlit Dashboard

1. **Dashboard** — KPIs, crime distribution, district-wise breakdown, daily trend.
2. **Hotspots Map** — Interactive map showing KMeans hotspot zones.
3. **Predict** — Real-time risk prediction by selecting time, location, weather, etc.
4. **Insights** — Day-of-week × hour heatmap, hourly crime pattern.

---

## 🧠 Key Findings

- **Late-night (10 PM – 4 AM)** sees the highest concentration of burglary and vehicle theft.
- **Weekends** have higher assault and vandalism incidents.
- **Downtown & Central districts** are persistent hotspots.
- **Random Forest** and **Gradient Boosting** typically achieve **F1 > 0.85** and **ROC-AUC > 0.92**.

---

## 📝 Viva Talking Points

| Question | Answer |
|---|---|
| Why these features? | They capture spatial (district, lat/lon), temporal (hour, day, month), and contextual (weather, severity, prior incidents) drivers of crime. |
| Why multiple models? | To compare bias-variance tradeoffs and select the best performer. |
| Why KMeans? | Unsupervised clustering reveals natural geographic groupings of incidents — useful for patrol allocation. |
| Why ROC-AUC? | Threshold-independent metric robust to class imbalance. |
| How to deploy? | Streamlit dashboard + saved `.pkl` models can be containerized with Docker or deployed to Streamlit Cloud / Heroku. |

---

## 🔮 Future Enhancements
- Real city crime datasets (Chicago, LA, NYC open data portals)
- Deep Learning (LSTM) for time-series forecasting
- Geospatial DB (PostGIS) + Folium heatmaps
- REST API with FastAPI
- Mobile alerts for high-risk zones
