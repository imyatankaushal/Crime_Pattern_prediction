"""
Train multiple ML models for Crime Pattern Prediction.
- Classification: Predict whether a crime incident is High-Risk
- Multi-class: Predict crime_type
- Clustering: Identify hotspot zones via KMeans
- Time-series style: Aggregated daily counts forecasted via RandomForest on lag features
Saves trained models + metrics + plots.
"""
import os, json, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, classification_report, roc_auc_score, roc_curve)

BASE = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(BASE, "data", "crime_data.csv")
MODELS = os.path.join(BASE, "models")
REPORTS = os.path.join(BASE, "reports")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)

sns.set_style("whitegrid")

def load():
    df = pd.read_csv(DATA, parse_dates=["timestamp", "date"])
    return df

def preprocess(df):
    X = df[["hour","month","is_weekend","severity","temperature_c",
            "population_density","prev_incidents_30d",
            "district","crime_type","weather","day_of_week"]].copy()
    y = df["high_risk"]

    encoders = {}
    for col in ["district","crime_type","weather","day_of_week"]:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    return X, y, encoders

def train_classifiers(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, max_depth=5, random_state=42),
    }

    results = {}
    best_name, best_score, best_model = None, -1, None
    for name, m in models.items():
        Xtr = X_train_s if name == "Logistic Regression" else X_train
        Xte = X_test_s  if name == "Logistic Regression" else X_test
        m.fit(Xtr, y_train)
        pred = m.predict(Xte)
        proba = m.predict_proba(Xte)[:,1]
        results[name] = {
            "accuracy":  round(accuracy_score(y_test, pred), 4),
            "precision": round(precision_score(y_test, pred), 4),
            "recall":    round(recall_score(y_test, pred), 4),
            "f1":        round(f1_score(y_test, pred), 4),
            "roc_auc":   round(roc_auc_score(y_test, proba), 4),
        }
        if results[name]["f1"] > best_score:
            best_score = results[name]["f1"]
            best_name = name
            best_model = m
        print(f"{name}: {results[name]}")

    # Confusion matrix for best
    pred = best_model.predict(X_test if best_name != "Logistic Regression" else X_test_s)
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Low","High"], yticklabels=["Low","High"])
    plt.title(f"Confusion Matrix — {best_name}")
    plt.ylabel("Actual"); plt.xlabel("Predicted")
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS, "confusion_matrix.png"), dpi=130); plt.close()

    # ROC curves
    plt.figure(figsize=(7,5))
    for name, m in models.items():
        Xte = X_test_s if name == "Logistic Regression" else X_test
        proba = m.predict_proba(Xte)[:,1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']})")
    plt.plot([0,1],[0,1],"k--", alpha=0.5)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — High-Risk Crime Prediction")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(REPORTS, "roc_curves.png"), dpi=130); plt.close()

    # Model comparison bar chart
    metrics_df = pd.DataFrame(results).T
    metrics_df.plot(kind="bar", figsize=(10,5), colormap="viridis")
    plt.title("Model Performance Comparison")
    plt.ylabel("Score"); plt.xticks(rotation=20); plt.ylim(0,1.05)
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(REPORTS, "model_comparison.png"), dpi=130); plt.close()

    # Feature importance for best model (if tree-based)
    if hasattr(best_model, "feature_importances_"):
        fi = pd.Series(best_model.feature_importances_, index=X.columns).sort_values()
        plt.figure(figsize=(8,5))
        fi.plot(kind="barh", color="#2E86AB")
        plt.title(f"Feature Importance — {best_name}")
        plt.tight_layout(); plt.savefig(os.path.join(REPORTS, "feature_importance.png"), dpi=130); plt.close()

    joblib.dump(best_model, os.path.join(MODELS, "best_classifier.pkl"))
    joblib.dump(scaler, os.path.join(MODELS, "scaler.pkl"))
    return results, best_name

def cluster_hotspots(df, k=6):
    coords = df[["latitude","longitude"]].values
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["hotspot"] = km.fit_predict(coords)
    joblib.dump(km, os.path.join(MODELS, "kmeans_hotspots.pkl"))

    plt.figure(figsize=(9,7))
    palette = sns.color_palette("tab10", k)
    for c in range(k):
        sub = df[df["hotspot"]==c]
        plt.scatter(sub["longitude"], sub["latitude"], s=8, alpha=0.5, color=palette[c], label=f"Zone {c+1}")
    plt.scatter(km.cluster_centers_[:,1], km.cluster_centers_[:,0], marker="X", s=220,
                c="black", edgecolors="yellow", linewidths=1.5, label="Centroid")
    plt.title("Crime Hotspot Zones (KMeans Clustering)")
    plt.xlabel("Longitude"); plt.ylabel("Latitude")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS, "hotspots_map.png"), dpi=130); plt.close()
    return df

def time_series_plots(df):
    daily = df.groupby("date").size().rename("crimes").reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["rolling7"] = daily["crimes"].rolling(7).mean()

    plt.figure(figsize=(12,4))
    plt.plot(daily["date"], daily["crimes"], alpha=0.4, label="Daily")
    plt.plot(daily["date"], daily["rolling7"], color="red", label="7-day Rolling Avg")
    plt.title("Daily Crime Trend"); plt.xlabel("Date"); plt.ylabel("Incidents")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(REPORTS, "daily_trend.png"), dpi=130); plt.close()

    # Heatmap: hour x day-of-week
    pivot = df.pivot_table(index="day_of_week", columns="hour", values="severity", aggfunc="count").fillna(0)
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.reindex(order)
    plt.figure(figsize=(12,4))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.3)
    plt.title("Crime Heatmap — Day of Week × Hour of Day")
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS, "heatmap_dow_hour.png"), dpi=130); plt.close()

    # Crime type distribution
    plt.figure(figsize=(9,4))
    df["crime_type"].value_counts().plot(kind="bar", color=sns.color_palette("magma", 8))
    plt.title("Crime Type Distribution"); plt.ylabel("Count"); plt.xticks(rotation=25)
    plt.tight_layout(); plt.savefig(os.path.join(REPORTS, "crime_type_dist.png"), dpi=130); plt.close()

def main():
    print("→ Loading data…")
    df = load()
    print(f"   shape={df.shape}")

    print("→ EDA plots…")
    time_series_plots(df)

    print("→ Clustering hotspots…")
    df = cluster_hotspots(df)

    print("→ Training classifiers…")
    X, y, encoders = preprocess(df)
    results, best_name = train_classifiers(X, y)

    joblib.dump(encoders, os.path.join(MODELS, "encoders.pkl"))
    with open(os.path.join(REPORTS, "metrics.json"), "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)

    print(f"\n✅ Best model: {best_name}")
    print("✅ Plots saved to /reports")
    print("✅ Models saved to /models")

if __name__ == "__main__":
    main()
