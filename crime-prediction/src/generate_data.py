"""
Generate a realistic synthetic crime dataset for the Crime Pattern Prediction System.
Simulates crimes across a city with spatial, temporal, and contextual patterns.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "crime_data.csv")

DISTRICTS = [
    ("Central",   28.6139, 77.2090, 1.6),
    ("North",     28.7041, 77.1025, 1.1),
    ("South",     28.5355, 77.2490, 0.9),
    ("East",      28.6300, 77.3000, 1.3),
    ("West",      28.6280, 77.0800, 1.0),
    ("Downtown",  28.6500, 77.2300, 1.8),
    ("Suburb-1",  28.5800, 77.1500, 0.6),
    ("Suburb-2",  28.7200, 77.2500, 0.7),
]

CRIME_TYPES = ["Theft", "Assault", "Burglary", "Robbery", "Vandalism", "Fraud", "Drug Offense", "Vehicle Theft"]
CRIME_BASE_PROB = np.array([0.28, 0.14, 0.13, 0.10, 0.11, 0.09, 0.08, 0.07])

WEATHER = ["Clear", "Rainy", "Foggy", "Cloudy", "Hot"]

def generate(n=15000):
    rows = []
    start = datetime(2022, 1, 1)
    end = datetime(2024, 12, 31)
    total_days = (end - start).days

    for _ in range(n):
        # Temporal patterns: more crimes at night & weekends
        day_offset = np.random.randint(0, total_days)
        date = start + timedelta(days=day_offset)
        hour_weights = np.array([3,2,2,1,1,1,2,3,4,5,5,5,6,6,6,7,8,9,10,11,10,9,7,5], dtype=float)
        hour = np.random.choice(24, p=hour_weights / hour_weights.sum())
        minute = np.random.randint(0, 60)
        ts = date.replace(hour=hour, minute=minute)

        is_weekend = ts.weekday() >= 5

        # District selection weighted by activity
        d_weights = np.array([d[3] for d in DISTRICTS])
        d_idx = np.random.choice(len(DISTRICTS), p=d_weights / d_weights.sum())
        name, lat, lon, _ = DISTRICTS[d_idx]
        latitude = lat + np.random.normal(0, 0.012)
        longitude = lon + np.random.normal(0, 0.012)

        # Crime type influenced by hour & district
        probs = CRIME_BASE_PROB.copy()
        if 22 <= hour or hour <= 4:
            probs[CRIME_TYPES.index("Burglary")] *= 1.8
            probs[CRIME_TYPES.index("Vehicle Theft")] *= 1.7
            probs[CRIME_TYPES.index("Assault")] *= 1.4
        if 9 <= hour <= 17:
            probs[CRIME_TYPES.index("Fraud")] *= 1.6
            probs[CRIME_TYPES.index("Theft")] *= 1.3
        if is_weekend:
            probs[CRIME_TYPES.index("Assault")] *= 1.3
            probs[CRIME_TYPES.index("Vandalism")] *= 1.4
        probs /= probs.sum()
        crime_type = np.random.choice(CRIME_TYPES, p=probs)

        # Weather + temperature
        weather = np.random.choice(WEATHER, p=[0.45, 0.15, 0.10, 0.20, 0.10])
        month = ts.month
        base_temp = 15 + 12 * np.sin((month - 4) * np.pi / 6)
        temperature = round(base_temp + np.random.normal(0, 4), 1)

        # Severity (1-5) — depends on type
        sev_map = {"Theft":2,"Assault":4,"Burglary":3,"Robbery":4,"Vandalism":2,"Fraud":3,"Drug Offense":3,"Vehicle Theft":3}
        severity = max(1, min(5, sev_map[crime_type] + np.random.choice([-1,0,0,1])))

        # Population density & previous incidents (synthetic features)
        pop_density = int(np.random.normal(8000, 2000) * (1 + 0.1 * d_weights[d_idx]))
        prev_incidents_30d = max(0, int(np.random.normal(20 * d_weights[d_idx], 8)))

        # Risk label (target for classification): High if severity>=4 or burglary/robbery at night
        risk_high = int(severity >= 4 or (crime_type in ["Burglary","Robbery","Vehicle Theft"] and (hour>=22 or hour<=4)))

        rows.append({
            "timestamp": ts,
            "date": ts.date(),
            "hour": hour,
            "day_of_week": ts.strftime("%A"),
            "month": ts.month,
            "year": ts.year,
            "is_weekend": int(is_weekend),
            "district": name,
            "latitude": round(latitude, 6),
            "longitude": round(longitude, 6),
            "crime_type": crime_type,
            "severity": severity,
            "weather": weather,
            "temperature_c": temperature,
            "population_density": pop_density,
            "prev_incidents_30d": prev_incidents_30d,
            "high_risk": risk_high,
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"Generated {len(df)} records → {OUT}")
    print(df.head())
    print("\nClass balance (high_risk):")
    print(df["high_risk"].value_counts(normalize=True))

if __name__ == "__main__":
    generate()
