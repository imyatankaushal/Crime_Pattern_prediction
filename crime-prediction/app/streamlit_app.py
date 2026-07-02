import os, joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE, "models")
DATA = os.path.join(BASE, "data", "crime_data.csv")

st.set_page_config(page_title="Crime Intelligence System", page_icon="🚔", layout="wide")

# ================= UI STYLE =================
st.markdown("""
<style>
.stApp {background: radial-gradient(circle, #020617, #0f172a);}
h1,h2,h3 {color:#00ffff;text-shadow:0 0 10px #00ffff;}
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 15px;
    box-shadow: 0 0 15px rgba(0,255,255,0.3);
}
.stButton>button {
    border-radius: 10px;
    background: linear-gradient(90deg,#00ffff,#0077ff);
    color: black;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* ===== GLOBAL FADE ===== */
.fade-in {
    animation: fadeIn 0.8s ease-in;
}
@keyframes fadeIn {
    from {opacity:0; transform:translateY(10px);}
    to {opacity:1; transform:translateY(0);}
}

/* ===== METRIC HOVER ===== */
[data-testid="metric-container"] {
    transition: all 0.3s ease;
}
[data-testid="metric-container"]:hover {
    transform: scale(1.05);
    box-shadow: 0 0 25px rgba(0,255,255,0.6);
}

/* ===== GRAPH HOVER GLOW ===== */
.stPlotlyChart:hover {
    transform: scale(1.01);
    box-shadow: 0 0 25px rgba(0,255,255,0.3);
    transition: all 0.3s ease;
}

/* ===== BUTTON ANIMATION ===== */
.stButton>button {
    transition: 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px cyan;
}

/* ===== SECTION SPACING ===== */
.section {
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* ===== NAV BAR CONTAINER ===== */
.navbar {
    display: flex;
    justify-content: center;   /* center alignment */
    gap: 30px;                 /* spacing between tabs */
    padding: 12px 0;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

/* ===== EACH TAB ===== */
.nav-item {
    font-size: 15px;
    color: #cbd5e1;
    cursor: pointer;
    padding-bottom: 6px;
    transition: 0.3s ease;
}

/* ===== HOVER EFFECT ===== */
.nav-item:hover {
    color: #00ffff;
}

/* ===== ACTIVE TAB ===== */
.nav-active {
    color: #00ffff;
    border-bottom: 2px solid #00ffff;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* ===== ALERT CARD ===== */
.alert-card {
    background: rgba(255, 0, 0, 0.08);
    border-left: 4px solid #ff4d4d;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 12px;
    backdrop-filter: blur(8px);
    box-shadow: 0 0 15px rgba(255,0,0,0.15);
    transition: 0.3s ease;
}

/* ===== HOVER EFFECT ===== */
.alert-card:hover {
    transform: translateX(6px);
    box-shadow: 0 0 25px rgba(255,0,0,0.4);
}

/* ===== ALERT TEXT ===== */
.alert-text {
    color: #ff6b6b;
    font-size: 14px;
    font-weight: 500;
}

/* ===== ALERT HEADER ===== */
.alert-header {
    font-size: 22px;
    font-weight: 600;
    color: #00ffff;
    margin-bottom: 10px;
    text-shadow: 0 0 10px #00ffff;
}

</style>
""", unsafe_allow_html=True)

# ================= SAFE FIG STYLE =================
def style_fig(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#00ffff"),
        transition=dict(duration=700, easing="cubic-in-out")  # ✅ animation added
    )
    for trace in fig.data:
        if hasattr(trace, "marker") and trace.marker is not None:
            try:
                trace.marker.line = dict(width=1, color="white")
            except:
                pass
        if hasattr(trace, "line") and trace.line is not None:
            try:
                trace.line.width = 3
            except:
                pass
    return fig

# ================= LOAD =================
@st.cache_data
def load_data():
    return pd.read_csv(DATA, parse_dates=["timestamp","date"])

@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODELS,"best_classifier.pkl"))
    encoders = joblib.load(os.path.join(MODELS,"encoders.pkl"))
    kmeans = joblib.load(os.path.join(MODELS,"kmeans_hotspots.pkl"))
    scaler = joblib.load(os.path.join(MODELS,"scaler.pkl"))  # ✅ FIX

    return model, encoders, kmeans, scaler

df = load_data()
model, encoders, kmeans, scaler = load_artifacts()

model, encoders, kmeans, scaler = load_artifacts()

# ✅ ADD HERE
X_cols = [
    "hour","month","is_weekend","severity","temperature_c",
    "population_density","prev_incidents_30d",
    "district","crime_type","weather","day_of_week"
]

st.markdown("""
<div style='text-align:center; margin-bottom:25px;'>
    <h1 style='color:#00ffff; text-shadow:0 0 25px #00ffff;'>🚔 Crime Intelligence System</h1>
    <p style='color:#94a3b8; font-size:20px;'>Analyze crime trends • Detect hotspots • Predict risk</p>
</div>
""", unsafe_allow_html=True)

# ================= TABS =================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
["📊 Dashboard","🗺️ Hotspots","🔮 Predict","📈 Insights","🤖 Chatbot","🧠 Advanced"]
)

# ================= DASHBOARD =================

# ================= DASHBOARD =================
with tab1:
    st.markdown("### 🎯 Filters")

    st.markdown('<div class="fade-in section">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    selected_district = col1.multiselect("📍 District", df["district"].unique())
    selected_crime = col2.multiselect("🚔 Crime Type", df["crime_type"].unique())
    st.markdown('</div>', unsafe_allow_html=True)

    # ===== FILTER LOGIC =====
    filtered_df = df.copy()
    if selected_district:
        filtered_df = filtered_df[filtered_df["district"].isin(selected_district)]
    if selected_crime:
        filtered_df = filtered_df[filtered_df["crime_type"].isin(selected_crime)]

    # ===== ADVANCED FILTERS =====
    st.markdown("### ⚙️ Advanced Filters")

    st.markdown('<div class="fade-in section">', unsafe_allow_html=True)
    colA, colB, colC = st.columns([2,2,1])

    date_range = colA.date_input("📅 Date Range",
                                [df["date"].min(), df["date"].max()])
    hour_range = colB.slider("⏰ Hour Range", 0, 23, (0, 23))
    only_high_risk = colC.checkbox("⚠️ High Risk Only")

    st.markdown('</div>', unsafe_allow_html=True)

    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df["date"] >= pd.to_datetime(date_range[0])) &
            (filtered_df["date"] <= pd.to_datetime(date_range[1]))
        ]

    filtered_df = filtered_df[
        (filtered_df["hour"] >= hour_range[0]) &
        (filtered_df["hour"] <= hour_range[1])
    ]

    if only_high_risk:
        filtered_df = filtered_df[filtered_df["high_risk"] == 1]

    # ===== METRICS =====
    st.markdown('<div class="fade-in section">', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total", len(filtered_df))
    c2.metric("High Risk %", f"{filtered_df['high_risk'].mean()*100:.1f}%")
    c3.metric("Districts", filtered_df["district"].nunique())
    c4.metric("Crime Types", filtered_df["crime_type"].nunique())
    st.markdown('</div>', unsafe_allow_html=True)

    avg_daily = filtered_df.groupby("date").size().mean()
    st.metric("Avg Incidents / Day", f"{avg_daily:.1f}")

    # ===== QUICK INSIGHTS =====
    st.markdown("### 🧠 Quick Insights")
    colx, coly = st.columns(2)
    colx.success(f"📅 Peak Day: {filtered_df['day_of_week'].mode()[0]}")
    coly.success(f"🛡️ Safest: {filtered_df.groupby('district')['high_risk'].mean().idxmin()}")

    # ===== CHARTS =====
    top_n = st.slider("Top N Categories", 5, 30, 15)

    crime_counts = filtered_df["crime_type"].value_counts().head(top_n).reset_index()
    crime_counts.columns = ["crime_type","count"]

    fig = px.bar(crime_counts, y="crime_type", x="count",
                 orientation="h", color="count",
                 color_continuous_scale="Turbo")
    st.markdown('<div class="fade-in section">', unsafe_allow_html=True)
    st.plotly_chart(style_fig(fig), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    fig = px.bar(filtered_df.groupby("district").size().reset_index(name="count"),
                 x="district", y="count", color="count")
    st.markdown('<div class="fade-in section">', unsafe_allow_html=True)
    st.plotly_chart(style_fig(fig), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    trend = filtered_df.groupby("date").size().reset_index(name="count")
    fig = px.line(trend, x="date", y="count")
    st.markdown('<div class="fade-in section">', unsafe_allow_html=True)
    st.plotly_chart(style_fig(fig), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ===== SMART ALERT SYSTEM =====
    st.markdown("## 🚨 Smart Alerts")

    colA, colB = st.columns([2,1])
    risk_threshold = colA.slider("Risk Threshold", 0.0, 1.0, 0.7)
    auto_refresh = colB.checkbox("Auto Refresh")

    if "risk_score" not in filtered_df.columns:
        filtered_df["risk_score"] = filtered_df["high_risk"]

    alerts_df = filtered_df[filtered_df["risk_score"] >= risk_threshold]

    st.success(f"🚨 {len(alerts_df)} alerts triggered")

    if alerts_df.empty:
        st.info("No alerts at this threshold ✅")
    else:
        for _, row in alerts_df.tail(5).iterrows():
            st.markdown(f"""
            <div class="alert-card">
                <div class="alert-text">
                    📍 <b>{row['district']}</b> • {row['crime_type']}  
                    ⏰ Hour: {row['hour']} | Risk: {round(row['risk_score'],2)}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ===== DATA TABLE =====
    st.markdown("""
    <div style="
        font-size: 22px;
        font-weight: 600;
        color: #00ffff;
        text-shadow: 0 0 10px #00ffff;
        margin-top: 20px;
        margin-bottom: 10px;
    ">
    📊 Crime Data Explorer
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3,1])
    search = col1.text_input("🔍 Search (district or crime type)")
    limit = col2.slider("Rows", 10, 100, 50)

    display_df = filtered_df.copy()

    if search:
        display_df = display_df[
            display_df["crime_type"].str.contains(search, case=False, na=False) |
            display_df["district"].str.contains(search, case=False, na=False)
        ]

    st.caption(f"Showing {len(display_df)} records")

    def highlight_risk(row):
        return ['background-color: rgba(255,0,0,0.15)' if row['high_risk'] == 1 else '' for _ in row]

    styled_df = display_df.head(limit).style.apply(highlight_risk, axis=1)

    st.dataframe(styled_df, use_container_width=True)

    st.download_button(
        "Download data",
        filtered_df.to_csv(index=False),
        "data.csv"
    )
    # ===== REPORT GENERATOR =====
st.markdown("### 🧾 Generate Report")

if st.button("Generate Report"):

    total = len(filtered_df)
    high_risk_pct = filtered_df["high_risk"].mean() * 100
    top_crime = filtered_df["crime_type"].mode()[0]
    top_district = filtered_df["district"].mode()[0]
    peak_hour = filtered_df["hour"].mode()[0]

    report = f"""
CRIME ANALYSIS REPORT
=====================

Total Records: {total}
High Risk Percentage: {high_risk_pct:.2f}%

Most Common Crime: {top_crime}
Most Active District: {top_district}
Peak Crime Hour: {peak_hour}:00

Generated from Crime Intelligence System
"""

    st.text_area("Report Preview", report, height=250)

    st.download_button(
        "⬇️ Download Report",
        report,
        file_name="crime_report.txt"
    )

# ================= HOTSPOTS =================
# ================= HOTSPOTS =================
with tab2:
    st.markdown("### 🗺️ Hotspot Analysis")

    # Keep your existing clustering logic
    df["hotspot"] = kmeans.predict(df[["latitude","longitude"]])

    # ===== CONTROLS =====
    col1, col2, col3 = st.columns(3)

    selected_cluster = col1.selectbox(
        "Select Cluster",
        ["All"] + sorted(df["hotspot"].unique().tolist())
    )

    show_heatmap = col2.checkbox("Show Density Heatmap")
    sample_size = col3.slider("Sample Size", 500, 5000, 2000)

    map_df = df.copy()

    if selected_cluster != "All":
        map_df = map_df[map_df["hotspot"] == selected_cluster]

    map_df = map_df.sample(min(sample_size, len(map_df)))

    # ===== MAP =====
    if show_heatmap:
        fig = px.density_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            z=None,
            radius=10,
            zoom=10,
            mapbox_style="carto-darkmatter"
        )
    else:
        fig = px.scatter_mapbox(
            map_df,
            lat="latitude",
            lon="longitude",
            color="hotspot",
            hover_data=["district","crime_type"],
            zoom=10,
            mapbox_style="carto-darkmatter"
        )

    st.markdown('<div class="fade-in section">', unsafe_allow_html=True)
    st.plotly_chart(style_fig(fig), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ===== HOTSPOT SUMMARY =====
    st.markdown("### 📊 Hotspot Summary")

    colA, colB = st.columns(2)

    # Most dangerous cluster
    cluster_risk = df.groupby("hotspot")["high_risk"].mean()
    most_dangerous = cluster_risk.idxmax()

    colA.metric("Most Dangerous Cluster", most_dangerous)

    # Safest cluster
    safest = cluster_risk.idxmin()
    colB.metric("Safest Cluster", safest)

    # ===== CLUSTER DISTRIBUTION =====
    st.markdown("### 📈 Cluster Distribution")

    cluster_counts = df["hotspot"].value_counts().reset_index()
    cluster_counts.columns = ["cluster","count"]

    fig = px.bar(
        cluster_counts,
        x="cluster",
        y="count",
        color="count",
        color_continuous_scale="Turbo"
    )

    st.markdown('<div class="fade-in section">', unsafe_allow_html=True)
    st.plotly_chart(style_fig(fig), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ===== TOP DISTRICTS IN CLUSTER =====
    st.markdown("### 🏙️ Top Districts in Selected Cluster")

    if selected_cluster != "All":
        top_districts = (
            df[df["hotspot"] == selected_cluster]
            .groupby("district")
            .size()
            .sort_values(ascending=False)
            .head(5)
        )

        for d, v in top_districts.items():
            st.write(f"• {d}: {v} incidents")

# ================= PREDICT =================
# ================= PREDICT =================

# ================= PREDICT =================
with tab3:
    st.markdown("### 🔮 Risk Prediction")

    col1, col2 = st.columns(2)

    hour = col1.slider("Hour",0,23,22)
    district = col1.selectbox("District", df["district"].unique())
    crime = col1.selectbox("Crime Type", df["crime_type"].unique())

    weather = col2.selectbox("Weather", df["weather"].unique())
    dow = col2.selectbox("Day", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

    # ===== WHAT-IF QUICK PREVIEW =====
# ===== CREATE INPUT ROW =====
temp_row = pd.DataFrame([{
    "hour": hour,
    "month": 6,
    "is_weekend": int(dow in ["Saturday","Sunday"]),
    "severity": 3,
    "temperature_c": 25,
    "population_density": 8000,
    "prev_incidents_30d": 20,
    "district": encoders["district"].transform([district])[0],
    "crime_type": encoders["crime_type"].transform([crime])[0],
    "weather": encoders["weather"].transform([weather])[0],
    "day_of_week": encoders["day_of_week"].transform([dow])[0],
}])

# ===== MATCH TRAINING COLUMNS =====
temp_row = temp_row[X_cols]

# ===== SCALE + PREDICT =====
try:
    temp_scaled = scaler.transform(temp_row)
    live_proba = model.predict_proba(temp_scaled)[0,1]
except:
    live_proba = model.predict_proba(temp_row)[0,1]

st.info(f"⚡ Live Risk Estimate: {live_proba*100:.2f}%")

    # ===== FINAL PREDICTION =====
if st.button("Predict Risk"):

        proba = live_proba

        # ===== RISK CATEGORY =====
        if proba < 0.3:
            category = "Low Risk"
        elif proba < 0.7:
            category = "Medium Risk"
        else:
            category = "High Risk"

        st.success(f"Prediction: {proba*100:.2f}% ({category})")

        # ===== GAUGE (Donut) =====
        fig = px.pie(
            names=["Risk","Safe"],
            values=[proba, 1-proba],
            hole=0.7
        )
        fig.update_traces(textinfo="none")
        fig.update_layout(showlegend=False)

        st.plotly_chart(style_fig(fig), use_container_width=True)

        # ===== COMPARISON =====
        avg_risk = df["high_risk"].mean()

        st.markdown("### 📊 Comparison")

        colA, colB = st.columns(2)
        colA.metric("Your Prediction", f"{proba*100:.2f}%")
        colB.metric("Average Risk", f"{avg_risk*100:.2f}%")

        # ===== BASIC INSIGHTS =====
        st.markdown("### 🧠 Insights")

        if hour >= 20 or hour <= 4:
            st.write("• Night hours show higher crime probability")
        if dow in ["Saturday","Sunday"]:
            st.write("• Weekend risk is generally higher")
        if "THEFT" in crime.upper():
            st.write("• Theft-related crimes are frequent")

        # ===== EXPLAINABLE AI (NEW) =====
        st.markdown("### 🔍 Why this prediction?")

        explanations = []

        if hour >= 20 or hour <= 5:
            explanations.append("🌙 Night hours increase crime probability")

        if district in df["district"].value_counts().head(5).index:
            explanations.append("📍 This district has historically high crime rate")

        if crime in df["crime_type"].value_counts().head(5).index:
            explanations.append("⚠️ This crime type is frequently reported")

        if weather in ["Rain", "Storm"]:
            explanations.append("🌧️ Weather conditions may influence crime patterns")

        if dow in ["Friday", "Saturday"]:
            explanations.append("🎉 Weekend activity increases incidents")

        if explanations:
            for exp in explanations:
                st.info(exp)
        else:
            st.success("Low risk scenario based on current inputs")

        # ===== SAVE HISTORY =====
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "district":district,
            "crime":crime,
            "hour":hour,
            "risk":round(proba*100,2)
        })

    # ===== HISTORY =====
if "history" in st.session_state:
        st.markdown("### 🕘 Prediction History")
        hist_df = pd.DataFrame(st.session_state.history[::-1])
        st.dataframe(hist_df.head(10))

# ================= INSIGHTS =================
# ================= INSIGHTS =================
with tab4:
    st.markdown("### 📈 Insights & Analysis")

    # ===== FILTERS =====
    col1, col2, col3 = st.columns(3)

    selected_view = col1.selectbox("View Type", ["Hourly", "Day", "Month"])
    selected_district = col2.selectbox("District", ["All"] + sorted(df["district"].unique()))
    selected_crime = col3.selectbox("Crime Type", ["All"] + sorted(df["crime_type"].unique()))

    insights_df = df.copy()

    if selected_district != "All":
        insights_df = insights_df[insights_df["district"] == selected_district]

    if selected_crime != "All":
        insights_df = insights_df[insights_df["crime_type"] == selected_crime]

    # ===== MAIN TREND =====
    st.markdown("#### 📊 Trend Analysis")

    if selected_view == "Hourly":
        data = insights_df.groupby("hour").size().reset_index(name="count")
        fig = px.line(data, x="hour", y="count", markers=True)
    elif selected_view == "Day":
        data = insights_df.groupby("day_of_week").size().reset_index(name="count")
        fig = px.bar(data, x="day_of_week", y="count")
    else:
        data = insights_df.groupby("month").size().reset_index(name="count")
        fig = px.line(data, x="month", y="count", markers=True)

    st.plotly_chart(style_fig(fig), use_container_width=True)

    # ===== PEAK ANALYSIS =====
    st.markdown("#### 🔥 Peak Detection")

    peak_value = data.loc[data["count"].idxmax()]

    st.info(f"Peak {selected_view}: {peak_value.iloc[0]} with {peak_value['count']} incidents")

    # ===== TOP CRIMES =====
    st.markdown("#### 🚔 Top Crime Types")

    top_crimes = (
        insights_df["crime_type"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_crimes.columns = ["crime_type","count"]

    fig = px.bar(top_crimes, x="crime_type", y="count", color="count")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    # ===== TOP DISTRICTS =====
    st.markdown("#### 🏙️ Top Districts")

    top_districts = (
        insights_df["district"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_districts.columns = ["district","count"]

    fig = px.bar(top_districts, x="district", y="count", color="count")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    # ===== WEEKEND VS WEEKDAY =====
    st.markdown("#### 📅 Weekend vs Weekday")

    insights_df["type"] = np.where(
        insights_df["day_of_week"].isin(["Saturday","Sunday"]),
        "Weekend",
        "Weekday"
    )

    compare = insights_df.groupby("type").size().reset_index(name="count")

    fig = px.pie(compare, names="type", values="count", hole=0.5)
    st.plotly_chart(style_fig(fig), use_container_width=True)

    # ===== DRILL DOWN =====
    st.markdown("#### 🔎 Drill-down Analysis")

    selected_drill = st.selectbox("Drill By", ["district", "crime_type", "hour"])

    drill_data = (
        insights_df.groupby(selected_drill)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(15)
    )

    fig = px.bar(drill_data, x=selected_drill, y="count", color="count")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    # ===== AUTO INSIGHTS =====
    st.markdown("#### 🧠 Automated Insights")

    most_common_crime = insights_df["crime_type"].mode()[0]
    most_active_district = insights_df["district"].mode()[0]
    busiest_hour = insights_df["hour"].mode()[0]

    st.write(f"• Most frequent crime: **{most_common_crime}**")
    st.write(f"• Most active district: **{most_active_district}**")
    st.write(f"• Peak hour: **{busiest_hour}:00**")

    # ===== EXPORT =====
    st.download_button(
        "Download Insights Data",
        insights_df.to_csv(index=False),
        "insights.csv"
    )

# ================= CHATBOT =================
# ================= CHATBOT =================

# ================= CHATBOT =================
with tab5:
    st.markdown("### 🤖 Smart Crime Assistant")

    # ===== SESSION STATE =====
    if "msgs" not in st.session_state:
        st.session_state.msgs = []

    # ===== SUGGESTIONS =====
    st.markdown("#### 💡 Try asking:")
    st.caption("""
    • Which district is most dangerous?  
    • Show crime trend  
    • Compare districts  
    • Theft cases  
    • Give summary  
    • Peak crime hour  
    """)

    # ===== BOT FUNCTION =====
    def bot(q):
        q = q.lower()

        # ===== DANGEROUS DISTRICT =====
        if "danger" in q:
            district = df.groupby('district')['high_risk'].mean().idxmax()
            return f"🚨 Most dangerous district: {district}"

        # ===== COMMON CRIME =====
        elif "common" in q or "crime" in q:
            crime = df['crime_type'].mode()[0]
            return f"🔎 Most common crime: {crime}"

        # ===== PEAK HOUR =====
        elif "hour" in q or "time" in q:
            hour = df['hour'].mode()[0]
            return f"⏰ Peak crime hour: {hour}:00"

        # ===== TREND CHART =====
        elif "trend" in q:
            trend = df.groupby("date").size().reset_index(name="count")
            fig = px.line(trend, x="date", y="count")
            return fig

        # ===== DISTRICT COMPARISON =====
        elif "compare" in q:
            comp = df.groupby("district").size().reset_index(name="count")
            fig = px.bar(comp, x="district", y="count", color="count")
            return fig

        # ===== FILTERED DATA =====
        elif "theft" in q:
            sub = df[df["crime_type"].str.contains("THEFT", case=False)]
            return f"🔍 Found {len(sub)} theft-related incidents"

        elif "assault" in q:
            sub = df[df["crime_type"].str.contains("ASSAULT", case=False)]
            return f"⚠️ Found {len(sub)} assault cases"

        elif "weekend" in q:
            weekend = df[df["day_of_week"].isin(["Saturday","Sunday"])].shape[0]
            weekday = df.shape[0] - weekend
            return f"📅 Weekend incidents: {weekend}, Weekday incidents: {weekday}"

        # ===== SUMMARY =====
        elif "summary" in q or "insight" in q:
            top_crime = df["crime_type"].mode()[0]
            top_district = df["district"].mode()[0]
            peak_hour = df["hour"].mode()[0]

            return f"""
🧠 Summary:
• Most common crime: {top_crime}
• Most active district: {top_district}
• Peak hour: {peak_hour}:00
"""

        # ===== DEFAULT =====
        else:
            return "🤖 Try asking about trends, districts, crime types, or comparisons."

    # ===== USER INPUT =====
    user = st.chat_input("Ask something about crime data...")

    if user:
        st.session_state.msgs.append(("user", user))

        response = bot(user)

        if isinstance(response, go.Figure):
            st.session_state.msgs.append(("bot_chart", response))
        else:
            st.session_state.msgs.append(("bot", response))

    # ===== DISPLAY CHAT =====
    for role, msg in st.session_state.msgs:

        if role == "user":
            st.chat_message("user").write(msg)

        elif role == "bot":
            st.chat_message("assistant").write(msg)

        elif role == "bot_chart":
            with st.chat_message("assistant"):
                st.plotly_chart(style_fig(msg), use_container_width=True)

# ================= ADVANCED =================
# ================= ADVANCED =================
with tab6:
    st.markdown("### 🧠 Advanced Analytics Lab")

    adv_df = df.copy()

    # ===== CATEGORY CREATION (keep your logic) =====
    adv_df["category"] = "Other"
    adv_df.loc[adv_df["crime_type"].str.contains("THEFT|ROBBERY|BURGLARY", case=False), "category"] = "Theft"
    adv_df.loc[adv_df["crime_type"].str.contains("ASSAULT|BATTERY", case=False), "category"] = "Violence"
    adv_df.loc[adv_df["crime_type"].str.contains("FRAUD", case=False), "category"] = "Fraud"

    # ===== CATEGORY DISTRIBUTION =====
    st.markdown("#### 📊 Crime Category Distribution")
    cat_counts = adv_df["category"].value_counts().reset_index()
    cat_counts.columns = ["category", "count"]

    fig = px.pie(cat_counts, names="category", values="count", hole=0.5)
    st.plotly_chart(style_fig(fig), use_container_width=True)

    # ===== CATEGORY TREND =====
    st.markdown("#### 📈 Category Trend Over Time")

    trend_cat = adv_df.groupby(["date", "category"]).size().reset_index(name="count")

    fig = px.line(trend_cat, x="date", y="count", color="category")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    # ===== RISK SEGMENTATION =====
    st.markdown("#### ⚠️ Risk Segmentation")

    adv_df["risk_level"] = pd.cut(
        adv_df["high_risk"],
        bins=[-1, 0, 1],
        labels=["Low Risk", "High Risk"]
    )

    risk_counts = adv_df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["risk_level", "count"]

    fig = px.bar(risk_counts, x="risk_level", y="count", color="risk_level")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    # ===== HEATMAP =====
    st.markdown("#### 🔥 Crime Heatmap (Hour vs Day)")

    heat = adv_df.groupby(["day_of_week", "hour"]).size().reset_index(name="count")

    fig = px.density_heatmap(heat, x="hour", y="day_of_week", z="count")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    # ===== CORRELATION =====
    st.markdown("#### 🔗 Correlation Analysis")

    corr_df = adv_df[["hour", "month", "population_density", "temperature_c", "prev_incidents_30d"]]
    corr = corr_df.corr()

    fig = px.imshow(corr, text_auto=True)
    st.plotly_chart(style_fig(fig), use_container_width=True)

    # ===== ROLLING TREND =====
    st.markdown("#### 📉 Smoothed Trend (Rolling Average)")

    ts = adv_df.groupby("date").size().reset_index(name="count")
    ts["rolling"] = ts["count"].rolling(7).mean()

    fig = px.line(ts, x="date", y=["count", "rolling"])
    st.plotly_chart(style_fig(fig), use_container_width=True)

    # ===== ANOMALY DETECTION =====
    st.markdown("#### 🚨 Anomaly Detection")

    threshold = ts["count"].mean() + 2 * ts["count"].std()
    anomalies = ts[ts["count"] > threshold]

    fig = px.line(ts, x="date", y="count")
    fig.add_scatter(
        x=anomalies["date"],
        y=anomalies["count"],
        mode="markers",
        name="Anomalies"
    )
    st.plotly_chart(style_fig(fig), use_container_width=True)

    # ===== FORECAST =====
    st.markdown("#### 🔮 Forecast (Next 7 Days)")

    ts["t"] = np.arange(len(ts))
    lr = LinearRegression().fit(ts[["t"]], ts["count"])

    future = pd.DataFrame({"t": np.arange(len(ts), len(ts)+7)})
    future["pred"] = lr.predict(future)

    future_dates = pd.date_range(ts["date"].max(), periods=7)

    fig = px.line(ts, x="date", y="count")
    fig.add_scatter(x=future_dates, y=future["pred"], name="Forecast")
    st.plotly_chart(style_fig(fig), use_container_width=True)

    # ===== TOP PATTERNS =====
    st.markdown("#### 🧠 Key Patterns")

    st.write(f"• Most common category: **{adv_df['category'].mode()[0]}**")
    st.write(f"• Peak crime hour: **{adv_df['hour'].mode()[0]}:00**")
    st.write(f"• Most active district: **{adv_df['district'].mode()[0]}**")

    # ===== EXPORT =====
    st.download_button(
        "Download Advanced Data",
        adv_df.to_csv(index=False),
        "advanced_analysis.csv"
    )