import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import linregress
import os

st.set_page_config(page_title="Earthquake Dashboard", layout="wide")
st.title("🌍 Earthquake Magnitude & Frequency Dashboard")

# ==========================================================
#     SAFE DATA LOADER — auto-detect correct file path
# ==========================================================
@st.cache_data
def load_data():

    possible_paths = [
        "data/cleaned_earthquakes.xls",
        "./cleaned_earthquakes.xls",
        "/mount/src/earthquake/data/cleaned_earthquakes.xls",
        "/mount/src/data/cleaned_earthquakes.xls"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            df = pd.read_excel(path)
            df['date'] = pd.to_datetime(df['date'])
            return df

    st.error("❌ File 'cleaned_earthquakes.xls' not found. Make sure it is inside the 'data/' folder or root directory.")
    return pd.DataFrame()

df = load_data()

# Stop execution if file missing
if df.empty:
    st.stop()

# ==========================================================
#                       SIDEBAR FILTERS
# ==========================================================
st.sidebar.header("Filters")

min_mag = st.sidebar.slider("Minimum Magnitude", 4.0, 7.5, 4.0)

continents = st.sidebar.multiselect(
    "Select Continents",
    df["continent"].unique(),
    default=df["continent"].unique()
)

alert_level = st.sidebar.selectbox(
    "Alert Level",
    ["All", "green", "yellow", "orange", "red"]
)

filtered_df = df[
    (df["magnitude"] >= min_mag) &
    (df["continent"].isin(continents))
].copy()

if alert_level != "All":
    filtered_df = filtered_df[filtered_df["alert"] == alert_level]

# ==========================================================
#                          KPI CARDS
# ==========================================================
st.markdown("### 🔢 Key Statistics")

if filtered_df.empty:
    st.warning("⚠ No data matches your filters. Try adjusting them.")
else:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Earthquakes", len(filtered_df))
    col2.metric("Average Magnitude", f"{filtered_df['magnitude'].mean():.2f}")
    col3.metric("Strong Events (≥6.0)", (filtered_df["magnitude"] >= 6.0).sum())

# ==========================================================
#                          MAP VIEW
# ==========================================================
st.subheader("🗺 Global Earthquake Map")

fig1 = px.scatter_geo(
    filtered_df,
    lat="latitude",
    lon="longitude",
    size="magnitude",
    color="continent",
    hover_name="place",
    hover_data={"date": True, "alert": True, "magnitude": True},
    title="Earthquake Locations & Magnitudes",
)
fig1.update_layout(height=500, margin=dict(l=0, r=0, t=30, b=0))
st.plotly_chart(fig1, use_container_width=True)

# ==========================================================
#                  REGIONAL & TEMPORAL TRENDS
# ==========================================================
st.markdown("---")
st.header("📊 Regional & Trend Analysis")

col_left, col_right = st.columns(2)

# ------------------ BAR GRAPH ------------------
with col_left:
    bar_data = filtered_df.groupby("continent")["magnitude"].mean().reset_index()
    fig2 = px.bar(
        bar_data, x="continent", y="magnitude",
        title="Average Magnitude by Continent", text_auto=True
    )
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)

# ------------------ MONTHLY LINE ------------------
with col_right:
    filtered_df["month"] = filtered_df["date"].dt.month
    month_data = filtered_df.groupby("month")["magnitude"].mean().reset_index()

    fig3 = px.line(
        month_data, x="month", y="magnitude", markers=True,
        title="Monthly Average Magnitude"
    )
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

# ==========================================================
#                         HEATMAP
# ==========================================================
st.markdown("---")
st.subheader("🔥 Correlation Heatmap")

corr_data = filtered_df[["magnitude", "latitude", "longitude"]].corr()

fig4 = px.imshow(
    corr_data,
    text_auto=True,
    title="Correlation Matrix (Magnitude, Latitude, Longitude)",
    color_continuous_scale="RdBu"
)
st.plotly_chart(fig4, use_container_width=True)

# ==========================================================
#                   GUTENBERG-RICHTER PLOT
# ==========================================================
st.markdown("---")
st.subheader("📈 Gutenberg–Richter Frequency Law")

bins = np.arange(4, filtered_df["magnitude"].max() + 0.5, 0.5)
hist, edges = np.histogram(filtered_df["magnitude"], bins=bins)
mid_bins = (edges[:-1] + edges[1:]) / 2
log_freq = np.log10(hist + 1)
mask = hist > 0

slope, intercept, _, _, _ = linregress(mid_bins[mask], log_freq[mask])

fig5 = px.scatter(
    x=mid_bins, y=log_freq,
    title=f"Log-Frequency vs Magnitude (b-value: {-slope:.2f})",
    labels={"x": "Magnitude", "y": "Log10(Frequency)"}
)
fig5.add_scatter(
    x=mid_bins, y=intercept + slope * mid_bins,
    mode="lines", name="Fit Line", line=dict(color="red", dash="dash")
)
st.plotly_chart(fig5, use_container_width=True)

# ==========================================================
#             NEW VISUALIZATION 1 — DEPTH ANALYSIS
# ==========================================================
st.markdown("---")
st.subheader("🔵 Depth vs Magnitude")

if "depth" in df.columns:
    fig6 = px.scatter(
        filtered_df,
        x="depth", y="magnitude",
        color="continent",
        trendline="ols",
        title="Depth vs Magnitude"
    )
    st.plotly_chart(fig6, use_container_width=True)
else:
    st.info("ℹ No 'depth' column found in dataset.")

# ==========================================================
#      NEW VISUALIZATION 2 — YEARLY EARTHQUAKE COUNT
# ==========================================================
st.subheader("📅 Earthquake Frequency by Year")

filtered_df["year"] = filtered_df["date"].dt.year
year_counts = filtered_df.groupby("year")["magnitude"].count().reset_index()

fig7 = px.bar(
    year_counts,
    x="year", y="magnitude",
    title="Earthquake Frequency Over Years",
    labels={"magnitude": "Number of Earthquakes"}
)
fig7.update_xaxes(type="category")
st.plotly_chart(fig7, use_container_width=True)

# ==========================================================
#                       FOOTER
# ==========================================================
st.markdown("---")
st.caption("Dashboard: KPIs → Map → Region → Time → Heatmap → Gutenberg → Depth → Yearly Trends")
st.caption("CSA202 Project | Pelgye Dorji, Karma Sangay Palden, Dewas Chhuwan, Karma Wangdi (2025)")
