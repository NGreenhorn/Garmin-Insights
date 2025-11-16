import streamlit as st
import pandas as pd
import plotly.express as px
import re
import json
import os
import requests

px.defaults.template = "plotly_white"

# OpenRouter config (set OPENROUTER_API_KEY in env or Streamlit secrets)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "tngtech/deepseek-r1t2-chimera:free"  # model from OpenRouter example
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Page config and basic theming
st.set_page_config(
    page_title="Garmin Activity Dashboard",
    page_icon="🏃",
    layout="wide",
)

# Custom CSS to polish the UI
st.markdown(
    """
    <style>
    /* Make the main background a bit softer */
    .stApp {
        background: radial-gradient(circle at top left, #f9fafb 0, #e5e7eb 45%, #e5e7eb 100%);
        color: #111827;
    }

    /* Tweak sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f3f4f6;
    }

    /* Card-style containers */
    .metric-card {
        padding: 1.25rem 1.5rem;
        border-radius: 0.75rem;
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(209, 213, 219, 0.9);
        backdrop-filter: blur(4px);
        margin-bottom: 1rem;
    }

    .metric-title {
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #4b5563;
        margin-bottom: 0.25rem;
    }

    .metric-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #111827;
    }

    .metric-sub {
        font-size: 0.85rem;
        color: #4b5563;
    }

    /* Tab labels */
    button[role="tab"] {
        font-weight: 500 !important;
        letter-spacing: 0.03em;
    }


    /* Headers */
    h1, h2, h3 {
        color: #111827 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="padding: 0.5rem 0 1.5rem 0;">
        <h1 style="margin-bottom: 0.25rem;">🏃 Garmin Activity Dashboard</h1>
        <p style="color:#9ca3af; max-width: 640px; font-size:0.95rem;">
            Explore 8+ years of runs, heart rate and training trends. 
            Filter by year and activity type, then let your own data tell the story.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# File uploader
uploaded_file = st.file_uploader("Upload Garmin CSV file", type="csv")
if not uploaded_file:
    st.info("Please upload a Garmin activities CSV export to begin.")
    st.stop()

# Load data
# Read the CSV and ensure Date is parsed correctly
df = pd.read_csv(uploaded_file, parse_dates=["Date"])

# Try to detect the distance column (Garmin names often contain 'Distance')
distance_candidates = [c for c in df.columns if "distance" in c.lower()]
if distance_candidates:
    dist_col = distance_candidates[0]
    # Ensure distance column is numeric
    df[dist_col] = pd.to_numeric(df[dist_col], errors="coerce")
else:
    st.error("Could not find a distance column in the CSV.")
    st.write("Columns found:", list(df.columns))
    st.stop()

# Make sure Date is a proper datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Sort and set index
df = df.sort_values("Date")
df.set_index("Date", inplace=True)


# --- Helper functions for data summary and LLM Q&A ---
@st.cache_data
def build_data_summary(df, dist_col):
    # Focus on running activities if the column exists
    runs = df.copy()
    if "Activity Type" in runs.columns:
        runs = runs[runs["Activity Type"] == "Running"]

    # Ensure distance is numeric
    runs_dist = pd.to_numeric(runs[dist_col], errors="coerce")

    total_miles = float(runs_dist.sum())
    first_date = runs.index.min()
    last_date = runs.index.max()

    # Yearly totals (distance)
    yearly = runs_dist.groupby(runs.index.year).sum().to_dict()

    # Weekly totals for the last 8 weeks (distance)
    weekly = runs_dist.resample("W").sum().last("8W").to_dict()

    # Longest run and average runs per week (last 8 weeks)
    longest_run = float(runs_dist.max()) if not runs_dist.empty else 0.0
    runs_per_week_last_8_weeks = 0.0
    if not runs_dist.empty:
        weekly_counts = runs_dist.resample("W").count().last("8W")
        if len(weekly_counts) > 0:
            runs_per_week_last_8_weeks = float(weekly_counts.mean())

    # Heart rate summaries (Avg HR and Max HR)
    avg_hr_overall = None
    peak_hr_overall = None
    yearly_avg_hr = {}
    weekly_avg_hr_last_8_weeks = {}

    if "Avg HR" in runs.columns:
        hr_values = pd.to_numeric(runs["Avg HR"], errors="coerce").dropna()
        if not hr_values.empty:
            avg_hr_overall = float(hr_values.mean())
            yearly_avg_hr = hr_values.groupby(hr_values.index.year).mean().to_dict()
            weekly_avg_hr_last_8_weeks = (
                hr_values.resample("W").mean().last("8W").to_dict()
            )

    if "Max HR" in runs.columns:
        max_hr_values = pd.to_numeric(runs["Max HR"], errors="coerce").dropna()
        if not max_hr_values.empty:
            peak_hr_overall = float(max_hr_values.max())

    summary = {
        "total_miles": round(total_miles, 1),
        "first_date": str(first_date.date()) if pd.notna(first_date) else None,
        "last_date": str(last_date.date()) if pd.notna(last_date) else None,
        "yearly_miles": {int(k): round(float(v), 1) for k, v in yearly.items()},
        "weekly_miles_last_8_weeks": {
            str(k.date()): round(float(v), 1) for k, v in weekly.items()
        },
        "longest_run": round(longest_run, 1),
        "runs_per_week_last_8_weeks": round(runs_per_week_last_8_weeks, 2),
        # Heart rate-related fields
        "average_hr_overall": round(avg_hr_overall, 1) if avg_hr_overall is not None else None,
        "peak_hr_overall": round(peak_hr_overall, 1) if peak_hr_overall is not None else None,
        "yearly_average_hr": {int(k): round(float(v), 1) for k, v in yearly_avg_hr.items()},
        "weekly_average_hr_last_8_weeks": {
            str(k.date()): round(float(v), 1) for k, v in weekly_avg_hr_last_8_weeks.items()
        },
    }
    return summary


def ask_llm_about_runs(question, df, dist_col):
    summary = build_data_summary(df, dist_col)
    prompt = f"""
You are an assistant helping a runner understand their training.

Here is a JSON summary of their Garmin data:
{json.dumps(summary, indent=2)}

Answer the user's question using ONLY this data.
Be concise and concrete, and if you don't have enough data, say so.

User question: {question}
"""

    if not OPENROUTER_API_KEY:
        return "LLM error: OPENROUTER_API_KEY is not set. Please configure it in your environment or Streamlit secrets."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501/",  # change to your deployed URL if hosting
        "X-Title": "Garmin Activity Dashboard",     # app name for OpenRouter rankings
    }

    body = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful running coach."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    try:
        resp = requests.post(
            OPENROUTER_BASE_URL,
            headers=headers,
            data=json.dumps(body),
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        # OpenRouter uses the same shape as OpenAI's chat completion API
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"LLM error: {e}"


def best_time_for_distance(df, dist_col, target_distance, tolerance=0.5):
    """
    Find the best (shortest) time for runs around a target distance.
    target_distance is in the same units as the distance column.
    """
    if "Time" not in df.columns:
        return None

    df2 = df.copy()
    dist = pd.to_numeric(df2[dist_col], errors="coerce")
    mask = (dist >= target_distance - tolerance) & (dist <= target_distance + tolerance)
    subset = df2[mask]
    if subset.empty:
        return None

    times = pd.to_timedelta(subset["Time"], errors="coerce")
    if times.isna().all():
        return None

    best_idx = times.idxmin()
    best_time = times.loc[best_idx]
    return best_time, best_idx  # time and index (date)


def get_biggest_week(df, dist_col):
    """
    Return the week (start date) and mileage of the biggest mileage week.
    """
    weekly = pd.to_numeric(df[dist_col], errors="coerce").resample("W").sum()
    if weekly.empty:
        return None, None
    max_week_date = weekly.idxmax()
    max_week_miles = weekly.max()
    return max_week_date, max_week_miles


# Sidebar filters
st.sidebar.header("Filters")
year_filter = st.sidebar.multiselect("Year", options=sorted(df.index.year.unique()), default=None)
type_filter = st.sidebar.multiselect("Activity Type", options=df["Activity Type"].unique(), default=None)

# Training goals
st.sidebar.header("Goals")
annual_goal = st.sidebar.number_input("Annual mileage goal", value=1000, step=50)
weekly_goal = st.sidebar.number_input("Weekly mileage goal", value=30, step=5)

# Apply filters to dataframe
df_filtered = df.copy()
if year_filter:
    df_filtered = df_filtered[df_filtered.index.year.isin(year_filter)]
if type_filter:
    df_filtered = df_filtered[df_filtered["Activity Type"].isin(type_filter)]

# Tabs for layout
tab_overview, tab_trends, tab_ai = st.tabs(["📊 Overview", "📈 Trends", "🤖 AI Coach"])

with tab_overview:
    st.header("Overview")

    # Overall / YTD stats (based on filtered dataset)
    if not df_filtered.empty:
        latest_year = df_filtered.index.max().year
        this_year_df = df_filtered[df_filtered.index.year == latest_year]
        ytd_miles = pd.to_numeric(this_year_df[dist_col], errors="coerce").sum()

        last_4w = df_filtered.last("28D")
        prev_4w = (
            df_filtered[df_filtered.index < last_4w.index.min()].last("28D")
            if not last_4w.empty
            else df_filtered.iloc[0:0]
        )

        last_4w_miles = (
            pd.to_numeric(last_4w[dist_col], errors="coerce").sum()
            if not last_4w.empty
            else 0.0
        )
        prev_4w_miles = (
            pd.to_numeric(prev_4w[dist_col], errors="coerce").sum()
            if not prev_4w.empty
            else 0.0
        )

        longest_run = float(pd.to_numeric(df_filtered[dist_col], errors="coerce").max())

        with st.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card"><div class="metric-title">Year to date</div>'
                            f'<div class="metric-value">{round(ytd_miles, 1)} mi</div>'
                            '<div class="metric-sub">Total distance this year</div></div>',
                            unsafe_allow_html=True)
            with col2:
                delta_4w = round(last_4w_miles - prev_4w_miles, 1) if prev_4w_miles > 0 else None
                delta_text = f"{delta_4w:+.1f} vs prev 4 weeks" if delta_4w is not None else "No previous period"
                st.markdown('<div class="metric-card"><div class="metric-title">Last 4 weeks</div>'
                            f'<div class="metric-value">{round(last_4w_miles, 1)} mi</div>'
                            f'<div class="metric-sub">{delta_text}</div></div>',
                            unsafe_allow_html=True)
            with col3:
                longest_val = round(longest_run, 1) if longest_run > 0 else 0.0
                st.markdown('<div class="metric-card"><div class="metric-title">Longest run</div>'
                            f'<div class="metric-value">{longest_val} mi</div>'
                            '<div class="metric-sub">Single longest activity</div></div>',
                            unsafe_allow_html=True)

        # Heart rate overview metrics (based on filtered dataset)
        if "Avg HR" in df_filtered.columns:
            avg_hr_all = pd.to_numeric(df_filtered["Avg HR"], errors="coerce").mean()
        else:
            avg_hr_all = None

        if "Max HR" in df_filtered.columns:
            max_hr_all = pd.to_numeric(df_filtered["Max HR"], errors="coerce").max()
        else:
            max_hr_all = None

        if avg_hr_all is not None or max_hr_all is not None:
            hr_col1, hr_col2, _ = st.columns(3)
            if avg_hr_all is not None and not pd.isna(avg_hr_all):
                with hr_col1:
                    st.markdown('<div class="metric-card"><div class="metric-title">Average HR</div>'
                                f'<div class="metric-value">{int(round(avg_hr_all))} bpm</div>'
                                '<div class="metric-sub">Across filtered activities</div></div>',
                                unsafe_allow_html=True)
            if max_hr_all is not None and not pd.isna(max_hr_all):
                with hr_col2:
                    st.markdown('<div class="metric-card"><div class="metric-title">Max HR</div>'
                                f'<div class="metric-value">{int(round(max_hr_all))} bpm</div>'
                                '<div class="metric-sub">Peak recorded heart rate</div></div>',
                                unsafe_allow_html=True)

        # Goal progress
        if annual_goal > 0:
            progress = (ytd_miles / annual_goal) * 100 if annual_goal else 0
            st.metric(
                "Annual goal progress",
                f"{round(ytd_miles, 1)} / {annual_goal}",
                f"{round(progress, 1)}%"
            )

        # Weekly goal comparison (last 7 days based on filtered dataset)
        last_week = df_filtered.last("7D")
        last_week_miles = (
            pd.to_numeric(last_week[dist_col], errors="coerce").sum()
            if not last_week.empty
            else 0.0
        )
        if weekly_goal > 0:
            weekly_delta = round(last_week_miles - weekly_goal, 1)
            st.metric(
                "Last 7 days vs weekly goal",
                f"{round(last_week_miles, 1)} / {weekly_goal}",
                weekly_delta
            )

    # Personal bests and records
    with st.expander("Personal bests and records"):
        pb_5k = best_time_for_distance(df_filtered, dist_col, 5.0)
        pb_10k = best_time_for_distance(df_filtered, dist_col, 10.0)
        pb_hm = best_time_for_distance(df_filtered, dist_col, 21.1)

        if any([pb_5k, pb_10k, pb_hm]):
            if pb_5k:
                time_5k, date_5k = pb_5k
                st.write(f"Best ~5k: {time_5k} on {date_5k.date()}")
            if pb_10k:
                time_10k, date_10k = pb_10k
                st.write(f"Best ~10k: {time_10k} on {date_10k.date()}")
            if pb_hm:
                time_hm, date_hm = pb_hm
                st.write(f"Best ~half-marathon: {time_hm} on {date_hm.date()}")
        else:
            st.write("No personal bests found (not enough distance/time data).")

        # Biggest week
        big_week_date, big_week_miles = get_biggest_week(df_filtered, dist_col)
        if big_week_date is not None:
            st.write(
                f"Biggest week: {round(float(big_week_miles), 1)} miles (week of {big_week_date.date()})"
            )

with tab_trends:
    st.header("Trends")

    # Weekly mileage chart (based on filtered data)
    if not df_filtered.empty:
        weekly = pd.to_numeric(df_filtered[dist_col], errors="coerce").resample("W").sum()
        fig = px.line(
            x=weekly.index,
            y=weekly.values,
            labels={"x": "Week", "y": "Miles"},
            title="Weekly Mileage (filtered)"
        )
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            font_color="#111827",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Weekly average heart rate trend (if available)
    if "Avg HR" in df_filtered.columns and not df_filtered.empty:
        weekly_hr = pd.to_numeric(df_filtered["Avg HR"], errors="coerce").resample("W").mean()
        weekly_hr = weekly_hr.dropna()
        if not weekly_hr.empty:
            fig_hr_trend = px.line(
                x=weekly_hr.index,
                y=weekly_hr.values,
                labels={"x": "Week", "y": "Avg HR (bpm)"},
                title="Weekly Average Heart Rate (filtered)"
            )
            fig_hr_trend.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#ffffff",
                font_color="#111827",
            )
            st.plotly_chart(fig_hr_trend, use_container_width=True)

    # Pace trend chart (try Duration (h:m:s) first, then Time)
    duration_col = None
    if "Duration (h:m:s)" in df_filtered.columns:
        duration_col = "Duration (h:m:s)"
    elif "Time" in df_filtered.columns:
        duration_col = "Time"

    if duration_col is not None and not df_filtered.empty:
        pace_df = df_filtered.copy()
        pace_df["Duration_min"] = pd.to_timedelta(pace_df[duration_col], errors="coerce").dt.total_seconds() / 60
        pace_df["Pace_min_per_unit"] = pace_df["Duration_min"] / pd.to_numeric(pace_df[dist_col], errors="coerce")

        fig2 = px.scatter(
            pace_df,
            x=pace_df.index,
            y="Pace_min_per_unit",
            labels={"x": "Date", "Pace_min_per_unit": "Pace (min per distance unit)"},
            title="Pace over Time"
        )
        fig2.update_layout(
            template="plotly_white",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#ffffff",
            font_color="#111827",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Heart rate vs distance
    if "Avg HR" in df_filtered.columns:
        hr_df = df_filtered.copy()
        hr_df[dist_col] = pd.to_numeric(hr_df[dist_col], errors="coerce")
        hr_df = hr_df.dropna(subset=[dist_col, "Avg HR"])
        if not hr_df.empty:
            fig_hr = px.scatter(
                hr_df,
                x=dist_col,
                y="Avg HR",
                title="Heart rate vs distance",
                labels={dist_col: "Distance", "Avg HR": "Average HR (bpm)"}
            )
            fig_hr.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#ffffff",
                font_color="#111827",
            )
            st.plotly_chart(fig_hr, use_container_width=True)

    # Heart rate zones (crude buckets)
    if "Avg HR" in df_filtered.columns:
        # Coerce to numeric to avoid string values breaking pd.cut
        hr_values = pd.to_numeric(df_filtered["Avg HR"], errors="coerce").dropna()
        if not hr_values.empty:
            bins = [0, 120, 140, 160, 180, 220]
            labels_z = ["Easy", "Steady", "Tempo", "Hard", "Max"]
            zones = pd.cut(hr_values, bins=bins, labels=labels_z, right=False)
            zone_counts = zones.value_counts().reindex(labels_z, fill_value=0)
            st.subheader("Heart rate zones (by activity count)")
            st.bar_chart(zone_counts)

with tab_ai:
    st.header("AI Coach")

    st.write(
        "Ask questions about your training. The AI sees a summary of your mileage, heart rate, and recent trends, "
        "and will answer based only on that data."
    )

    user_q = st.text_input(
        "Ask a question (e.g. 'How many miles did I run in 2024?', 'How has my mileage changed recently?', "
        "or 'Is my average heart rate improving?')"
    )

    if user_q:
        with st.spinner("Thinking..."):
            answer = ask_llm_about_runs(user_q, df, dist_col)
        st.markdown(answer)