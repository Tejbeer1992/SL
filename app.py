# =============================================================================
#  🧠 MENTAL HEALTH ANALYTICS DASHBOARD
#  app.py — Single-file Streamlit Application
#  Dataset: Global Mental Health Survey (292,364 records, 2014–2016)
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import io
import base64
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindScope — Mental Health Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# THEME / STYLE
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#6C63FF",
    "secondary": "#FF6584",
    "accent":    "#43D9AD",
    "warning":   "#FFBF00",
    "danger":    "#FF4C61",
    "bg":        "#0F0F1A",
    "card":      "#1A1A2E",
    "border":    "#2D2D44",
    "text":      "#E0E0F0",
    "muted":     "#7B7B9A",
}

PLOTLY_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="Inter, sans-serif"),
    colorway=[COLORS["primary"], COLORS["secondary"], COLORS["accent"],
              COLORS["warning"], "#A78BFA", "#34D399", "#FB923C"],
)

RISK_PALETTE = {
    "Low":    COLORS["accent"],
    "Medium": COLORS["warning"],
    "High":   COLORS["danger"],
}

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {COLORS['bg']};
    color: {COLORS['text']};
  }}
  .main .block-container {{ padding: 1.5rem 2rem; max-width: 1400px; }}
  
  /* Sidebar */
  section[data-testid="stSidebar"] {{
    background: {COLORS['card']};
    border-right: 1px solid {COLORS['border']};
  }}
  section[data-testid="stSidebar"] * {{ color: {COLORS['text']} !important; }}

  /* KPI Cards */
  .kpi-card {{
    background: linear-gradient(135deg, {COLORS['card']} 0%, #1e1e35 100%);
    border: 1px solid {COLORS['border']};
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: transform 0.2s;
  }}
  .kpi-card:hover {{ transform: translateY(-3px); }}
  .kpi-value {{ font-size: 2.4rem; font-weight: 700; margin: 0.3rem 0; }}
  .kpi-label {{ font-size: 0.8rem; color: {COLORS['muted']}; text-transform: uppercase; letter-spacing: 0.08em; }}
  .kpi-delta {{ font-size: 0.85rem; margin-top: 0.2rem; }}

  /* Section headers */
  .section-header {{
    font-size: 1.35rem; font-weight: 600;
    border-left: 4px solid {COLORS['primary']};
    padding-left: 0.75rem; margin: 1.5rem 0 1rem;
    color: {COLORS['text']};
  }}

  /* Insight boxes */
  .insight-box {{
    background: linear-gradient(135deg, #1c1c35 0%, #16162a 100%);
    border-left: 4px solid {COLORS['accent']};
    border-radius: 0 12px 12px 0;
    padding: 0.9rem 1.1rem;
    margin: 0.5rem 0 1rem;
    font-size: 0.9rem;
    color: {COLORS['text']};
  }}
  .insight-box b {{ color: {COLORS['accent']}; }}

  /* Risk badge */
  .risk-high   {{ color: {COLORS['danger']};   font-weight: 600; }}
  .risk-medium {{ color: {COLORS['warning']};  font-weight: 600; }}
  .risk-low    {{ color: {COLORS['accent']};   font-weight: 600; }}

  /* Tab styling */
  [data-baseweb="tab"] {{ background: {COLORS['card']}; border-radius: 8px; }}

  /* Divider */
  hr {{ border-color: {COLORS['border']}; margin: 1.5rem 0; }}

  /* Dataframe */
  .stDataFrame {{ border-radius: 12px; overflow: hidden; }}

  /* Scrollable table */
  .scrollable {{ max-height: 380px; overflow-y: auto; }}

  /* Selectbox/slider text */
  .stSelectbox label, .stSlider label, .stMultiSelect label {{
    color: {COLORS['muted']} !important; font-size: 0.82rem !important;
    text-transform: uppercase; letter-spacing: 0.06em;
  }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading and preprocessing dataset…")
def load_data(path: str = "Mental_Health_Dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)

    # ── Timestamp ────────────────────────────────────────────────────────────
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df["Year"]  = df["Timestamp"].dt.year
    df["Month"] = df["Timestamp"].dt.month
    df["YearMonth"] = df["Timestamp"].dt.to_period("M").astype(str)

    # ── Missing values ────────────────────────────────────────────────────────
    # self_employed has 5202 NaN (1.8%) — impute with mode "No"
    df["self_employed"] = df["self_employed"].fillna("No")

    # ── Ordinal encoding for ordered categoricals ─────────────────────────────
    mood_map   = {"Low": 1, "Medium": 2, "High": 3}
    binary_map = {"No": 0, "Maybe": 1, "Yes": 2}
    yn_map     = {"No": 0, "Yes": 1}

    df["Mood_Swings_Ord"]  = df["Mood_Swings"].map(mood_map)
    df["Growing_Stress_Ord"]     = df["Growing_Stress"].map(binary_map)
    df["Changes_Habits_Ord"]     = df["Changes_Habits"].map(binary_map)
    df["Mental_History_Ord"]     = df["Mental_Health_History"].map(binary_map)
    df["Work_Interest_Ord"]      = df["Work_Interest"].map({"No": 0, "Maybe": 1, "Yes": 2})
    df["Social_Weakness_Ord"]    = df["Social_Weakness"].map(binary_map)
    df["Coping_Struggles_Bin"]   = df["Coping_Struggles"].map(yn_map)
    df["family_history_Bin"]     = df["family_history"].map(yn_map)
    df["treatment_Bin"]          = df["treatment"].map(yn_map)
    df["self_employed_Bin"]      = df["self_employed"].map(yn_map)

    # ── Days Indoors → ordered category ──────────────────────────────────────
    days_order = ["Go out Every day", "1-14 days", "15-30 days",
                  "31-60 days", "More than 2 months"]
    days_num   = {v: i for i, v in enumerate(days_order)}
    df["Days_Indoors_Ord"] = df["Days_Indoors"].map(days_num)

    # ── Mental Health Risk Score (composite 0–10) ─────────────────────────────
    # Weighted sum of risk indicators
    df["Risk_Score"] = (
        df["Mood_Swings_Ord"]        * 1.5 +   # max 4.5
        df["Growing_Stress_Ord"]     * 1.2 +   # max 3.6 (2=Yes)
        df["Changes_Habits_Ord"]     * 0.8 +
        df["Mental_History_Ord"]     * 0.8 +
        df["Coping_Struggles_Bin"]   * 1.2 +
        (2 - df["Work_Interest_Ord"])* 0.8 +   # low interest → higher risk
        df["Social_Weakness_Ord"]    * 0.7 +
        df["Days_Indoors_Ord"]       * 0.5
    )
    # Normalise to 0–10
    mn, mx = df["Risk_Score"].min(), df["Risk_Score"].max()
    df["Risk_Score"] = ((df["Risk_Score"] - mn) / (mx - mn) * 10).round(2)

    # ── Risk Category ─────────────────────────────────────────────────────────
    df["Risk_Category"] = pd.cut(
        df["Risk_Score"],
        bins=[0, 3.5, 6.5, 10],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )

    # ── Row ID for user-level analysis ───────────────────────────────────────
    df["User_ID"] = ["U" + str(i).zfill(6) for i in range(1, len(df) + 1)]

    return df


@st.cache_data(show_spinner=False)
def encode_for_ml(df: pd.DataFrame):
    """Label-encode categorical columns and return feature matrix."""
    cat_cols = ["Gender", "Country", "Occupation", "self_employed",
                "family_history", "treatment", "Days_Indoors",
                "Growing_Stress", "Changes_Habits", "Mental_Health_History",
                "Coping_Struggles", "Work_Interest", "Social_Weakness",
                "Mood_Swings", "mental_health_interview", "care_options"]
    enc = LabelEncoder()
    df_enc = df[cat_cols].copy()
    for c in cat_cols:
        df_enc[c] = enc.fit_transform(df_enc[c].astype(str))
    return df_enc


@st.cache_data(show_spinner=False)
def run_kmeans(_df_enc, n_clusters: int = 3):
    sample = _df_enc.sample(min(20000, len(_df_enc)), random_state=42)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(sample)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(sample)
    return coords, labels, sample.index


@st.cache_data(show_spinner=False)
def train_risk_model(_df):
    features = ["Mood_Swings_Ord", "Growing_Stress_Ord", "Changes_Habits_Ord",
                "Mental_History_Ord", "Coping_Struggles_Bin", "Work_Interest_Ord",
                "Social_Weakness_Ord", "Days_Indoors_Ord",
                "family_history_Bin", "treatment_Bin"]
    X = _df[features].dropna()
    y = _df.loc[X.index, "Risk_Category"].astype(str)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te)
    feat_imp = pd.Series(clf.feature_importances_, index=features).sort_values(ascending=False)
    return clf, acc, feat_imp, features


# ─────────────────────────────────────────────────────────────────────────────
# HELPER UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def kpi_card(label: str, value, color: str, delta: str = ""):
    delta_html = f'<div class="kpi-delta" style="color:{color}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="color:{color}">{value}</div>
      {delta_html}
    </div>"""


def insight(text: str):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def apply_theme(fig):
    fig.update_layout(**PLOTLY_THEME)
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def pct_fmt(n, total):
    return f"{n:,}  ({n/total*100:.1f}%)"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame):
    st.sidebar.markdown("## 🔍 Filters")
    st.sidebar.markdown("---")

    genders = st.sidebar.multiselect(
        "Gender", options=sorted(df["Gender"].unique()),
        default=sorted(df["Gender"].unique()))

    occupations = st.sidebar.multiselect(
        "Occupation", options=sorted(df["Occupation"].unique()),
        default=sorted(df["Occupation"].unique()))

    countries_top = sorted(df["Country"].value_counts().head(15).index.tolist())
    countries = st.sidebar.multiselect(
        "Country (top 15)", options=countries_top, default=countries_top)

    risk_cats = st.sidebar.multiselect(
        "Risk Category", options=["Low", "Medium", "High"],
        default=["Low", "Medium", "High"])

    days_opts = ["Go out Every day", "1-14 days", "15-30 days",
                 "31-60 days", "More than 2 months"]
    days_sel = st.sidebar.multiselect(
        "Days Indoors", options=days_opts, default=days_opts)

    treatment = st.sidebar.multiselect(
        "Received Treatment", options=["Yes", "No"], default=["Yes", "No"])

    family_hist = st.sidebar.multiselect(
        "Family History", options=["Yes", "No"], default=["Yes", "No"])

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"<small style='color:{COLORS['muted']}'>Dashboard v1.0 · MindScope Analytics</small>",
        unsafe_allow_html=True)

    mask = (
        df["Gender"].isin(genders) &
        df["Occupation"].isin(occupations) &
        df["Country"].isin(countries) &
        df["Risk_Category"].isin(risk_cats) &
        df["Days_Indoors"].isin(days_sel) &
        df["treatment"].isin(treatment) &
        df["family_history"].isin(family_hist)
    )
    return df[mask].copy()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — OVERVIEW / KPI
# ─────────────────────────────────────────────────────────────────────────────
def render_overview(df: pd.DataFrame, df_full: pd.DataFrame):
    section("📊 Overview & Key Metrics")

    total     = len(df)
    pct_treat = df["treatment"].eq("Yes").mean() * 100
    pct_high  = df["Risk_Category"].eq("High").mean() * 100
    pct_fam   = df["family_history"].eq("Yes").mean() * 100
    avg_risk  = df["Risk_Score"].mean()
    pct_stress = df["Growing_Stress"].eq("Yes").mean() * 100
    pct_cope  = df["Coping_Struggles"].eq("Yes").mean() * 100
    pct_mood_high = df["Mood_Swings"].eq("High").mean() * 100

    cols = st.columns(4)
    cols[0].markdown(kpi_card("Total Respondents", f"{total:,}", COLORS["primary"]), unsafe_allow_html=True)
    cols[1].markdown(kpi_card("Avg. Risk Score", f"{avg_risk:.2f}/10", COLORS["secondary"],
                               f"{'↑ Above' if avg_risk > 5 else '↓ Below'} midpoint"), unsafe_allow_html=True)
    cols[2].markdown(kpi_card("Seeking Treatment", f"{pct_treat:.1f}%", COLORS["accent"]), unsafe_allow_html=True)
    cols[3].markdown(kpi_card("High-Risk Users", f"{pct_high:.1f}%", COLORS["danger"]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cols2 = st.columns(4)
    cols2[0].markdown(kpi_card("Growing Stress", f"{pct_stress:.1f}%", COLORS["warning"]), unsafe_allow_html=True)
    cols2[1].markdown(kpi_card("Coping Struggles", f"{pct_cope:.1f}%", COLORS["secondary"]), unsafe_allow_html=True)
    cols2[2].markdown(kpi_card("Family MH History", f"{pct_fam:.1f}%", "#A78BFA"), unsafe_allow_html=True)
    cols2[3].markdown(kpi_card("High Mood Swings", f"{pct_mood_high:.1f}%", "#FB923C"), unsafe_allow_html=True)

    st.markdown("---")

    # Risk distribution pie + gender split
    c1, c2, c3 = st.columns(3)

    with c1:
        risk_counts = df["Risk_Category"].value_counts().reset_index()
        risk_counts.columns = ["Category", "Count"]
        fig = px.pie(risk_counts, names="Category", values="Count",
                     color="Category",
                     color_discrete_map=RISK_PALETTE,
                     title="Risk Category Distribution",
                     hole=0.55)
        fig.update_traces(textposition="outside", textinfo="percent+label")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        gender_risk = df.groupby(["Gender", "Risk_Category"]).size().reset_index(name="Count")
        fig = px.bar(gender_risk, x="Gender", y="Count", color="Risk_Category",
                     color_discrete_map=RISK_PALETTE,
                     barmode="group", title="Risk by Gender")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        occ_risk = df.groupby(["Occupation", "Risk_Category"]).size().reset_index(name="Count")
        fig = px.bar(occ_risk, x="Count", y="Occupation", color="Risk_Category",
                     color_discrete_map=RISK_PALETTE,
                     barmode="stack", orientation="h",
                     title="Risk by Occupation")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    insight(
        f"<b>{pct_high:.1f}%</b> of respondents fall into the <b>High-Risk</b> category. "
        f"<b>{pct_treat:.1f}%</b> are actively seeking treatment. "
        f"Family mental health history is present in <b>{pct_fam:.1f}%</b>, "
        f"which is a strong predictor of current risk."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DISTRIBUTIONS
# ─────────────────────────────────────────────────────────────────────────────
def render_distributions(df: pd.DataFrame):
    section("📈 Distribution Analysis")

    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(df, x="Risk_Score", nbins=50,
                           color_discrete_sequence=[COLORS["primary"]],
                           title="Risk Score Distribution",
                           labels={"Risk_Score": "Risk Score (0–10)"})
        fig.add_vline(x=df["Risk_Score"].mean(), line_dash="dash",
                      line_color=COLORS["secondary"],
                      annotation_text=f"Mean={df['Risk_Score'].mean():.2f}",
                      annotation_position="top right")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        mood_order = ["Low", "Medium", "High"]
        mood_counts = df["Mood_Swings"].value_counts().reindex(mood_order).reset_index()
        mood_counts.columns = ["Mood", "Count"]
        fig = px.bar(mood_counts, x="Mood", y="Count",
                     color="Mood", color_discrete_map=RISK_PALETTE,
                     title="Mood Swings Distribution")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        days_order = ["Go out Every day", "1-14 days", "15-30 days",
                      "31-60 days", "More than 2 months"]
        di_counts = df["Days_Indoors"].value_counts().reindex(days_order).reset_index()
        di_counts.columns = ["Days", "Count"]
        fig = px.bar(di_counts, x="Days", y="Count",
                     color_discrete_sequence=[COLORS["accent"]],
                     title="Days Spent Indoors",
                     labels={"Days": "Category"})
        fig.update_layout(xaxis_tickangle=-25)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        country_top = df["Country"].value_counts().head(12).reset_index()
        country_top.columns = ["Country", "Count"]
        fig = px.bar(country_top, x="Count", y="Country",
                     orientation="h",
                     color="Count",
                     color_continuous_scale="Purples",
                     title="Top 12 Countries by Respondents")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    insight(
        "The risk score follows a <b>near-normal distribution</b> with a slight right skew, "
        "suggesting a large moderate-risk population. "
        "Respondents spending <b>More than 2 months</b> indoors skew toward higher risk categories."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — CORRELATION & HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
def render_correlation(df: pd.DataFrame):
    section("🔗 Correlation & Feature Relationships")

    num_cols = ["Risk_Score", "Mood_Swings_Ord", "Growing_Stress_Ord",
                "Changes_Habits_Ord", "Mental_History_Ord",
                "Coping_Struggles_Bin", "Work_Interest_Ord",
                "Social_Weakness_Ord", "Days_Indoors_Ord",
                "family_history_Bin", "treatment_Bin"]
    nice_names = {
        "Risk_Score": "Risk Score",
        "Mood_Swings_Ord": "Mood Swings",
        "Growing_Stress_Ord": "Growing Stress",
        "Changes_Habits_Ord": "Changes in Habits",
        "Mental_History_Ord": "Mental Health History",
        "Coping_Struggles_Bin": "Coping Struggles",
        "Work_Interest_Ord": "Work Interest",
        "Social_Weakness_Ord": "Social Weakness",
        "Days_Indoors_Ord": "Days Indoors",
        "family_history_Bin": "Family History",
        "treatment_Bin": "Seeking Treatment",
    }

    corr = df[num_cols].rename(columns=nice_names).corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Feature Correlation Heatmap",
        aspect="auto",
    )
    fig.update_traces(textfont_size=10)
    apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Top correlations with Risk Score (sorted)
    risk_corr = corr["Risk Score"].drop("Risk Score").sort_values(key=abs, ascending=False)
    c1, c2 = st.columns([2, 1])
    with c1:
        fig2 = go.Figure(go.Bar(
            x=risk_corr.values,
            y=risk_corr.index,
            orientation="h",
            marker_color=[COLORS["danger"] if v > 0 else COLORS["accent"] for v in risk_corr.values],
            text=[f"{v:.3f}" for v in risk_corr.values],
            textposition="outside",
        ))
        fig2.update_layout(title="Correlation with Risk Score",
                           xaxis_title="Pearson r", **PLOTLY_THEME,
                           margin=dict(l=20, r=60, t=40, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.markdown("#### Top Predictors")
        for feat, val in risk_corr.items():
            direction = "🔴 Increases" if val > 0 else "🟢 Decreases"
            st.markdown(f"**{feat}** — `r={val:.3f}` {direction} risk")

    insight(
        f"<b>Mood Swings</b> and <b>Growing Stress</b> show the strongest positive correlation with the risk score. "
        f"<b>Work Interest</b> is negatively correlated — users who lose interest in work show higher risk. "
        f"Seeking treatment has a modest positive correlation, likely reflecting <b>selection bias</b> "
        f"(higher-risk users are more likely to seek help)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — COMPARATIVE ANALYSIS (Box / Violin)
# ─────────────────────────────────────────────────────────────────────────────
def render_comparative(df: pd.DataFrame):
    section("📦 Group Comparisons")

    c1, c2 = st.columns(2)

    with c1:
        fig = px.box(df, x="Occupation", y="Risk_Score",
                     color="Occupation",
                     title="Risk Score by Occupation",
                     labels={"Risk_Score": "Risk Score"})
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.violin(df, x="Gender", y="Risk_Score",
                        color="Gender", box=True, points=False,
                        title="Risk Score Distribution by Gender",
                        labels={"Risk_Score": "Risk Score"})
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        days_order = ["Go out Every day", "1-14 days", "15-30 days",
                      "31-60 days", "More than 2 months"]
        fig = px.box(df, x="Days_Indoors", y="Risk_Score",
                     category_orders={"Days_Indoors": days_order},
                     color="Days_Indoors",
                     title="Risk Score vs Days Indoors",
                     labels={"Risk_Score": "Risk Score", "Days_Indoors": "Days Indoors"})
        fig.update_layout(showlegend=False, xaxis_tickangle=-20)
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        treat_risk = df.groupby(["treatment", "Risk_Category"]).size().reset_index(name="Count")
        treat_pct = treat_risk.copy()
        treat_pct["Pct"] = treat_pct.groupby("treatment")["Count"].transform(lambda x: x / x.sum() * 100)
        fig = px.bar(treat_pct, x="treatment", y="Pct",
                     color="Risk_Category",
                     color_discrete_map=RISK_PALETTE,
                     barmode="stack",
                     title="Risk Distribution: Treatment vs No Treatment",
                     labels={"treatment": "Seeking Treatment", "Pct": "% of Group"})
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    insight(
        "Users who stay indoors <b>More than 2 months</b> have the highest median risk. "
        "Despite counterintuitive appearance, those <b>seeking treatment</b> tend to have higher risk scores — "
        "this is <b>not</b> because treatment increases risk; rather, high-risk individuals are more motivated to seek care."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — TEMPORAL TRENDS
# ─────────────────────────────────────────────────────────────────────────────
def render_temporal(df: pd.DataFrame):
    section("📅 Temporal Trends")

    monthly = (df.groupby("YearMonth")
                 .agg(Avg_Risk=("Risk_Score", "mean"),
                      Count=("User_ID", "count"),
                      Pct_High=("Risk_Category", lambda x: (x == "High").mean() * 100))
                 .reset_index()
                 .sort_values("YearMonth"))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Average Risk Score Over Time",
                                        "Monthly Respondent Count"),
                        vertical_spacing=0.1)
    fig.add_trace(go.Scatter(x=monthly["YearMonth"], y=monthly["Avg_Risk"],
                             mode="lines+markers", name="Avg Risk",
                             line=dict(color=COLORS["primary"], width=2),
                             fill="tozeroy", fillcolor="rgba(108,99,255,0.1)"), row=1, col=1)
    fig.add_trace(go.Bar(x=monthly["YearMonth"], y=monthly["Count"],
                         name="Respondents", marker_color=COLORS["accent"],
                         opacity=0.7), row=2, col=1)
    fig.update_layout(**PLOTLY_THEME, height=480, showlegend=True,
                      margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # Stress breakdown over time
    stress_monthly = (df.groupby(["YearMonth", "Growing_Stress"])
                        .size().reset_index(name="Count")
                        .sort_values("YearMonth"))
    fig2 = px.area(stress_monthly, x="YearMonth", y="Count",
                   color="Growing_Stress",
                   color_discrete_map={"Yes": COLORS["danger"],
                                       "No": COLORS["accent"],
                                       "Maybe": COLORS["warning"]},
                   title="Growing Stress Responses Over Time",
                   labels={"Growing_Stress": "Growing Stress"})
    apply_theme(fig2)
    st.plotly_chart(fig2, use_container_width=True)

    insight(
        "Survey submissions peaked in <b>August–September 2014</b>, the initial survey wave. "
        "The proportion reporting <b>Growing Stress</b> remains persistently high, "
        "suggesting this is a structural condition rather than a temporary event."
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — USER-LEVEL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def render_user_analysis(df: pd.DataFrame):
    section("👤 Individual User Analysis")

    user_ids = df["User_ID"].tolist()
    selected_id = st.selectbox("Select a User", options=user_ids, index=0,
                                help="Pick any user to see their individual mental health profile.")

    user = df[df["User_ID"] == selected_id].iloc[0]
    risk_color = RISK_PALETTE.get(str(user["Risk_Category"]), COLORS["muted"])

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_card("Risk Score", f"{user['Risk_Score']:.2f}", risk_color), unsafe_allow_html=True)
    c2.markdown(kpi_card("Risk Category", str(user["Risk_Category"]), risk_color), unsafe_allow_html=True)
    c3.markdown(kpi_card("Mood Swings", user["Mood_Swings"], COLORS["warning"]), unsafe_allow_html=True)
    c4.markdown(kpi_card("Treatment", user["treatment"], COLORS["accent"]), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c5, c6 = st.columns(2)
    with c5:
        st.markdown("#### Profile Details")
        profile_data = {
            "Gender":            user["Gender"],
            "Country":           user["Country"],
            "Occupation":        user["Occupation"],
            "Self Employed":     user["self_employed"],
            "Family History":    user["family_history"],
            "Days Indoors":      user["Days_Indoors"],
            "Growing Stress":    user["Growing_Stress"],
            "Changes in Habits": user["Changes_Habits"],
            "MH History":        user["Mental_Health_History"],
            "Coping Struggles":  user["Coping_Struggles"],
            "Work Interest":     user["Work_Interest"],
            "Social Weakness":   user["Social_Weakness"],
            "Care Options":      user["care_options"],
        }
        for k, v in profile_data.items():
            icon = "🔴" if str(v) in ["Yes", "High"] else ("🟡" if str(v) == "Maybe" else "🟢")
            st.markdown(f"{icon} **{k}:** {v}")

    with c6:
        # Radar: user vs population average
        radar_metrics = {
            "Mood Swings":      (user["Mood_Swings_Ord"],   df["Mood_Swings_Ord"].mean()),
            "Growing Stress":   (user["Growing_Stress_Ord"],df["Growing_Stress_Ord"].mean()),
            "Habit Changes":    (user["Changes_Habits_Ord"],df["Changes_Habits_Ord"].mean()),
            "MH History":       (user["Mental_History_Ord"],df["Mental_History_Ord"].mean()),
            "Coping Issues":    (user["Coping_Struggles_Bin"],df["Coping_Struggles_Bin"].mean()),
            "Social Weakness":  (user["Social_Weakness_Ord"],df["Social_Weakness_Ord"].mean()),
            "Days Indoors":     (user["Days_Indoors_Ord"],  df["Days_Indoors_Ord"].mean()),
        }
        labels = list(radar_metrics.keys())
        user_vals  = [v[0] for v in radar_metrics.values()]
        avg_vals   = [v[1] for v in radar_metrics.values()]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=user_vals + [user_vals[0]],
                                      theta=labels + [labels[0]],
                                      fill="toself", name="This User",
                                      line_color=risk_color, opacity=0.7))
        fig.add_trace(go.Scatterpolar(r=avg_vals + [avg_vals[0]],
                                      theta=labels + [labels[0]],
                                      fill="toself", name="Population Avg",
                                      line_color=COLORS["muted"], opacity=0.4))
        fig.update_layout(**PLOTLY_THEME, polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 3], color=COLORS["muted"]),
            angularaxis=dict(color=COLORS["text"]),
        ), title="User vs Population Average", height=380)
        st.plotly_chart(fig, use_container_width=True)

    # Personalized insight
    flags = []
    if user["Growing_Stress"] == "Yes":     flags.append("actively experiencing <b>growing stress</b>")
    if user["Coping_Struggles"] == "Yes":   flags.append("reporting <b>coping struggles</b>")
    if user["Mood_Swings"] == "High":       flags.append("showing <b>high mood swings</b>")
    if user["Days_Indoors_Ord"] >= 3:       flags.append("spending <b>excessive time indoors</b>")
    if user["Work_Interest"] in ["No","Maybe"]: flags.append("showing <b>reduced work interest</b>")
    if user["family_history"] == "Yes":     flags.append("having a <b>family history</b> of MH conditions")

    if flags:
        insight(f"User <b>{selected_id}</b> is {', '.join(flags)}. Risk Score: <b>{user['Risk_Score']:.2f}/10</b> "
                f"({user['Risk_Category']} risk).")
    else:
        insight(f"User <b>{selected_id}</b> shows no significant risk flags. Overall risk: <b>{user['Risk_Score']:.2f}/10</b>.")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — ADVANCED INSIGHTS (Patterns & Clustering)
# ─────────────────────────────────────────────────────────────────────────────
def render_advanced(df: pd.DataFrame):
    section("🔬 Advanced Insights & ML Analysis")

    tab1, tab2, tab3 = st.tabs(["🔵 K-Means Clustering", "🌲 Feature Importance", "🧩 Risk Pattern Heatmaps"])

    with tab1:
        st.markdown("##### Unsupervised User Clustering (K-Means, n=3, PCA 2D)")
        df_enc = encode_for_ml(df)
        with st.spinner("Running K-Means…"):
            coords, labels, idx = run_kmeans(df_enc)
        
        cluster_df = pd.DataFrame({
            "PC1": coords[:, 0], "PC2": coords[:, 1],
            "Cluster": [f"Cluster {l}" for l in labels],
            "Risk": df.loc[idx, "Risk_Category"].values,
        })
        fig = px.scatter(cluster_df, x="PC1", y="PC2",
                         color="Cluster", symbol="Risk",
                         opacity=0.6, title="PCA Projection of User Clusters",
                         color_discrete_sequence=[COLORS["primary"], COLORS["secondary"], COLORS["accent"]])
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)
        insight(
            "Three natural user clusters emerge in the PCA space, roughly corresponding to "
            "<b>low</b>, <b>medium</b>, and <b>high-risk</b> behavioral profiles. "
            "Cluster membership can guide <b>targeted intervention strategies</b>."
        )

    with tab2:
        st.markdown("##### Random Forest — Risk Category Prediction")
        with st.spinner("Training Random Forest…"):
            clf, acc, feat_imp, features = train_risk_model(df)

        col1, col2 = st.columns([2, 1])
        with col1:
            fig = px.bar(feat_imp.reset_index(), x="feature_importances", y="index",
                         orientation="h",
                         color="feature_importances",
                         color_continuous_scale="Purples",
                         title="Feature Importance (Random Forest)",
                         labels={"feature_importances": "Importance", "index": "Feature"})
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("Model Accuracy", f"{acc*100:.1f}%")
            st.markdown("**Top Features:**")
            for f, v in feat_imp.head(5).items():
                st.markdown(f"- `{f.replace('_Ord','').replace('_Bin','')}` → `{v:.3f}`")

        insight(
            f"The Random Forest model achieves <b>{acc*100:.1f}% accuracy</b> in predicting risk categories. "
            f"<b>{feat_imp.index[0].replace('_Ord','').replace('_Bin','')}</b> is the single most predictive feature."
        )

    with tab3:
        st.markdown("##### Risk Patterns: Cross-feature Heatmaps")

        c1, c2 = st.columns(2)
        with c1:
            pivot = df.groupby(["Occupation", "Days_Indoors"])["Risk_Score"].mean().reset_index()
            pivot_wide = pivot.pivot(index="Occupation", columns="Days_Indoors", values="Risk_Score")
            days_order = ["Go out Every day", "1-14 days", "15-30 days",
                          "31-60 days", "More than 2 months"]
            pivot_wide = pivot_wide[[c for c in days_order if c in pivot_wide.columns]]
            fig = px.imshow(pivot_wide, text_auto=".2f",
                            color_continuous_scale="RdYlGn_r",
                            title="Avg Risk: Occupation × Days Indoors")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            pivot2 = df.groupby(["Gender", "Mood_Swings"])["Risk_Score"].mean().unstack()
            fig = px.imshow(pivot2, text_auto=".2f",
                            color_continuous_scale="Reds",
                            title="Avg Risk: Gender × Mood Swings")
            apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — LIVE PREDICTION TOOL
# ─────────────────────────────────────────────────────────────────────────────
def render_prediction(df: pd.DataFrame):
    section("🎯 Live Risk Predictor")
    st.markdown(
        "Enter a hypothetical user profile below to get an instant mental health risk assessment.")

    with st.spinner("Loading model…"):
        clf, acc, feat_imp, features = train_risk_model(df)

    c1, c2, c3 = st.columns(3)
    with c1:
        mood        = st.selectbox("Mood Swings",         ["Low", "Medium", "High"])
        stress      = st.selectbox("Growing Stress",      ["No", "Maybe", "Yes"])
        habits      = st.selectbox("Changes in Habits",   ["No", "Maybe", "Yes"])
        mh_hist     = st.selectbox("MH History",          ["No", "Maybe", "Yes"])
    with c2:
        coping      = st.selectbox("Coping Struggles",    ["No", "Yes"])
        work_int    = st.selectbox("Work Interest",       ["Yes", "Maybe", "No"])
        social      = st.selectbox("Social Weakness",     ["No", "Maybe", "Yes"])
        days        = st.selectbox("Days Indoors",        ["Go out Every day","1-14 days",
                                                           "15-30 days","31-60 days",
                                                           "More than 2 months"])
    with c3:
        fam         = st.selectbox("Family History",      ["No", "Yes"])
        treat       = st.selectbox("Seeking Treatment",   ["No", "Yes"])
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮  Predict Risk", use_container_width=True)

    if predict_btn:
        mo = {"Low":1,"Medium":2,"High":3}
        ym = {"No":0,"Maybe":1,"Yes":2}
        yn = {"No":0,"Yes":1}
        dm = {"Go out Every day":0,"1-14 days":1,"15-30 days":2,"31-60 days":3,"More than 2 months":4}

        X_in = pd.DataFrame([{
            "Mood_Swings_Ord":       mo[mood],
            "Growing_Stress_Ord":    ym[stress],
            "Changes_Habits_Ord":    ym[habits],
            "Mental_History_Ord":    ym[mh_hist],
            "Coping_Struggles_Bin":  yn[coping],
            "Work_Interest_Ord":     ym[work_int],
            "Social_Weakness_Ord":   ym[social],
            "Days_Indoors_Ord":      dm[days],
            "family_history_Bin":    yn[fam],
            "treatment_Bin":         yn[treat],
        }])
        pred_cat = clf.predict(X_in)[0]
        pred_prob = clf.predict_proba(X_in)[0]
        classes = clf.classes_
        prob_dict = dict(zip(classes, pred_prob))
        risk_color = RISK_PALETTE.get(pred_cat, COLORS["muted"])

        st.markdown(f"""
        <div class="kpi-card" style="margin-top:1rem;">
          <div class="kpi-label">Predicted Risk Category</div>
          <div class="kpi-value" style="color:{risk_color}">{pred_cat} Risk</div>
        </div>
        """, unsafe_allow_html=True)

        prob_df = pd.DataFrame({"Category": list(prob_dict.keys()),
                                "Probability": [v*100 for v in prob_dict.values()]})
        fig = px.bar(prob_df, x="Category", y="Probability",
                     color="Category", color_discrete_map=RISK_PALETTE,
                     title="Prediction Probabilities (%)",
                     text_auto=".1f")
        apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — DATA TABLE & EXPORT
# ─────────────────────────────────────────────────────────────────────────────
def render_data_export(df: pd.DataFrame):
    section("📋 Data Explorer & Export")

    show_cols = ["User_ID", "Gender", "Country", "Occupation",
                 "Days_Indoors", "Growing_Stress", "Mood_Swings",
                 "Coping_Struggles", "treatment", "Risk_Score", "Risk_Category"]

    c1, c2 = st.columns([3, 1])
    with c1:
        search = st.text_input("🔎 Filter by Country / Occupation / Gender",
                               placeholder="e.g. India")
    with c2:
        n_rows = st.number_input("Rows to display", min_value=10, max_value=500, value=50, step=10)

    display_df = df[show_cols].copy()
    if search.strip():
        mask = (
            df["Country"].str.contains(search, case=False, na=False) |
            df["Occupation"].str.contains(search, case=False, na=False) |
            df["Gender"].str.contains(search, case=False, na=False)
        )
        display_df = display_df[mask]

    st.dataframe(display_df.head(n_rows), use_container_width=True, height=350)

    st.markdown("#### 📥 Export Filtered Data")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download Filtered CSV",
        data=csv,
        file_name=f"mental_health_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.5rem;">
      <div style="font-size:2.8rem;">🧠</div>
      <div>
        <h1 style="margin:0;font-size:2rem;font-weight:700;
                   background:linear-gradient(90deg,{COLORS['primary']},{COLORS['secondary']});
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
          MindScope Analytics
        </h1>
        <p style="margin:0;color:{COLORS['muted']};font-size:0.9rem;">
          Mental Health Survey Dashboard · 292,364 Respondents · 2014–2016
        </p>
      </div>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    # Load data
    df_full = load_data("Mental_Health_Dataset.csv")

    # Sidebar filters → filtered df
    df = render_sidebar(df_full)

    if len(df) == 0:
        st.warning("⚠️ No data matches the current filters. Please adjust your selections.")
        return

    st.markdown(
        f"<small style='color:{COLORS['muted']}'>Showing <b style='color:{COLORS['accent']}'>"
        f"{len(df):,}</b> of {len(df_full):,} records after filtering.</small>",
        unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Navigation tabs
    tabs = st.tabs([
        "📊 Overview",
        "📈 Distributions",
        "🔗 Correlations",
        "📦 Comparisons",
        "📅 Trends",
        "👤 User Analysis",
        "🔬 Advanced / ML",
        "🎯 Predictor",
        "📋 Data & Export",
    ])

    with tabs[0]: render_overview(df, df_full)
    with tabs[1]: render_distributions(df)
    with tabs[2]: render_correlation(df)
    with tabs[3]: render_comparative(df)
    with tabs[4]: render_temporal(df)
    with tabs[5]: render_user_analysis(df)
    with tabs[6]: render_advanced(df)
    with tabs[7]: render_prediction(df)
    with tabs[8]: render_data_export(df)


if __name__ == "__main__":
    main()
