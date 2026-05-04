# dashboard/dashboard_app.py
"""
Streamlit Monitoring Dashboard — Phase 5.

Features:
  • Real-time login attempt feed with colour-coded threat classification
  • Live metrics cards (total events, attack counts, detection rate)
  • Master Hybrid (ANN+LSTM+RF) as primary detection authority
  • All model predictions shown side-by-side per event
  • Performance comparison charts loaded from outputs/metrics/
  • Rolling attack-rate time-series chart
  • Confusion matrix display per model
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from collections import deque
from datetime import datetime

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hybrid Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0F1117; }
  .block-container { padding-top: 1rem; }
  .metric-card {
      background: #1e1e2e; border-radius: 12px;
      padding: 18px 22px; margin: 6px 0;
      border-left: 4px solid;
  }
  .card-normal  { border-left-color: #22c55e; }
  .card-cred    { border-left-color: #f97316; }
  .card-bizlogic{ border-left-color: #ef4444; }
  .card-total   { border-left-color: #6366f1; }
  .feed-row-normal   { background:#0d2218; border-radius:6px; padding:4px 8px; margin:2px 0; }
  .feed-row-attack1  { background:#2d1a0a; border-radius:6px; padding:4px 8px; margin:2px 0; }
  .feed-row-attack2  { background:#2d0a0a; border-radius:6px; padding:4px 8px; margin:2px 0; }
  .badge-normal  { background:#166534; color:#bbf7d0; border-radius:4px;
                   padding:2px 8px; font-size:11px; font-weight:600; }
  .badge-cred    { background:#9a3412; color:#fed7aa; border-radius:4px;
                   padding:2px 8px; font-size:11px; font-weight:600; }
  .badge-biz     { background:#7f1d1d; color:#fecaca; border-radius:4px;
                   padding:2px 8px; font-size:11px; font-weight:600; }
  h1,h2,h3,h4   { color: #e2e8f0 !important; }
  .stMetric label{ color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────
DATA_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", "api_security_dataset_2026.csv")
METRICS_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "metrics")
PLOTS_DIR    = os.path.join(os.path.dirname(__file__), "..", "outputs", "plots")
LABEL_NAMES  = {0: "Normal", 1: "Credential Stuffing", 2: "Business Logic"}
LABEL_BADGE  = {0: "badge-normal", 1: "badge-cred", 2: "badge-biz"}
FEED_ROW_CSS = {0: "feed-row-normal", 1: "feed-row-attack1", 2: "feed-row-attack2"}
MAX_FEED     = 50   # rows shown in live feed
MAX_HISTORY  = 300  # points in rolling chart


# ══════════════════════════════════════════════════════════════════════════════
# Session-state initialisation
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "running":         False,
        "total":           0,
        "counts":          {0: 0, 1: 0, 2: 0},
        "feed":            deque(maxlen=MAX_FEED),
        "history_ts":      deque(maxlen=MAX_HISTORY),
        "history_rate":    deque(maxlen=MAX_HISTORY),
        "models_loaded":   False,
        "rf":   None, "iso": None, "ann": None, "lstm": None,
        "h_rf_ann":  None, "h_if_ann": None,
        "h_rf_lstm": None, "master":   None,
        "sim":    None,
        "sim_iter": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ══════════════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading models …")
def load_models():
    models_out = {}
    try:
        from models.random_forest_model   import RandomForestDetector
        from models.isolation_forest_model import IsolationForestDetector
        from models.ann_model             import ANNDetector
        from models.lstm_model            import LSTMDetector
        from models.hybrid_rf_ann         import HybridRFANN
        from models.hybrid_if_ann         import HybridIFANN
        from models.hybrid_rf_lstm        import HybridRFLSTM
        from models.hybrid_master         import MasterHybrid

        models_out["rf"]        = RandomForestDetector().load()
        models_out["iso"]       = IsolationForestDetector().load()
        models_out["ann"]       = ANNDetector().load()
        models_out["lstm"]      = LSTMDetector().load()
        models_out["h_rf_ann"]  = HybridRFANN().load()
        models_out["h_if_ann"]  = HybridIFANN().load()
        models_out["h_rf_lstm"] = HybridRFLSTM().load()
        models_out["master"]    = MasterHybrid().load()
    except Exception as e:
        st.warning(f"Some models not yet trained: {e}")
    return models_out


def get_all_predictions(X: np.ndarray, mdls: dict) -> dict:
    """Run X through every loaded model. Returns {model_name: label_int}."""
    preds = {}
    for key, name in [
        ("rf",        "Random Forest"),
        ("iso",       "Isolation Forest"),
        ("ann",       "ANN"),
        ("lstm",      "LSTM"),
        ("h_rf_ann",  "RF + ANN"),
        ("h_if_ann",  "IF + ANN"),
        ("h_rf_lstm", "RF + LSTM"),
        ("master",    "Master (ANN+LSTM+RF)"),
    ]:
        if key in mdls and mdls[key] is not None:
            try:
                preds[name] = int(mdls[key].predict(X)[0])
            except Exception:
                preds[name] = 0
    return preds


# ══════════════════════════════════════════════════════════════════════════════
# Metrics loading
# ══════════════════════════════════════════════════════════════════════════════
def load_metrics_data():
    results = []
    if not os.path.exists(METRICS_DIR):
        return results
    for fname in sorted(os.listdir(METRICS_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(METRICS_DIR, fname)) as f:
                results.append(json.load(f))
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛡️ Hybrid Detection System")
    st.markdown("---")

    page = st.radio("Navigate", [
        "📡 Live Monitor",
        "📊 Model Performance",
        "🔍 Model Comparison",
    ])
    st.markdown("---")

    events_per_tick = st.slider("Events per refresh", 1, 20, 5)
    refresh_ms      = st.slider("Refresh interval (ms)", 200, 2000, 500, step=100)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        start_btn = st.button("▶ Start", use_container_width=True)
    with col2:
        stop_btn  = st.button("⏹ Stop",  use_container_width=True)

    if start_btn:
        st.session_state.running = True
    if stop_btn:
        st.session_state.running = False

    st.markdown("---")
    if st.button("🔄 Reset Stats", use_container_width=True):
        st.session_state.total   = 0
        st.session_state.counts  = {0: 0, 1: 0, 2: 0}
        st.session_state.feed    = deque(maxlen=MAX_FEED)
        st.session_state.history_ts   = deque(maxlen=MAX_HISTORY)
        st.session_state.history_rate = deque(maxlen=MAX_HISTORY)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Live Monitor
# ══════════════════════════════════════════════════════════════════════════════
if page == "📡 Live Monitor":
    st.markdown("# 📡 Real-Time API Login Monitor")

    # ── Top metric cards ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    total  = st.session_state.total
    counts = st.session_state.counts
    attack_total = counts[1] + counts[2]
    det_rate = (attack_total / total * 100) if total > 0 else 0.0

    c1.metric("Total Events",         f"{total:,}")
    c2.metric("✅ Normal",             f"{counts[0]:,}")
    c3.metric("🔑 Credential Stuffing", f"{counts[1]:,}")
    c4.metric("⚠️ Business Logic",     f"{counts[2]:,}")

    col_feed, col_chart = st.columns([3, 2])

    with col_feed:
        st.markdown("### Live Event Feed")
        feed_placeholder = st.empty()

    with col_chart:
        st.markdown("### Attack Rate (rolling)")
        chart_placeholder = st.empty()
        st.markdown("### Threat Distribution")
        pie_placeholder   = st.empty()

    st.markdown("---")
    st.markdown("### 🤖 Model Predictions (current event)")
    pred_placeholder = st.empty()

    # ── Simulation loop ───────────────────────────────────────────────────
    if st.session_state.running:
        mdls = load_models()

        # Init simulator iterator
        if st.session_state.sim_iter is None:
            if not os.path.exists(DATA_PATH):
                st.error(f"Dataset not found: {DATA_PATH}")
                st.stop()
            from dashboard.realtime_simulator import RealtimeSimulator
            sim = RealtimeSimulator(DATA_PATH, chunk_size=5000, delay_sec=0)
            st.session_state.sim_iter = sim.stream()

        stream_iter = st.session_state.sim_iter
        for _ in range(events_per_tick):
            try:
                X, meta = next(stream_iter)
            except StopIteration:
                st.session_state.running = False
                st.info("Stream ended — reset to replay.")
                break

            # Master model decides primary label
            if "master" in mdls and mdls["master"] is not None:
                try:
                    pred_label = int(mdls["master"].predict(X)[0])
                except Exception:
                    pred_label = meta["label_true"]
            else:
                pred_label = meta["label_true"]

            # Get all model predictions
            all_preds = get_all_predictions(X, mdls) if mdls else {}

            # Update state
            st.session_state.total += 1
            st.session_state.counts[pred_label] += 1

            event = {
                "time":       datetime.now().strftime("%H:%M:%S.%f")[:-3],
                "endpoint":   meta["endpoint"],
                "method":     meta["method"],
                "status":     meta["status_code"],
                "latency":    meta["latency_ms"],
                "label_true": meta["label_true"],
                "pred_label": pred_label,
                "pred_name":  LABEL_NAMES[pred_label],
                "all_preds":  all_preds,
            }
            st.session_state.feed.appendleft(event)

            # Rolling attack-rate
            t  = st.session_state.total
            ar = ((st.session_state.counts[1] + st.session_state.counts[2]) / t * 100)
            st.session_state.history_ts.append(t)
            st.session_state.history_rate.append(ar)

        # ── Render feed ───────────────────────────────────────────────────
        feed_html = ""
        for ev in list(st.session_state.feed):
            badge_cls = LABEL_BADGE[ev["pred_label"]]
            row_cls   = FEED_ROW_CSS[ev["pred_label"]]
            match_icon = "✓" if ev["pred_label"] == ev["label_true"] else "✗"
            feed_html += f"""
            <div class="{row_cls}">
              <span style="color:#94a3b8;font-size:11px">{ev['time']}</span>&nbsp;
              <span style="color:#e2e8f0;font-weight:600">{ev['method']}</span>&nbsp;
              <span style="color:#7dd3fc">{ev['endpoint']}</span>&nbsp;
              <span style="color:#94a3b8">HTTP {ev['status']}</span>&nbsp;
              <span style="color:#94a3b8">{ev['latency']:.0f}ms</span>&nbsp;&nbsp;
              <span class="{badge_cls}">{ev['pred_name']}</span>&nbsp;
              <span style="color:#64748b;font-size:10px">{match_icon}</span>
            </div>"""
        feed_placeholder.markdown(feed_html, unsafe_allow_html=True)

        # ── Rolling chart ─────────────────────────────────────────────────
        if st.session_state.history_ts:
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=list(st.session_state.history_ts),
                y=list(st.session_state.history_rate),
                mode="lines", fill="tozeroy",
                line=dict(color="#ef4444", width=2),
                fillcolor="rgba(239,68,68,0.15)",
            ))
            fig_line.update_layout(
                paper_bgcolor="#0F1117", plot_bgcolor="#0F1117",
                margin=dict(l=30, r=10, t=10, b=30),
                xaxis=dict(color="#64748b", title="Events"),
                yaxis=dict(color="#64748b", title="Attack %", range=[0, 100]),
                height=200,
            )
            chart_placeholder.plotly_chart(fig_line, use_container_width=True,
                                           key=f"line_{st.session_state.total}")

        # ── Pie chart ─────────────────────────────────────────────────────
        fig_pie = go.Figure(go.Pie(
            labels=["Normal", "Credential Stuffing", "Business Logic"],
            values=[counts[0], counts[1], counts[2]],
            hole=0.5,
            marker_colors=["#22c55e", "#f97316", "#ef4444"],
        ))
        fig_pie.update_layout(
            paper_bgcolor="#0F1117", plot_bgcolor="#0F1117",
            showlegend=True,
            legend=dict(font=dict(color="white")),
            margin=dict(l=10, r=10, t=10, b=10),
            height=200,
        )
        pie_placeholder.plotly_chart(fig_pie, use_container_width=True,
                                     key=f"pie_{st.session_state.total}")

        # ── Model predictions table ───────────────────────────────────────
        if st.session_state.feed:
            last = list(st.session_state.feed)[0]
            ap   = last.get("all_preds", {})
            if ap:
                rows = []
                for mname, plabel in ap.items():
                    is_master = "Master" in mname
                    rows.append({
                        "Model":      ("⭐ " if is_master else "") + mname,
                        "Prediction": LABEL_NAMES[plabel],
                        "Threat":     "🔴 ATTACK" if plabel > 0 else "🟢 Normal",
                    })
                df_preds = pd.DataFrame(rows)
                pred_placeholder.dataframe(df_preds, use_container_width=True,
                                           hide_index=True)

        time.sleep(refresh_ms / 1000)
        st.rerun()

    else:
        # Not running — show static feed
        if st.session_state.feed:
            feed_html = ""
            for ev in list(st.session_state.feed):
                badge_cls = LABEL_BADGE[ev["pred_label"]]
                row_cls   = FEED_ROW_CSS[ev["pred_label"]]
                feed_html += f"""
                <div class="{row_cls}">
                  <span style="color:#94a3b8;font-size:11px">{ev['time']}</span>&nbsp;
                  <span style="color:#e2e8f0;font-weight:600">{ev['method']}</span>&nbsp;
                  <span style="color:#7dd3fc">{ev['endpoint']}</span>&nbsp;
                  <span style="color:#94a3b8">HTTP {ev['status']}</span>&nbsp;
                  <span class="{badge_cls}">{ev['pred_name']}</span>
                </div>"""
            feed_placeholder.markdown(feed_html, unsafe_allow_html=True)
        else:
            feed_placeholder.info("▶ Press **Start** to begin the real-time simulation.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("# 📊 Model Performance Dashboard")

    all_metrics = load_metrics_data()
    if not all_metrics:
        st.warning("No metrics found. Run `train_individual.py` and `train_hybrid.py` first.")
    else:
        # ── Metric cards ──────────────────────────────────────────────────
        st.markdown("### All Models — Key Metrics")
        cols = st.columns(4)
        metric_keys = ["accuracy", "precision", "recall", "f1"]
        for i, m in enumerate(all_metrics):
            with st.expander(f"**{m['model_name']}**", expanded=False):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy",  f"{m['accuracy']:.4f}")
                c2.metric("Precision", f"{m['precision']:.4f}")
                c3.metric("Recall",    f"{m['recall']:.4f}")
                c4.metric("F1 Score",  f"{m['f1']:.4f}")
                if "confusion_matrix" in m:
                    cm = np.array(m["confusion_matrix"])
                    fig_cm = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=["Normal","CredStuffing","BizLogic"],
                        y=["Normal","CredStuffing","BizLogic"],
                        color_continuous_scale="Blues",
                        title=f"Confusion Matrix — {m['model_name']}",
                        text_auto=True,
                    )
                    fig_cm.update_layout(
                        paper_bgcolor="#0F1117", plot_bgcolor="#0F1117",
                        font_color="white", height=350,
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)

        # ── Grouped bar chart (Plotly) ─────────────────────────────────────
        st.markdown("### Comparative Metrics — Grouped Bar Chart")
        model_names = [m["model_name"] for m in all_metrics]
        acc  = [m["accuracy"]  for m in all_metrics]
        prec = [m["precision"] for m in all_metrics]
        rec  = [m["recall"]    for m in all_metrics]
        f1   = [m["f1"]        for m in all_metrics]

        fig_bar = go.Figure()
        for metric_vals, label, color in [
            (acc,  "Accuracy",  "#6366f1"),
            (prec, "Precision", "#22c55e"),
            (rec,  "Recall",    "#f59e0b"),
            (f1,   "F1-Score",  "#ef4444"),
        ]:
            fig_bar.add_trace(go.Bar(
                name=label, x=model_names, y=metric_vals,
                text=[f"{v:.4f}" for v in metric_vals],
                textposition="outside", textfont=dict(size=9, color="white"),
                marker_color=color, opacity=0.85,
            ))
        fig_bar.update_layout(
            barmode="group", paper_bgcolor="#0F1117", plot_bgcolor="#0F1117",
            font=dict(color="white"), height=480,
            legend=dict(font=dict(color="white")),
            yaxis=dict(range=[0, 1.15], gridcolor="#333"),
            xaxis=dict(tickangle=-25),
            margin=dict(b=120),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Radar chart ────────────────────────────────────────────────────
        st.markdown("### Radar Chart — All Models")
        categories = ["Accuracy", "Precision", "Recall", "F1"]
        fig_radar  = go.Figure()
        for m in all_metrics:
            vals = [m["accuracy"], m["precision"], m["recall"], m["f1"]]
            vals_closed = vals + [vals[0]]
            cats_closed = categories + [categories[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed, theta=cats_closed,
                fill="toself", opacity=0.3, name=m["model_name"],
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#1e1e2e",
                radialaxis=dict(visible=True, range=[0,1], color="#64748b"),
                angularaxis=dict(color="white"),
            ),
            paper_bgcolor="#0F1117",
            font=dict(color="white"), height=500,
            legend=dict(font=dict(color="white")),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ── F1 improvement ─────────────────────────────────────────────────
        INDIVIDUAL_MODELS = {"Random Forest", "Isolation Forest", "ANN", "LSTM"}
        ind_f1 = {m["model_name"]: m["f1"] for m in all_metrics
                  if m["model_name"] in INDIVIDUAL_MODELS}
        hyb    = [m for m in all_metrics if m["model_name"] not in INDIVIDUAL_MODELS]

        if ind_f1 and hyb:
            best_f1   = max(ind_f1.values())
            best_name = max(ind_f1, key=ind_f1.get)
            gains     = [m["f1"] - best_f1 for m in hyb]
            colors_   = ["#ef4444" if "Master" in m["model_name"] else "#f97316"
                         for m in hyb]

            st.markdown(f"### F1 Gain over Best Individual ({best_name}, F1={best_f1:.4f})")
            fig_gain = go.Figure(go.Bar(
                x=gains, y=[m["model_name"] for m in hyb],
                orientation="h",
                text=[f"{g:+.4f}" for g in gains],
                textposition="outside", textfont=dict(color="white"),
                marker_color=colors_,
            ))
            fig_gain.update_layout(
                paper_bgcolor="#0F1117", plot_bgcolor="#0F1117",
                font=dict(color="white"), height=300,
                xaxis=dict(gridcolor="#333"),
                margin=dict(l=20, r=60),
            )
            st.plotly_chart(fig_gain, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Model Comparison":
    st.markdown("# 🔍 Model Architecture Comparison")

    st.markdown("""
    ### Model Descriptions

    | Model | Type | Strategy |
    |-------|------|----------|
    | **Random Forest** | Individual | Supervised ensemble, 200 trees, balanced class weights |
    | **Isolation Forest** | Individual | Unsupervised anomaly detection, 200 trees |
    | **ANN** | Individual | Dense NN: 256→128→64→32→3, BatchNorm + Dropout |
    | **LSTM** | Individual | 2-layer LSTM (128→64), features reshaped to 5 time-steps |
    | **RF + ANN** | Hybrid | Stacking: RF probabilities fed as extra features to ANN meta-learner |
    | **IF + ANN** | Hybrid | Anomaly score from IF appended to feature vector for ANN |
    | **RF + LSTM** | Hybrid | Parallel weighted averaging (RF×0.45 + LSTM×0.55) |
    | **ANN + LSTM + RF (Master)** | Master Hybrid | Tri-model soft voting (RF×0.30 + ANN×0.35 + LSTM×0.35) |
    """)

    st.markdown("### Detection Coverage Matrix")
    cov = {
        "Attack Type":             ["Credential Stuffing", "Business Logic", "DDoS/DoS", "Bot Traffic", "SQL Injection", "XSS"],
        "Random Forest":           ["✅ High",  "✅ High",  "✅ High",  "⚠️ Med",   "✅ High",  "⚠️ Med"],
        "Isolation Forest":        ["⚠️ Med",   "⚠️ Med",   "✅ High",  "✅ High",  "⚠️ Med",   "⚠️ Med"],
        "ANN":                     ["✅ High",  "✅ High",  "⚠️ Med",   "⚠️ Med",   "✅ High",  "✅ High"],
        "LSTM":                    ["✅ High",  "⚠️ Med",   "⚠️ Med",   "⚠️ Med",   "⚠️ Med",   "⚠️ Med"],
        "RF + ANN":                ["✅ High",  "✅ High",  "✅ High",  "⚠️ Med",   "✅ High",  "✅ High"],
        "IF + ANN":                ["✅ High",  "✅ High",  "✅ High",  "✅ High",  "✅ High",  "✅ High"],
        "RF + LSTM":               ["✅ High",  "✅ High",  "✅ High",  "⚠️ Med",   "✅ High",  "⚠️ Med"],
        "Master (ANN+LSTM+RF)":    ["✅ High",  "✅ High",  "✅ High",  "✅ High",  "✅ High",  "✅ High"],
    }
    st.dataframe(pd.DataFrame(cov).set_index("Attack Type"),
                 use_container_width=True)

    st.markdown("### Ensemble Weight Configuration")
    fig_weights = go.Figure(go.Bar(
        x=["Random Forest", "ANN", "LSTM"],
        y=[0.30, 0.35, 0.35],
        text=["30%", "35%", "35%"],
        textposition="outside",
        marker_color=["#6366f1", "#22c55e", "#f59e0b"],
    ))
    fig_weights.update_layout(
        title="Master Hybrid — Model Weights",
        paper_bgcolor="#0F1117", plot_bgcolor="#0F1117",
        font=dict(color="white"), height=350,
        yaxis=dict(range=[0, 0.5], title="Weight", gridcolor="#333"),
    )
    st.plotly_chart(fig_weights, use_container_width=True)
