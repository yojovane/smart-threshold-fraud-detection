"""
FraudSense design system — matches fraud_app_prototype.html.
CSS variables, noise texture, sidebar, topbar, metrics, dial, reason-list, panels, feed.
"""
COLORS = {
    "bg": "#0a0c10",
    "surface": "#111318",
    "surface2": "#181c24",
    "border": "#1e2330",
    "accent": "#e63946",
    "accent2": "#f4a261",
    "green": "#2dc653",
    "blue": "#4895ef",
    "muted": "#4a5568",
    "text": "#e2e8f0",
    "text2": "#8892a4",
}
OPTIMAL_THRESHOLD = 0.43
COST_FN_USD = 180
COST_FP_USD = 12


def get_css() -> str:
    return """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&family=Inter:wght@300;400;500&display=swap');

:root {
  --bg: #0a0c10;
  --surface: #111318;
  --surface2: #181c24;
  --border: #1e2330;
  --accent: #e63946;
  --accent2: #f4a261;
  --green: #2dc653;
  --blue: #4895ef;
  --muted: #4a5568;
  --text: #e2e8f0;
  --text2: #8892a4;
  --font-head: 'Syne', sans-serif;
  --font-mono: 'IBM Plex Mono', monospace;
  --font-body: 'Inter', sans-serif;
}

.stApp { background: var(--bg) !important; }
.stApp::before {
  content: '';
  position: fixed; inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E");
  pointer-events: none; z-index: 0;
}

section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .stMarkdown { color: var(--text2); }

/* Sidebar logo */
.sidebar-logo-mark { font-family: var(--font-head); font-weight: 800; font-size: 18px; color: var(--accent); letter-spacing: -0.5px; }
.sidebar-logo-sub { font-family: var(--font-mono); font-size: 9px; color: var(--muted); letter-spacing: 2px; text-transform: uppercase; margin-top: 2px; }

/* Topbar */
.fraud-topbar { height: 56px; background: var(--surface); border-bottom: 1px solid var(--border); display: flex; align-items: center; padding: 0 28px; gap: 16px; margin: -1rem -1rem 1.5rem -1rem; }
.fraud-page-title { font-family: var(--font-head); font-weight: 700; font-size: 15px; color: var(--text); }
.fraud-tag { display: inline-flex; align-items: center; gap: 4px; font-family: var(--font-mono); font-size: 9px; padding: 3px 8px; border-radius: 4px; text-transform: uppercase; letter-spacing: 1px; margin-right: 6px; }
.fraud-tag.safe { background: rgba(45,198,83,0.10); color: var(--green); border: 1px solid rgba(45,198,83,0.2); }
.fraud-tag.info { background: rgba(72,149,239,0.10); color: var(--blue); border: 1px solid rgba(72,149,239,0.2); }
.fraud-tag.warning { background: rgba(244,162,97,0.10); color: var(--accent2); border: 1px solid rgba(244,162,97,0.2); }
.fraud-tag.danger { background: rgba(230,57,70,0.12); color: var(--accent); border: 1px solid rgba(230,57,70,0.2); }

/* Metrics row */
.fraud-metrics-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }
.fraud-metric-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; position: relative; overflow: hidden;
}
.fraud-metric-card::after { content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 2px; }
.fraud-metric-card.red::after { background: var(--accent); }
.fraud-metric-card.orange::after { background: var(--accent2); }
.fraud-metric-card.green::after { background: var(--green); }
.fraud-metric-card.blue::after { background: var(--blue); }
.fraud-metric-label { font-family: var(--font-mono); font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }
.fraud-metric-value { font-family: var(--font-head); font-weight: 800; font-size: 28px; color: var(--text); line-height: 1; margin-bottom: 4px; }
.fraud-metric-delta { font-family: var(--font-mono); font-size: 10px; color: var(--text2); }

/* Input section */
.fraud-input-section { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 24px; margin-bottom: 24px; }
.fraud-input-title { font-family: var(--font-head); font-weight: 700; font-size: 14px; color: var(--text); }
.fraud-input-sub { font-family: var(--font-mono); font-size: 10px; color: var(--muted); margin-top: 3px; }
.fraud-analyze-btn { background: var(--accent); color: white; border: none; border-radius: 10px; padding: 12px 28px; font-family: var(--font-head); font-weight: 700; font-size: 13px; cursor: pointer; letter-spacing: 0.3px; box-shadow: 0 4px 20px rgba(230,57,70,0.25); }
.stButton > button[kind="primary"], .stButton > button:first-child { background: var(--accent) !important; color: white !important; border: none !important; font-family: var(--font-head) !important; font-weight: 700 !important; }
.stButton > button:hover { filter: brightness(1.1); transform: translateY(-1px); }

/* Hero: score card + explain */
.fraud-hero { display: grid; grid-template-columns: 1fr 2fr; gap: 20px; margin-bottom: 24px; }
@media (max-width: 900px) { .fraud-hero { grid-template-columns: 1fr; } }
.fraud-score-card {
  background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 28px 24px;
  display: flex; flex-direction: column; align-items: center; justify-content: center; position: relative; overflow: hidden;
}
.fraud-score-card::before {
  content: ''; position: absolute; inset: -1px; border-radius: 16px; padding: 1px;
  background: linear-gradient(135deg, var(--accent), transparent 50%);
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0); -webkit-mask-composite: xor;
  mask-composite: exclude; opacity: 0.6;
}
.fraud-score-label { font-family: var(--font-mono); font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 12px; }
.fraud-dial-wrap { position: relative; width: 140px; height: 140px; margin-bottom: 12px; }
.fraud-dial-center { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; }
.fraud-dial-number { font-family: var(--font-head); font-weight: 800; font-size: 36px; color: var(--accent); line-height: 1; }
.fraud-dial-unit { font-family: var(--font-mono); font-size: 10px; color: var(--muted); }
.fraud-verdict { font-family: var(--font-head); font-weight: 700; font-size: 14px; color: var(--accent); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 4px; }
.fraud-verdict.approved { color: var(--green); }
.fraud-verdict-sub { font-size: 11px; color: var(--text2); text-align: center; font-family: var(--font-mono); }

/* Explain card */
.fraud-explain-card { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 24px; }
.fraud-explain-title { font-family: var(--font-head); font-weight: 700; font-size: 14px; color: var(--text); margin-bottom: 4px; }
.fraud-explain-sub { font-size: 11px; color: var(--text2); font-family: var(--font-mono); margin-bottom: 20px; }
.fraud-reason-list { display: flex; flex-direction: column; gap: 10px; }
.fraud-reason-item {
  display: flex; align-items: flex-start; gap: 12px; padding: 12px 14px;
  background: var(--surface2); border-radius: 10px; border: 1px solid var(--border);
}
.fraud-reason-rank { font-family: var(--font-mono); font-size: 10px; color: var(--muted); min-width: 24px; padding-top: 2px; }
.fraud-reason-body { flex: 1; }
.fraud-reason-text { font-size: 12px; color: var(--text); line-height: 1.5; margin-bottom: 6px; }
.fraud-reason-bar-wrap { display: flex; align-items: center; gap: 8px; }
.fraud-reason-bar-bg { flex: 1; height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; }
.fraud-reason-bar-fill { height: 100%; border-radius: 2px; transition: width 0.5s ease; }
.fraud-reason-bar-fill.high { background: var(--accent); }
.fraud-reason-bar-fill.med { background: var(--accent2); }
.fraud-reason-bar-fill.low { background: var(--blue); }
.fraud-reason-val { font-family: var(--font-mono); font-size: 10px; color: var(--text2); min-width: 36px; text-align: right; }

/* Bottom grid */
.fraud-bottom-grid { display: grid; grid-template-columns: 1.5fr 1fr; gap: 20px; margin-bottom: 24px; }
@media (max-width: 900px) { .fraud-bottom-grid { grid-template-columns: 1fr; } }
.fraud-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 22px; }
.fraud-panel-title { font-family: var(--font-head); font-weight: 700; font-size: 13px; color: var(--text); }
.fraud-panel-tag { font-family: var(--font-mono); font-size: 9px; color: var(--muted); border: 1px solid var(--border); padding: 3px 8px; border-radius: 4px; text-transform: uppercase; letter-spacing: 1px; }
.fraud-roi-box {
  background: linear-gradient(135deg, rgba(230,57,70,0.08), rgba(244,162,97,0.06));
  border: 1px solid rgba(230,57,70,0.2); border-radius: 12px; padding: 18px; margin-bottom: 12px;
}
.fraud-roi-title { font-family: var(--font-mono); font-size: 9px; color: var(--muted); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; }
.fraud-roi-number { font-family: var(--font-head); font-weight: 800; font-size: 32px; color: var(--accent2); line-height: 1; margin-bottom: 4px; }
.fraud-roi-desc { font-size: 11px; color: var(--text2); line-height: 1.5; }
.fraud-threshold-row { display: flex; align-items: center; gap: 12px; padding: 12px 0; border-top: 1px solid var(--border); }
.fraud-threshold-label { font-family: var(--font-mono); font-size: 10px; color: var(--muted); flex: 1; }
.fraud-threshold-val { font-family: var(--font-head); font-weight: 700; font-size: 18px; color: var(--text); }
.fraud-feed-list { display: flex; flex-direction: column; gap: 8px; }
.fraud-feed-item { display: flex; align-items: center; gap: 12px; padding: 10px 12px; background: var(--surface2); border-radius: 8px; border: 1px solid var(--border); font-size: 11px; }
.fraud-feed-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.fraud-feed-dot.fraud { background: var(--accent); box-shadow: 0 0 6px var(--accent); }
.fraud-feed-dot.legit { background: var(--green); }
.fraud-feed-id { font-family: var(--font-mono); font-size: 10px; color: var(--muted); min-width: 70px; }
.fraud-feed-desc { flex: 1; color: var(--text2); }
.fraud-feed-score { font-family: var(--font-head); font-weight: 700; font-size: 13px; }
.fraud-feed-score.fraud { color: var(--accent); }
.fraud-feed-score.legit { color: var(--green); }
.fraud-glow-line { height: 1px; background: linear-gradient(90deg, transparent, var(--accent), transparent); margin: 24px 0; opacity: 0.3; }

/* Author card in sidebar */
.fraud-author-card { display: flex; align-items: center; gap: 10px; margin-top: auto; padding-top: 16px; border-top: 1px solid var(--border); }
.fraud-author-avatar { width: 32px; height: 32px; border-radius: 50%; background: linear-gradient(135deg, var(--accent), var(--accent2)); display: flex; align-items: center; justify-content: center; font-family: var(--font-head); font-weight: 800; font-size: 12px; color: white; flex-shrink: 0; }
.fraud-author-name { font-size: 12px; font-weight: 500; color: var(--text); }
.fraud-author-role { font-size: 10px; color: var(--muted); font-family: var(--font-mono); }

/* Hide default Streamlit padding where we use custom blocks */
.block-container { padding-top: 1rem !important; max-width: 1400px !important; }
h1, h2, h3 { font-family: var(--font-head) !important; color: var(--text) !important; }
[data-testid="stMetricValue"] { font-family: var(--font-head) !important; font-weight: 800 !important; color: var(--text) !important; }
[data-testid="stMetricLabel"] { color: var(--text2) !important; font-family: var(--font-mono) !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
"""


def inject_fraud_sense_style(st) -> None:
    st.markdown(get_css(), unsafe_allow_html=True)
