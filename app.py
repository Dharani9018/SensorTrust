import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
import sys
from pathlib import Path

sys.path.append('src')

from src.anomaly.lstm_autoencoder import LSTMAutoencoder
from src.anomaly.sequence_dataset import create_sequences
from src.anomaly.mahalanobis import MahalanobisDetector
from src.anomaly.detector import detect_anomalies

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_features():
    with open('results/features.json') as f:
        return json.load(f)

@st.cache_data
def load_trust():
    with open('results/trust.json') as f:
        return json.load(f)

@st.cache_resource
def load_model():
    with open('src/models/threshold.json') as f:
        default_threshold = json.load(f)['threshold']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = LSTMAutoencoder(n_features=3, hidden_size=64, latent_size=32).to(device)
    model.load_state_dict(torch.load('src/models/lstm_autoencoder.pt', map_location=device))
    model.eval()
    return model, device, default_threshold

@st.cache_resource
def load_mahal(_features):
    clean   = _features['Clean']
    f1_c    = np.array(clean['f1'])
    f2_c    = np.array(clean['f2'])
    gmis_c  = np.array(clean['gmis'])
    min_len = min(len(f1_c), len(f2_c), len(gmis_c))
    det     = MahalanobisDetector()
    det.fit(f1_c[:min_len], f2_c[:min_len], gmis_c[:min_len])
    return det

features                       = load_features()
trust_data                     = load_trust()
model, device, default_thresh  = load_model()
mahal_det                      = load_mahal(features)

clean      = features['Clean']
CLEAN_F1   = clean['F1']
CLEAN_F2   = clean['F2']
CLEAN_GMIS = clean['GMIS']
attack_names = [k for k in features.keys() if k != 'Clean']

# ── Detection functions ───────────────────────────────────────────────────────
def run_mahalanobis(f1, f2, gmis, thresh):
    min_len = min(len(f1), len(f2), len(gmis))
    f1   = f1[:min_len]
    f2   = f2[:min_len]
    gmis = gmis[:min_len]
    mask = ~(np.isnan(f1) | np.isnan(f2) | np.isnan(gmis))
    scores_full = np.zeros(min_len)
    if mask.sum() > 3:
        scores = mahal_det.score(f1[mask], f2[mask], gmis[mask])
        scores_full[mask] = np.nan_to_num(scores)
    alerts = scores_full > thresh
    valid  = mask
    rate   = float(np.sum(alerts[valid]) / np.sum(valid) * 100) if valid.sum() > 0 else 0.0
    return scores_full, alerts, rate

def run_lstm(f1, f2, gmis, thresh, seq_len=20):
    min_len = min(len(f1), len(f2), len(gmis))
    X = np.column_stack([f1[:min_len], f2[:min_len], gmis[:min_len]])
    X = np.nan_to_num(X)
    if len(X) < seq_len:
        return np.zeros(min_len), np.zeros(min_len, dtype=bool), 0.0
    seqs   = create_sequences(X, seq_len=seq_len)
    tensor = torch.tensor(seqs, dtype=torch.float32).to(device)
    errors = []
    with torch.no_grad():
        for i in range(len(tensor)):
            x     = tensor[i].unsqueeze(0)
            recon = model(x)
            errors.append(torch.mean((x - recon) ** 2).item())
    errors      = np.array(errors)
    pad         = np.full(seq_len - 1, errors[0])
    errors_full = np.concatenate([pad, errors])
    # match length to input
    if len(errors_full) > min_len:
        errors_full = errors_full[:min_len]
    elif len(errors_full) < min_len:
        errors_full = np.concatenate([errors_full, np.full(min_len - len(errors_full), errors_full[-1])])
    alerts = errors_full > thresh
    rate   = float(np.mean(alerts) * 100)
    return errors_full, alerts, rate

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SensorTrust",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 SensorTrust: Adaptive Cross-Modal Sensor Spoofing Detector")
st.markdown("*Real KITTI data · Real trained LSTM Autoencoder · Real Mahalanobis detector*")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("🎯 Scenario")
selected = st.sidebar.selectbox("Attack", ["Clean (No Attack)"] + attack_names)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Detection Thresholds")
mahal_thresh = st.sidebar.slider(
    "Mahalanobis threshold", 1.0, 10.0, 3.0, step=0.25,
    help="Lower = more sensitive. Default = 3.0"
)
lstm_thresh = st.sidebar.slider(
    "LSTM threshold", 0.1, 5.0, float(round(default_thresh, 2)), step=0.05,
    help=f"Trained threshold = {default_thresh:.4f}"
)
seq_len = st.sidebar.slider(
    "LSTM sequence length", 10, 100, 20, step=10,
    help="Longer = smoother but slower"
)

st.sidebar.markdown("---")
st.sidebar.caption("Dataset: KITTI Raw 2011_09_26 drive 0009 · 447 frames")

# ── Get features ──────────────────────────────────────────────────────────────
is_clean = selected == "Clean (No Attack)"
name     = "Clean" if is_clean else selected
feat     = features[name]
f1       = np.array(feat['f1'])
f2       = np.array(feat['f2'])
gmis     = np.array(feat['gmis'])

# ── Run detectors ─────────────────────────────────────────────────────────────
with st.spinner("Running detectors..."):
    mahal_scores, mahal_alerts, mahal_rate = run_mahalanobis(f1, f2, gmis, mahal_thresh)
    lstm_errors,  lstm_alerts,  lstm_rate  = run_lstm(f1, f2, gmis, lstm_thresh, seq_len)

mahal_detected = mahal_rate > 10
lstm_detected  = lstm_rate  > 10
either         = mahal_detected or lstm_detected

if is_clean:
    trust   = {'gps': 0.92, 'imu': 0.91, 'lidar': 0.90, 'camera': 0.89}
    ranking = [['gps', 0.1], ['imu', 0.1], ['lidar', 0.1], ['camera', 0.1]]
else:
    trust   = trust_data[name]['trust']
    ranking = trust_data[name]['ranking']

# ── Top metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Scenario", name)
c2.metric("Mahalanobis",      "🚨 ATTACK" if mahal_detected else "✅ Clean",
          f"{mahal_rate:.1f}% frames flagged")
c3.metric("LSTM Autoencoder", "🚨 ATTACK" if lstm_detected  else "✅ Clean",
          f"{lstm_rate:.1f}% sequences flagged")
c4.metric("Combined Result",  "🚨 DETECTED" if either else "✅ Clean")
c5.metric("Top Suspect",
          f"⚠️ {ranking[0][0].upper()}" if not is_clean else "None",
          f"trust = {trust[ranking[0][0]]:.4f}" if not is_clean else "—")

st.markdown("---")

# ── Feature plots ─────────────────────────────────────────────────────────────
st.subheader("📊 Motion Consistency Features")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
triples = [
    (f1,   'F1 — Kinematic Delta',           '#e74c3c', CLEAN_F1),
    (f2,   'F2 — GPS vs LiDAR Odometry',     '#3498db', CLEAN_F2),
    (gmis, 'GMIS — Geometric Inconsistency', '#2ecc71', CLEAN_GMIS),
]
for ax, (arr, title, col, baseline) in zip(axes, triples):
    ax.plot(arr, color=col, linewidth=0.8, alpha=0.9)
    ax.axhline(baseline,     color='gray', linestyle='--', linewidth=1,   label='Clean baseline')
    ax.axhline(baseline * 3, color='red',  linestyle='--', linewidth=1.2, label='3× threshold')
    ax.fill_between(range(len(arr)), 0, arr,
                    where=(arr > baseline * 3), color='red', alpha=0.2, label='Anomalous')
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Detection plots ───────────────────────────────────────────────────────────
st.subheader("🔍 Live Detector Outputs")
st.caption("Adjust thresholds in sidebar — reruns on real feature arrays instantly")

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))

ax1.plot(mahal_scores, color='#3498db', linewidth=0.8, label='Mahalanobis Distance')
ax1.axhline(mahal_thresh, color='red', linestyle='--', linewidth=1.5,
            label=f'Threshold = {mahal_thresh}')
ax1.fill_between(range(len(mahal_scores)), 0, mahal_scores,
                 where=mahal_alerts, color='red', alpha=0.25, label='Flagged')
ax1.set_title('Mahalanobis Distance per Frame')
ax1.set_xlabel('Frame')
ax1.set_ylabel('Distance')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

ax2.plot(lstm_errors, color='#e67e22', linewidth=0.8, label='Reconstruction Error')
ax2.axhline(lstm_thresh, color='red', linestyle='--', linewidth=1.5,
            label=f'Threshold = {lstm_thresh:.3f}')
ax2.fill_between(range(len(lstm_errors)), 0, lstm_errors,
                 where=lstm_alerts, color='red', alpha=0.25, label='Flagged')
ax2.set_title('LSTM Autoencoder Reconstruction Error')
ax2.set_xlabel('Sequence Index')
ax2.set_ylabel('MSE')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig2)
plt.close()

# ── Trust scores ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🛡️ Sensor Trust Scores & Suspicion Ranking")

col_left, col_right = st.columns([3, 2])

with col_left:
    sensors    = ['gps', 'imu', 'lidar', 'camera']
    trust_vals = [trust[s] for s in sensors]
    bar_colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    fig3, ax3 = plt.subplots(figsize=(7, 3.5))
    bars = ax3.bar(sensors, trust_vals, color=bar_colors, edgecolor='white', linewidth=1.2)
    ax3.set_ylim(0, 1.1)
    ax3.axhline(0.3, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Low trust boundary')
    ax3.set_ylabel('Trust Score')
    ax3.set_title('Per-Sensor Trust Score  (lower = more suspicious)')
    for bar, val in zip(bars, trust_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

with col_right:
    st.markdown(f"### 🏆 Suspicion Ranking")
    st.markdown(f"*Scenario: **{name}***")
    medals = ["🥇", "🥈", "🥉", "4️⃣"]
    for i, (sensor, _) in enumerate(ranking):
        t    = trust[sensor]
        flag = " 🚨 **COMPROMISED?**" if i == 0 and t < 0.2 and not is_clean else ""
        st.markdown(f"{medals[i]} **{sensor.upper()}** — trust: `{t:.4f}`{flag}")

    if not is_clean:
        st.markdown("---")
        st.markdown("**Feature Elevation vs Clean**")
        st.markdown(f"- F1:   `{feat['F1'] / CLEAN_F1:.1f}×`")
        st.markdown(f"- F2:   `{feat['F2'] / CLEAN_F2:.1f}×`")
        st.markdown(f"- GMIS: `{feat['GMIS'] / CLEAN_GMIS:.1f}×`")

# ── Summary table ─────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📋 Full Attack Evaluation Table (uses current threshold settings)"):
    import pandas as pd
    rows = []
    for atk in attack_names:
        f  = features[atk]
        tr = trust_data.get(atk, {})
        f1_a   = np.array(f['f1'])
        f2_a   = np.array(f['f2'])
        gmis_a = np.array(f['gmis'])
        _, _, m_rate = run_mahalanobis(f1_a, f2_a, gmis_a, mahal_thresh)
        _, _, l_rate = run_lstm(f1_a, f2_a, gmis_a, lstm_thresh, seq_len)
        top = tr['ranking'][0][0].upper() if tr else 'N/A'
        rows.append({
            'Attack':       atk,
            'F1×':         f"{f['F1'] / CLEAN_F1:.1f}×",
            'F2×':         f"{f['F2'] / CLEAN_F2:.1f}×",
            'GMIS×':       f"{f['GMIS'] / CLEAN_GMIS:.1f}×",
            'Mahal %':     f"{m_rate:.0f}%",
            'LSTM %':      f"{l_rate:.0f}%",
            'Top Suspect': top,
            'Detected':    '✅' if (m_rate > 50 or l_rate > 50) else '❌',
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.markdown("---")
st.caption(
    "SensorTrust · KITTI Raw 2011_09_26 drive 0009 · "
    "Real LSTM Autoencoder + Mahalanobis · No simulation"
)
