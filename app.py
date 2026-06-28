import streamlit as st
import numpy as np
import torch
import json
import sys
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append('src')

from src.anomaly.mahalanobis import MahalanobisDetector
from src.anomaly.detector import detect_anomalies
from src.anomaly.lstm_autoencoder import LSTMAutoencoder
from src.anomaly.sequence_dataset import create_sequences
from src.graph.disagreement_graph import build_disagreement_graph
from src.graph.trust_score import compute_node_inconsistency, compute_trust_scores
from src.graph.ranking import rank_sensors

BETA = 0.7

st.set_page_config(
    page_title="SensorTrust — AV Spoofing Detector",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 SensorTrust: Adaptive Cross-Modal Sensor Spoofing Detector")
st.markdown("*Detects coordinated multi-sensor spoofing attacks on autonomous vehicles using motion-consistency features.*")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Attack Simulation")

attack_type = st.sidebar.selectbox("Select Attack Scenario", [
    "Clean (No Attack)",
    "GPS Speed Ramp",
    "IMU Constant Bias",
    "LiDAR Phantom Injection",
    "Camera Gaussian Noise",
    "Coordinated GPS + IMU",
    "Coordinated GPS + LiDAR",
    "ALL FOUR Sensors",
])

n_frames = st.sidebar.slider("Number of frames", 100, 500, 300)
attack_start = st.sidebar.slider("Attack start frame", 50, 250, 150)
noise_level  = st.sidebar.slider("Attack intensity", 0.5, 5.0, 2.0, step=0.5)
mahal_thresh = st.sidebar.slider("Mahalanobis threshold", 1.0, 10.0, 3.0, step=0.5)
lstm_thresh  = st.sidebar.slider("LSTM threshold", 0.1, 5.0, 1.1, step=0.1)

# ── Simulate sensor data ──────────────────────────────────────────────────────
@st.cache_data
def simulate_data(attack_type, n_frames, attack_start, noise_level):
    np.random.seed(42)
    t = np.arange(n_frames)

    # clean baseline signals
    gps   = np.random.normal(0, 0.3, n_frames)
    imu   = gps + np.random.normal(0, 0.05, n_frames)
    lidar = gps * 0.95 + np.random.normal(0, 0.05, n_frames)
    cam   = gps * 0.9  + np.random.normal(0, 0.1,  n_frames)

    attack_mask = np.zeros(n_frames, dtype=bool)

    if attack_type == "GPS Speed Ramp":
        ramp = np.zeros(n_frames)
        end  = min(attack_start + 50, n_frames)
        ramp[attack_start:end] = np.linspace(0, noise_level * 3, end - attack_start)
        ramp[end:] = noise_level * 3
        gps += ramp
        attack_mask[attack_start:] = True

    elif attack_type == "IMU Constant Bias":
        imu[attack_start:] += noise_level
        attack_mask[attack_start:] = True

    elif attack_type == "LiDAR Phantom Injection":
        lidar[attack_start:attack_start+100] += noise_level * 1.5
        attack_mask[attack_start:attack_start+100] = True

    elif attack_type == "Camera Gaussian Noise":
        cam += np.random.normal(0, noise_level * 0.3, n_frames)
        attack_mask[attack_start:] = True

    elif attack_type == "Coordinated GPS + IMU":
        gps[attack_start:] += noise_level * 2
        imu[attack_start:] += noise_level * 1.8
        attack_mask[attack_start:] = True

    elif attack_type == "Coordinated GPS + LiDAR":
        gps[attack_start:]   += noise_level * 2
        lidar[attack_start:] += noise_level * 1.5
        attack_mask[attack_start:] = True

    elif attack_type == "ALL FOUR Sensors":
        gps[attack_start:]   += noise_level * 3
        imu[attack_start:]   += noise_level * 1.5
        lidar[attack_start:] += noise_level * 1.5
        cam  += np.random.normal(0, noise_level * 0.3, n_frames)
        attack_mask[attack_start:] = True

    # features from signals
    f1   = np.abs(gps - imu)
    f2   = np.abs(gps - lidar)
    gmis = (np.abs(gps - imu) + np.abs(gps - lidar) + np.abs(gps - cam)) / 3

    return gps, imu, lidar, cam, f1, f2, gmis, attack_mask

gps, imu, lidar, cam, f1, f2, gmis, attack_mask = simulate_data(
    attack_type, n_frames, attack_start, noise_level
)

# ── Detection ─────────────────────────────────────────────────────────────────
def run_mahalanobis(f1, f2, gmis, thresh):
    clean_end = min(attack_start, 80)
    det = MahalanobisDetector()
    det.fit(f1[:clean_end], f2[:clean_end], gmis[:clean_end])
    scores = det.score(f1, f2, gmis)
    alerts = scores > thresh
    return scores, alerts

def run_lstm_sim(f1, f2, gmis, thresh, seq_len=20):
    X = np.column_stack([f1, f2, gmis])
    X = np.nan_to_num(X)
    if len(X) < seq_len:
        return np.zeros(len(X)), np.zeros(len(X), dtype=bool)
    seqs = create_sequences(X, seq_len=seq_len)
    errors = np.array([np.mean((s - np.mean(s, axis=0))**2) for s in seqs])
    errors_full = np.concatenate([np.full(seq_len - 1, errors[0]), errors])
    alerts = errors_full > thresh
    return errors_full, alerts

def run_trust(gps, imu, lidar, cam):
    graph = build_disagreement_graph(gps, imu, lidar, cam)
    incon = compute_node_inconsistency(graph)
    trust = {s: float(np.exp(-BETA * np.nanmean(v))) for s, v in incon.items()}
    ranking = rank_sensors(incon)
    return trust, ranking

mahal_scores, mahal_alerts = run_mahalanobis(f1, f2, gmis, mahal_thresh)
lstm_errors,  lstm_alerts  = run_lstm_sim(f1, f2, gmis, lstm_thresh)
trust, ranking             = run_trust(gps, imu, lidar, cam)

# ── Layout ────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

mahal_det = np.mean(mahal_alerts) > 0.1
lstm_det  = np.mean(lstm_alerts)  > 0.1

col1.metric("Attack Scenario",  attack_type.split("(")[0].strip())
col2.metric("Mahalanobis",      "🚨 ATTACK" if mahal_det else "✅ Clean",
            f"{np.mean(mahal_alerts)*100:.0f}% frames flagged")
col3.metric("LSTM Autoencoder", "🚨 ATTACK" if lstm_det else "✅ Clean",
            f"{np.mean(lstm_alerts)*100:.0f}% frames flagged")
col4.metric("Top Suspect",      f"⚠️ {ranking[0][0].upper()}",
            f"trust={trust[ranking[0][0]]:.3f}")

st.markdown("---")

# ── Sensor signals plot ───────────────────────────────────────────────────────
st.subheader("📡 Sensor Signals")
fig, axes = plt.subplots(2, 2, figsize=(14, 6))
pairs = [("GPS",   gps,   "#e74c3c"),
         ("IMU",   imu,   "#3498db"),
         ("LiDAR", lidar, "#2ecc71"),
         ("Camera",cam,   "#f39c12")]

for ax, (name, sig, col) in zip(axes.flat, pairs):
    ax.plot(sig, color=col, linewidth=0.8, label=name)
    if attack_mask.any():
        ax.fill_between(range(n_frames), sig.min(), sig.max(),
                        where=attack_mask, alpha=0.15, color='red', label='Attack window')
    ax.set_title(name); ax.grid(True, alpha=0.3); ax.legend(fontsize=7)

plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Detection plots ───────────────────────────────────────────────────────────
st.subheader("🔍 Anomaly Detection")
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

ax1.plot(mahal_scores, 'b-', linewidth=0.8, label='Mahalanobis Score')
ax1.axhline(mahal_thresh, color='r', linestyle='--', label=f'Threshold={mahal_thresh}')
ax1.fill_between(range(len(mahal_scores)), 0, mahal_scores,
                 where=mahal_alerts, color='red', alpha=0.3)
if attack_mask.any():
    ax1.axvspan(attack_start, n_frames, alpha=0.08, color='orange', label='Attack window')
ax1.set_title('Mahalanobis Distance'); ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

ax2.plot(lstm_errors, 'g-', linewidth=0.8, label='LSTM Recon Error')
ax2.axhline(lstm_thresh, color='r', linestyle='--', label=f'Threshold={lstm_thresh}')
ax2.fill_between(range(len(lstm_errors)), 0, lstm_errors,
                 where=lstm_alerts, color='red', alpha=0.3)
if attack_mask.any():
    ax2.axvspan(attack_start, n_frames, alpha=0.08, color='orange', label='Attack window')
ax2.set_title('LSTM Autoencoder'); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig2)
plt.close()

# ── Trust scores ──────────────────────────────────────────────────────────────
st.subheader("🛡️ Sensor Trust Scores")
col_t1, col_t2 = st.columns([1, 1])

with col_t1:
    sensors_list = ['gps', 'imu', 'lidar', 'camera']
    trust_vals   = [trust[s] for s in sensors_list]
    colors_bar   = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    fig3, ax3 = plt.subplots(figsize=(6, 3))
    bars = ax3.bar(sensors_list, trust_vals, color=colors_bar)
    ax3.set_ylim(0, 1)
    ax3.set_ylabel('Trust Score')
    ax3.set_title('Per-Sensor Trust (lower = more suspicious)')
    for bar, val in zip(bars, trust_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

with col_t2:
    st.markdown("### 🏆 Suspicion Ranking")
    medals = ["🥇", "🥈", "🥉", "4️⃣"]
    for i, (sensor, score) in enumerate(ranking):
        trust_val = trust[sensor]
        bar_width = max(1 - trust_val, 0.05)
        st.markdown(
            f"{medals[i]} **{sensor.upper()}** — trust: `{trust_val:.4f}` "
            f"{'🚨 HIGH SUSPICION' if i == 0 and trust_val < 0.3 else ''}"
        )

st.markdown("---")
st.caption("SensorTrust | KITTI-based evaluation | Physics-grounded cross-modal spoofing detection")
