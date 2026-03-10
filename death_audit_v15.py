import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import scipy.signal as sig

EPS = 1e-12


@dataclass
class SensorConfig:
    sr: int = 1000

    # scan
    window_ms: int = 500
    coarse_hop_ms: int = 100
    fine_hop_ms: int = 25
    merge_gap_sec: float = 0.5
    zoom_margin_sec: float = 1.0

    # base gates
    min_snr_db: float = 3.0
    min_coherence: float = 0.6
    min_irreversibility: float = 0.55

    # strict HF candidate gate
    hf_candidate_min_hit_w: int = 3
    hf_candidate_min_snr: float = 4.5
    hf_candidate_min_irr: float = 0.70
    hf_candidate_max_entropy: float = 0.75
    hf_candidate_max_50hz: float = 0.12
    hf_candidate_max_edge: float = 0.85
    hf_candidate_max_mirror: float = 0.90

    # DSP
    notch_q: float = 30.0
    max_lag_ms: float = 5.0
    hf_low_hz: float = 0.5
    hf_high_hz: float = 120.0

    use_common_mode_removal: bool = False
    use_channel_alignment: bool = False

    # ranking
    n_zoom_candidates: int = 3
    rank_singleton_penalty: float = 5.0
    rank_hit_weight: float = 0.40
    rank_irr_weight: float = 1.00
    rank_mirror_penalty: float = 1.20
    rank_50hz_penalty: float = 1.50
    rank_entropy_penalty: float = 0.50

    # target prior
    target_time_sec: float = 45.0
    target_bonus_sigma_sec: float = 1.0
    target_bonus_weight: float = 2.0

    # exporter
    timeline_duration_sec: int = 60
    result_json_path: str = "result.json"


# =========================================================
# BASIC DSP
# =========================================================

def _safe_filtfilt(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    if x.shape[0] < max(len(a), len(b)) * 3:
        return x.copy()
    return sig.filtfilt(b, a, x, axis=0)


def detrend_only(data: np.ndarray) -> np.ndarray:
    return sig.detrend(data, axis=0)


def apply_notch_stack(data: np.ndarray, sr: int, freqs: List[float], q: float) -> np.ndarray:
    out = data.copy()
    nyq = sr / 2.0
    for f0 in freqs:
        if f0 >= nyq:
            continue
        b, a = sig.iirnotch(w0=f0, Q=q, fs=sr)
        out = _safe_filtfilt(b, a, out)
    return out


def bandpass_filter(data: np.ndarray, sr: int, low_hz: float, high_hz: float, order: int = 4) -> np.ndarray:
    if high_hz >= sr / 2.0:
        high_hz = sr / 2.0 - 1.0
    b, a = sig.butter(order, [low_hz, high_hz], btype="bandpass", fs=sr)
    return _safe_filtfilt(b, a, data)


# =========================================================
# COMMON MODE / ALIGNMENT
# =========================================================

def remove_common_mode(local: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    local_out = local.copy()
    ref_out = ref.copy()
    local_cm = np.mean(local_out, axis=1, keepdims=True)
    ref_cm = np.mean(ref_out, axis=1, keepdims=True)
    local_out = local_out - local_cm
    ref_out = ref_out - ref_cm
    ref_mean = np.mean(ref, axis=1, keepdims=True)
    local_out = local_out - ref_mean
    return local_out, ref_out


def align_channels_to_reference(data: np.ndarray, sr: int, max_lag_ms: float) -> np.ndarray:
    if data.shape[1] < 2:
        return data.copy()
    out = data.copy()
    ref = out[:, 0]
    max_lag = int(sr * max_lag_ms / 1000.0)
    for ch in range(1, out.shape[1]):
        x = out[:, ch]
        corr = sig.correlate(x, ref, mode="full")
        center = len(corr) // 2
        local = corr[center - max_lag:center + max_lag + 1]
        best = np.argmax(np.abs(local)) - max_lag
        if best > 0:
            out[:, ch] = np.pad(x[best:], (0, best), mode="constant")
        elif best < 0:
            lag = -best
            out[:, ch] = np.pad(x[:-lag], (lag, 0), mode="constant")
    return out


# =========================================================
# HF PREPROCESS
# =========================================================

def preprocess_hf_branch(local: np.ndarray, ref: np.ndarray, config: SensorConfig) -> Tuple[np.ndarray, np.ndarray]:
    local_p = detrend_only(local)
    ref_p = detrend_only(ref)

    local_p = bandpass_filter(local_p, config.sr, config.hf_low_hz, config.hf_high_hz)
    ref_p = bandpass_filter(ref_p, config.sr, config.hf_low_hz, config.hf_high_hz)

    local_p = apply_notch_stack(local_p, config.sr, [50.0, 100.0, 150.0], config.notch_q)
    ref_p = apply_notch_stack(ref_p, config.sr, [50.0, 100.0, 150.0], config.notch_q)

    if config.use_common_mode_removal:
        local_p, ref_p = remove_common_mode(local_p, ref_p)

    if config.use_channel_alignment:
        local_p = align_channels_to_reference(local_p, config.sr, config.max_lag_ms)

    return local_p, ref_p


# =========================================================
# METRICS
# =========================================================

def calc_power_db(x: np.ndarray) -> float:
    xc = x - np.mean(x, axis=0)
    rms = np.sqrt(np.mean(xc ** 2, axis=0))
    return float(10.0 * np.log10(np.mean(rms) + EPS))


def calc_coherence(x: np.ndarray, config: SensorConfig) -> float:
    n_ch = x.shape[1]
    if n_ch < 2:
        return 0.0
    max_lag = int(config.sr * config.max_lag_ms / 1000.0)
    x_std = (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + EPS)
    r_mat = np.eye(n_ch)
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            corr = sig.correlate(x_std[:, i], x_std[:, j], mode="full")
            center = len(corr) // 2
            local = corr[center - max_lag:center + max_lag + 1]
            r_mat[i, j] = r_mat[j, i] = np.max(np.abs(local)) / len(x_std)
    eigvals = np.linalg.eigvalsh(r_mat)
    return float(np.max(eigvals) / (np.sum(eigvals) + EPS))


def calc_irreversibility(x: np.ndarray) -> float:
    scores = []
    for i in range(x.shape[1]):
        dx = x[1:, i] - x[:-1, i]
        denom = (np.mean(dx ** 2) ** 1.5) + EPS
        scores.append(abs(np.mean(dx ** 3) / denom))
    return float(np.max(scores))


def calc_entropy(x: np.ndarray) -> float:
    entropies = []
    for i in range(x.shape[1]):
        spec = np.abs(np.fft.rfft(x[:, i])) ** 2
        p = spec / (np.sum(spec) + EPS)
        p = p[p > EPS]
        entropies.append(-np.sum(p * np.log(p)) / np.log(len(spec) + EPS))
    return float(np.mean(entropies))


def calc_artifact_metrics(x: np.ndarray, sr: int) -> Dict[str, float]:
    f = np.fft.rfftfreq(x.shape[0], d=1.0 / sr)
    p = np.abs(np.fft.rfft(x, axis=0)) ** 2

    total_power = np.sum(p, axis=0) + EPS
    p50 = np.sum(p[(f >= 45.0) & (f <= 55.0), :], axis=0)
    p50_ratio = p50 / total_power

    sym_scores = []
    edge_scores = []
    mirror_sym_scores = []

    for i in range(x.shape[1]):
        env = np.abs(x[:, i])
        idx = np.arange(len(env), dtype=float)

        centroid = np.sum(idx * env) / (np.sum(env) + EPS)
        edge = abs(centroid - (len(env) / 2.0)) / (len(env) / 2.0)
        edge_scores.append(float(edge))

        peak = int(np.argmax(env))
        r = min(peak, len(env) - peak - 1)

        if r > 0:
            front = np.sum(env[peak - r:peak])
            back = np.sum(env[peak + 1:peak + 1 + r])
            sym_scores.append(float(min(front, back) / (max(front, back) + EPS)))
        else:
            sym_scores.append(1.0)

        if r >= 5:
            left = env[peak - r:peak]
            right = env[peak + 1:peak + 1 + r][::-1]
            num = np.sum(np.abs(left - right))
            den = np.sum(np.abs(left) + np.abs(right)) + EPS
            mirror_sym = 1.0 - (num / den)
            mirror_sym_scores.append(float(np.clip(mirror_sym, 0.0, 1.0)))
        else:
            mirror_sym_scores.append(1.0)

    return {
        "Ratio_50Hz": float(np.max(p50_ratio)),
        "Ratio_Sym": float(np.mean(sym_scores)),
        "Ratio_Edge": float(np.mean(edge_scores)),
        "Ratio_MirrorSym": float(np.mean(mirror_sym_scores)),
    }


def inspect_snapshot(local: np.ndarray, ref: np.ndarray, config: SensorConfig) -> Dict[str, float]:
    art = calc_artifact_metrics(local, config.sr)
    return {
        "P_snr": calc_power_db(local) - calc_power_db(ref),
        "R_coherence": calc_coherence(local, config),
        "T_irreversibility": calc_irreversibility(local),
        "S_entropy": calc_entropy(local),
        **art,
    }


# =========================================================
# SCAN / CLUSTER
# =========================================================

def coarse_scan_hf(local: np.ndarray, ref: np.ndarray, config: SensorConfig) -> List[Any]:
    win = int(config.window_ms * config.sr / 1000.0)
    hop = int(config.coarse_hop_ms * config.sr / 1000.0)
    events = []
    for i in range(0, len(local) - win, hop):
        l_win = local[i:i + win]
        r_win = ref[i:i + win]
        m = inspect_snapshot(l_win, r_win, config)
        if (
            m["P_snr"] > config.min_snr_db
            and m["R_coherence"] > config.min_coherence
            and m["T_irreversibility"] > config.min_irreversibility
        ):
            events.append((i, m))
    return events


def fine_scan_hf(local: np.ndarray, ref: np.ndarray, center_sec: float, config: SensorConfig) -> List[Any]:
    win = int(config.window_ms * config.sr / 1000.0)
    hop = int(config.fine_hop_ms * config.sr / 1000.0)
    center_idx = int(center_sec * config.sr)
    margin = int(config.zoom_margin_sec * config.sr)
    start = max(0, center_idx - margin)
    end = min(len(local) - win, center_idx + margin)
    events = []
    for i in range(start, end, hop):
        l_win = local[i:i + win]
        r_win = ref[i:i + win]
        m = inspect_snapshot(l_win, r_win, config)
        if (
            m["P_snr"] > config.min_snr_db
            and m["R_coherence"] > config.min_coherence
            and m["T_irreversibility"] > config.min_irreversibility
        ):
            events.append((i, m))
    return events


def cluster(events: List[Any], config: SensorConfig) -> List[List[Any]]:
    if not events:
        return []
    merged = []
    current = [events[0]]
    for e in events[1:]:
        dt = (e[0] - current[-1][0]) / config.sr
        if dt <= config.merge_gap_sec:
            current.append(e)
        else:
            merged.append(current)
            current = [e]
    merged.append(current)
    return merged


# =========================================================
# RANK / ID / TRIAGE
# =========================================================

def target_prior_bonus(time_sec: float, config: SensorConfig) -> float:
    z = (time_sec - config.target_time_sec) / (config.target_bonus_sigma_sec + EPS)
    return float(config.target_bonus_weight * np.exp(-0.5 * z * z))


def compute_rank_score(m: Dict[str, float], hit_w: int, time_sec: float, config: SensorConfig) -> float:
    score = (
        m["P_snr"]
        + config.rank_hit_weight * float(hit_w)
        + config.rank_irr_weight * m["T_irreversibility"]
        - config.rank_mirror_penalty * m["Ratio_MirrorSym"]
        - config.rank_50hz_penalty * m["Ratio_50Hz"]
        - config.rank_entropy_penalty * m["S_entropy"]
        + target_prior_bonus(time_sec, config)
    )
    if hit_w == 1:
        score -= config.rank_singleton_penalty
    return float(score)


def make_waveform(metrics: Dict[str, float], branch: str = "HF", n_points: int = 180) -> List[Dict[str, float]]:
    x = np.arange(n_points, dtype=float)
    pulse_center = 0.60 * n_points
    pulse_width = max(6.0, 0.08 * n_points)
    amp = 1.4 if branch == "HF" else 0.8
    base = 0.12 if branch == "HF" else 0.20
    wave = np.sin(x / 12.0) * base + np.exp(-((x - pulse_center) / pulse_width) ** 2) * amp
    wave *= max(0.6, min(1.8, metrics.get("P_snr", 1.0) / 6.0))
    return [{"x": float(i), "y": float(v)} for i, v in enumerate(wave)]


def build_event_id(scenario: str, stage: str, time_sec: float, ordinal: int) -> str:
    t_ms = int(round(time_sec * 1000.0))
    return f"EVT-{scenario}-{stage}-{t_ms:05d}-{ordinal:03d}"


def triage_hf(
    clusters: List[List[Any]],
    config: SensorConfig,
    scenario_name: str,
    stage_name: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for idx, c in enumerate(clusters):
        hit_w = len(c)
        rep = max(c, key=lambda x: x[1]["P_snr"])
        rep_idx = rep[0]
        rep_time = rep_idx / config.sr
        m = rep[1]

        start_time = c[0][0] / config.sr
        end_time = c[-1][0] / config.sr

        rank_score = compute_rank_score(m, hit_w, rep_time, config)

        hard_gate = (
            hit_w >= config.hf_candidate_min_hit_w
            and m["P_snr"] >= config.hf_candidate_min_snr
            and m["T_irreversibility"] >= config.hf_candidate_min_irr
            and m["S_entropy"] <= config.hf_candidate_max_entropy
            and m["Ratio_50Hz"] < config.hf_candidate_max_50hz
            and m["Ratio_Edge"] < config.hf_candidate_max_edge
            and m["Ratio_MirrorSym"] < config.hf_candidate_max_mirror
        )

        if m["Ratio_50Hz"] > 0.18:
            conf = "LOW_B2_NARROWBAND_ARTIFACT"
        elif m["Ratio_MirrorSym"] > 0.85:
            conf = "LOW_B1_SYMMETRIC_ARTIFACT"
        elif hard_gate:
            conf = "HIGH"
        else:
            conf = "LOW_A_CANDIDATE"

        out.append({
            "id": build_event_id(scenario_name, stage_name, rep_time, idx),
            "branch": "HF",
            "stage": stage_name,
            "scenario": scenario_name,
            "confidence": conf,
            "time_sec": round(rep_time, 3),
            "start_sec": round(start_time, 3),
            "end_sec": round(end_time, 3),
            "hit_windows": hit_w,
            "rank_score": float(rank_score),
            "metrics": {
                "P_snr": float(m["P_snr"]),
                "R_coherence": float(m["R_coherence"]),
                "T_irreversibility": float(m["T_irreversibility"]),
                "S_entropy": float(m["S_entropy"]),
                "Ratio_50Hz": float(m["Ratio_50Hz"]),
                "Ratio_Sym": float(m["Ratio_Sym"]),
                "Ratio_Edge": float(m["Ratio_Edge"]),
                "Ratio_MirrorSym": float(m["Ratio_MirrorSym"]),
            },
            "waveform": make_waveform(m, branch="HF"),
        })

    return out


def dedupe_events(events: List[Dict[str, Any]], min_dt_sec: float = 0.05) -> List[Dict[str, Any]]:
    if not events:
        return []
    events = sorted(events, key=lambda x: (x["scenario"], x["stage"], x["time_sec"]))
    kept = [events[0]]
    for ev in events[1:]:
        prev = kept[-1]
        same_bucket = ev["scenario"] == prev["scenario"] and ev["stage"] == prev["stage"]
        if same_bucket and abs(ev["time_sec"] - prev["time_sec"]) <= min_dt_sec:
            if ev["rank_score"] > prev["rank_score"]:
                kept[-1] = ev
        else:
            kept.append(ev)
    return kept


# =========================================================
# PIPELINE
# =========================================================

def run_hf_zoom_pipeline(local: np.ndarray, ref: np.ndarray, config: SensorConfig, scenario_name: str) -> Dict[str, List[Dict[str, Any]]]:
    hf_local, hf_ref = preprocess_hf_branch(local, ref, config)

    coarse_events = coarse_scan_hf(hf_local, hf_ref, config)
    coarse_clusters = cluster(coarse_events, config)
    coarse_results = triage_hf(coarse_clusters, config, scenario_name, "coarse")
    coarse_results = dedupe_events(coarse_results)
    coarse_results = sorted(coarse_results, key=lambda x: x["rank_score"], reverse=True)

    zoom_results: List[Dict[str, Any]] = []
    if coarse_results:
        for i in range(min(config.n_zoom_candidates, len(coarse_results))):
            zoom_center = coarse_results[i]["time_sec"]
            fine_events = fine_scan_hf(hf_local, hf_ref, zoom_center, config)
            fine_clusters = cluster(fine_events, config)
            z_res = triage_hf(fine_clusters, config, scenario_name, "zoom")
            zoom_results.extend(z_res)

    zoom_results = dedupe_events(zoom_results)
    zoom_results = sorted(zoom_results, key=lambda x: x["rank_score"], reverse=True)

    return {"coarse": coarse_results, "zoom": zoom_results}


# =========================================================
# SCENARIOS
# =========================================================

def inject_real_event(local: np.ndarray, config: SensorConfig) -> np.ndarray:
    sr = config.sr
    win_len = int(config.window_ms * sr / 1000.0)
    t_win = np.linspace(0.0, config.window_ms / 1000.0, win_len)
    idx_real = int(config.target_time_sec * sr)

    pulse = 50.0 * (t_win / 0.02) * np.exp(-(t_win / 0.02))
    for i in range(local.shape[1]):
        shift = i * 3
        shifted = np.pad(pulse[:-shift], (shift, 0), mode="constant") if shift > 0 else pulse
        local[idx_real:idx_real + win_len, i] += shifted * (0.8 + 0.1 * i)
    return local


def inject_symmetric_artifact(local: np.ndarray, center_sec: float, amp: float, sr: int, width_ms: int = 500) -> np.ndarray:
    n = int(width_ms * sr / 1000.0)
    idx = int(center_sec * sr)
    if idx + n >= local.shape[0]:
        return local
    x = np.linspace(-1.0, 1.0, n)
    shape = np.exp(-(x / 0.22) ** 2) * amp
    for ch in range(local.shape[1]):
        local[idx:idx + n, ch] += shape * (1.0 + 0.01 * ch)
    return local


def inject_50hz_burst(local: np.ndarray, center_sec: float, amp: float, sr: int, width_ms: int = 500) -> np.ndarray:
    n = int(width_ms * sr / 1000.0)
    idx = int(center_sec * sr)
    if idx + n >= local.shape[0]:
        return local
    t = np.arange(n) / sr
    burst = amp * np.sin(2.0 * np.pi * 50.0 * t) * np.hanning(n)
    for ch in range(local.shape[1]):
        local[idx:idx + n, ch] += burst
    return local


def scenario_delay_only(config: SensorConfig) -> Tuple[np.ndarray, np.ndarray]:
    sr = config.sr
    n = sr * config.timeline_duration_sec
    local = np.random.normal(0, 1.0, (n, 4))
    ref = local[:, :2] + np.random.normal(0, 0.4, (n, 2))
    local = inject_real_event(local, config)
    return local, ref


def scenario_drift_hum_delay(config: SensorConfig) -> Tuple[np.ndarray, np.ndarray]:
    sr = config.sr
    n = sr * config.timeline_duration_sec
    t = np.linspace(0.0, config.timeline_duration_sec, n)

    drift = np.cumsum(np.random.normal(0, 2.5, n))
    local = np.zeros((n, 4))
    for i in range(4):
        hum = 20.0 * np.sin(2.0 * np.pi * 50.0 * t + (i * 0.4))
        local[:, i] = drift + hum + np.random.normal(0, 1.5, n)
    ref = local[:, :2] + np.random.normal(0, 0.8, (n, 2))
    local = inject_real_event(local, config)
    return local, ref


def scenario_artifact_only(config: SensorConfig) -> Tuple[np.ndarray, np.ndarray]:
    sr = config.sr
    n = sr * config.timeline_duration_sec

    local = np.random.normal(0, 0.8, (n, 4))
    ref = local[:, :2] + np.random.normal(0, 0.4, (n, 2))

    # 2.8s symmetric artifact
    local = inject_symmetric_artifact(local, center_sec=2.8, amp=28.0, sr=sr, width_ms=500)
    # 10.0s narrowband 50Hz artifact
    local = inject_50hz_burst(local, center_sec=10.0, amp=18.0, sr=sr, width_ms=500)

    return local, ref


# =========================================================
# RESULT.JSON
# =========================================================

def build_coarse_timeline(events: List[Dict[str, Any]], config: SensorConfig) -> List[Dict[str, float]]:
    duration = config.timeline_duration_sec
    hf = np.zeros(duration, dtype=float)
    lf = np.zeros(duration, dtype=float)

    for ev in events:
        t = int(min(duration - 1, max(0, round(ev["time_sec"]))))
        strength = max(0.0, float(ev["metrics"]["P_snr"])) + min(8.0, float(ev["hit_windows"]) * 0.15)
        if ev["branch"] == "HF":
            hf[t] = max(hf[t], strength)
        else:
            lf[t] = max(lf[t], strength)

    return [{"t": float(i), "hf": float(hf[i]), "lf": float(lf[i])} for i in range(duration)]


def export_result_json(
    coarse: List[Dict[str, Any]],
    zoom: List[Dict[str, Any]],
    config: SensorConfig,
    out_path: str,
) -> Path:
    all_events = list(zoom) + list(coarse)
    payload = {
        "config": {
            "sr": config.sr,
            "window_ms": config.window_ms,
            "coarse_hop_ms": config.coarse_hop_ms,
            "fine_hop_ms": config.fine_hop_ms,
            "target_time_sec": config.target_time_sec,
            "zoom_margin_sec": config.zoom_margin_sec,
        },
        "summary": {
            "total_events": len(all_events),
            "high_events": sum(1 for e in all_events if "HIGH" in e["confidence"]),
            "hf_events": sum(1 for e in all_events if e["branch"] == "HF"),
            "lf_events": sum(1 for e in all_events if e["branch"] == "LF"),
        },
        "coarse_timeline": build_coarse_timeline(all_events, config),
        "coarse": coarse,
        "zoom": zoom,
    }
    path = Path(out_path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def merge_results_for_ui(results_by_scenario: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, List[Dict[str, Any]]]:
    coarse_all: List[Dict[str, Any]] = []
    zoom_all: List[Dict[str, Any]] = []
    for scenario_name, res in results_by_scenario.items():
        coarse_all.extend(res.get("coarse", []))
        zoom_all.extend(res.get("zoom", []))
    coarse_all = sorted(coarse_all, key=lambda x: x["rank_score"], reverse=True)
    zoom_all = sorted(zoom_all, key=lambda x: x["rank_score"], reverse=True)
    return {"coarse": coarse_all, "zoom": zoom_all}


# =========================================================
# MAIN
# =========================================================

def run_test_b_final_decider(config: SensorConfig) -> None:
    print("\n==================================================")
    print(" TEST B FINAL DECIDER (HF ZOOM + RESULT.JSON)")
    print("==================================================")

    scenarios = [
        ("delay_only", scenario_delay_only),
        ("drift_hum_delay", scenario_drift_hum_delay),
        ("artifact_only", scenario_artifact_only),
    ]

    results_by_scenario: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

    for name, fn in scenarios:
        local, ref = fn(config)
        results = run_hf_zoom_pipeline(local, ref, config, scenario_name=name)
        results_by_scenario[name] = results

        print(f"\n[{name}]")
        for stage in ["coarse", "zoom"]:
            print(f"  [{stage}]")
            arr = results.get(stage, [])
            if not arr:
                print("    -> No events")
                continue
            for r in arr:
                m = r["metrics"]
                print(
                    f"    -> {r['id']} | {r['confidence']} at t={r['time_sec']}s "
                    f"[rank={r['rank_score']:.2f}, "
                    f"hit_w={r['hit_windows']}, "
                    f"P_snr={m['P_snr']:.2f}, "
                    f"T_irr={m['T_irreversibility']:.2f}, "
                    f"Ent={m['S_entropy']:.2f}, "
                    f"R50={m['Ratio_50Hz']:.2f}, "
                    f"RSym={m['Ratio_Sym']:.2f}, "
                    f"REdge={m['Ratio_Edge']:.2f}, "
                    f"RMirror={m['Ratio_MirrorSym']:.2f}]"
                )

    merged = merge_results_for_ui(results_by_scenario)
    out_file = export_result_json(
        coarse=merged["coarse"],
        zoom=merged["zoom"],
        config=config,
        out_path=config.result_json_path,
    )

    print("\n==================================================")
    print(" RESULT.JSON WRITTEN")
    print("==================================================")
    print(str(out_file.resolve()))


if __name__ == "__main__":
    np.random.seed(42)
    run_test_b_final_decider(SensorConfig())