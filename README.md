# non-equilibrium-auditor

**Time-Series Event Auditing for Structural Collapse Detection**

`non-equilibrium-auditor` is an experimental toolkit designed to detect and audit **non-equilibrium transitions** and **structural collapse events** in complex, noisy multi-channel time-series data.

Unlike traditional anomaly detection, this system focuses on the **regime break**—the moment a system crosses a stability barrier and enters a collapsed state.

---

## Concept: Structural Collapse

Many real-world systems remain in a **stable basin** until a perturbation pushes them across a **barrier**. This transition is rarely a simple spike; it is a structural shift characterized by:

* **Precursor Activity**: Subtle signs of instability before the main break.
* **Winner Event**: The primary transition that marks the regime shift.
* **Non-Equilibrium Asymmetry**: A physical "arrow of time" that makes the collapse irreversible.

---

## System Architecture
![system architecture diagram](architecture.svg)

The pipeline separates signal analysis into a **Dual Branch** architecture to isolate the event from its environment.

### LF Branch — Environment Map

Tracks slow environmental behavior, baseline drift, and long-term oscillations. It acts as a reference map to ensure environmental noise isn't mislabeled as a structural event.

### HF Branch — Transition Scan

Searches for rapid structural changes. This branch evaluates potential candidates using metrics like local SNR, irreversibility, and persistence.

---

## Audit & Classification

Detected candidates are categorized into a discussable and inspectable hierarchy:

* **Winner**: The strongest candidate transition near the target window.
* **Precursor**: Supporting transitions that precede the main event.
* **Artifact / Environment / Noise**: Rejected candidates based on symmetric artifacts, narrowband contamination, or long-term drift.

---

## Event Metrics

Each event is audited using interpretable physical metrics:

* **P_snr**: Local signal contrast against surrounding noise.
* **T_irreversibility**: Measures the transition-like asymmetry between rising and falling phases.
* **Persistence**: Counts supporting adjacent windows (`hit_windows`) to ensure the event has physical substance.
* **Artifact Ratios**: Rejects symmetric patterns (`Ratio_MirrorSym`) often produced by sensor glitches or DSP side-effects.

---

## Components

### 1. Python Engine (`death_audit_v15.py`)

The core backend that processes raw multi-channel data and ranks events.

### 2. Interactive UI (`index.html`)

A browser-based analysis dashboard to visualize, verify, and audit detected events without additional software.

---

## License

© DIMProductions
Research use only.
