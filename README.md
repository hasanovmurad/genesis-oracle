# Genesis Oracle

This project was developed for the course *Angewandte Modellierung und Systemsimulation (SoSe 2026)*.

It combines physical system simulation with machine learning using Keras 3 (JAX backend).

---

## Overview

- Simulation of continuous systems (ODEs)
- Synthetic signal generation (RC filter + Fourier series)
- Autoencoder-based anomaly detection
- Conv1D model for time-series learning

---

## Results

The trained autoencoder detects anomalies by increased reconstruction error:

![Autoencoder Reconstruction Loss](anomaly_detection_plot.png)

---

## Repository Structure
```genesis-oracle/
│
├── src/
│ ├── architecture.py
│ ├── data_generator.py
│ └── oracle_setup.py
│
├── data/
│ └── rc_signal.npy
│
├── index.md
├── README.md
├── pyproject.toml
└── uv.lock
```
---
## Live Project

GitHub Pages:  
https://hasanovmurad.github.io/genesis-oracle/