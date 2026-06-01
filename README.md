# Genesis Oracle

This repository contains the implementation of the **Project Genesis** series developed for the course **Angewandte Modellierung und Systemsimulation (SoSe 2026)**.

The project combines physical system simulation, machine learning, anomaly detection, high-performance numerical computing with JAX, and functional neural network architectures using Flax.

---

## Project Progress

### Week 1 – Foundation

* Local development environment with `uv`
* Project structure and dependency management
* Reproducible software setup

### Week 2 – Synthetic Physics Generation

* RC-filter simulation
* Synthetic signal generation
* Fourier-based signal synthesis
* Dataset generation and storage

### Week 3 – The Oracle Awakens

* Autoencoder architecture in Keras 3
* JAX backend integration
* Anomaly detection through reconstruction error
* Model training in Google Colab

### Week 4 – Silicon Ascension

* Large-scale damped oscillator simulation
* NumPy baseline implementation
* JAX acceleration using:

  * `vmap`
  * `jit`
  * `grad`
* Flax neural network architecture
* Explicit parameter management

---

## Performance Results

| Metric                        |        Result |
| ----------------------------- | ------------: |
| Legacy NumPy Simulation       |    2.766298 s |
| JAX Simulation (2nd Run)      |    0.040943 s |
| Speedup Factor                |        67.56× |
| Optimized Projectile Velocity | 29.999977 m/s |

---

## Autoencoder Anomaly Detection

The trained autoencoder identifies anomalies through increased reconstruction error.

![Autoencoder Reconstruction Loss](anomaly_detection_plot.png)

---

## Repository Structure

```text
genesis-oracle/
│
├── src/
│   ├── architecture.py
│   ├── data_generator.py
│   ├── oracle_setup.py
│   ├── legacy_swarm.py
│   ├── jax_swarm.py
│   └── flax_core.py
│
├── docs/
│   └── Ascension_Report.md
│
├── data/
│   └── rc_signal.npy
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
