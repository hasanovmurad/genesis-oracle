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

### Week 5 – The Fabric of Reality

* Physics-Informed Neural Network (PINN) implementation
* Mesh-free collocation sampling in space-time
* Automatic differentiation using `jax.grad`
* Physics residual minimization for the 1D heat equation
* Initial Condition (IC) and Boundary Condition (BC) constraints
* Optax-based optimization
* Interactive Plotly 3D visualization
* Exported HTML simulation for browser-based exploration


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
│   ├── flax_core.py
|   ├── pinn_data.py
|   └── fabric_pinn.py
│
├── docs/
│   |── Ascension_Report.md
|   ├── Fabric_Report.md
|   └── fabric_surface.png
|
├── data/
|   ├── rc_signal.npy
│   └── pinn_3d_fabric.html
│
├── index.md
├── README.md
├── pyproject.toml
└── uv.lock
```

---
## Week 5 Results

Physics-Informed Neural Networks were used to approximate the solution of the 1D heat equation over a continuous space-time domain.

The resulting temperature manifold was visualized as an interactive Plotly 3D surface and exported as a standalone HTML file for browser-based exploration.

See:

* `docs/Fabric_Report.md`
* `data/pinn_3d_fabric.html`

## Live Project

GitHub Pages:

https://hasanovmurad.github.io/genesis-oracle/
