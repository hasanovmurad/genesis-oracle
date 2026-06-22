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

### Week 6 – Chaos Engine

Week 6 extends the project into stochastic system simulation and Monte Carlo modeling.

Implemented components:

- Classical NumPy Monte Carlo Pi estimation
- JAX-based Monte Carlo revenue simulation with 1,000,000 paths
- Expected revenue and 95% Value-at-Risk calculation
- JAX compilation profiling
- Markov Chain simulation of macro-economic states
- Black Swan shock injection from day 180 to day 190
- Swarm stress report

Results:

- Pi estimate: 3.142768
- Classical Pi runtime: 1.087658 s
- Expected revenue: 416021.97
- VaR 95% threshold: 257700.36
- JAX first run: 1.653376 s
- JAX second run: 0.226458 s

Generated assets:

- `data/classical_pi_disp.png`
- `data/revenue_dist.png`
- `data/markov_states.png`
- `docs/Swarm_Stress_Report.md`

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
|   ├── classical_pi.py
|   ├── monte_carlo.py
|   ├── markov_boss.py
|   └── fabric_pinn.py
│
├── docs/
│   |── Ascension_Report.md
|   ├── Fabric_Report.md
|   ├── Swarm_Stress_Report.md
|   └── fabric_surface.png
|
├── data/
|   ├── rc_signal.npy
|   ├── classical_pi_disp.png
|   ├── revenue_dist.png
|   ├── markov_states.png
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
