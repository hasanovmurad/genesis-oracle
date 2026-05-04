# Genesis Oracle

This project was developed for the course *Angewandte Modellierung und Systemsimulation (SoSe 2026)*.  
It combines classical physical system modeling with modern machine learning approaches using Keras 3 and JAX.

---

## Week 1 – Continuous Systems

- Simulation of physical systems using `scipy.integrate.solve_ivp`
- Harmonic oscillator (second-order ODE)
- Radioactive decay (first-order ODE)
- Visualization of continuous dynamics

---

## Week 2 – Synthetic Data Generation (RC Filter)

- Square wave generation using Fourier series
- Analytical RC low-pass filtering in frequency domain
- Addition of Gaussian noise
- Injection of artificial anomaly (high-frequency spike)

Output:
- `data_feed.png`
- Dataset used for ML: `data/rc_signal.npy`

---

## Week 3 – Autoencoder & Anomaly Detection

### Dense Autoencoder

A subclassed Keras model was implemented:

- Input: time-series windows of length 50
- Encoder: 50 → 8 (latent representation)
- Decoder: 8 → 50 (reconstruction)

The model is trained only on normal signal data.  
When anomalous regions are processed, reconstruction error increases significantly.

### Anomaly Detection Result

![Autoencoder Reconstruction Loss](anomaly_detection_plot.png)

The reconstruction error remains low for normal regions and shows a clear spike at the injected anomaly.

---

## Conv1D Autoencoder (Refactoring)

The dense architecture was extended using convolutional layers:

- Encoder: `Conv1D`
- Decoder: `Conv1DTranspose`

### Why Conv1D?

Conv1D layers operate on local temporal patterns instead of the entire signal at once.  
This allows better detection of:

- short-term distortions
- spikes
- local signal variations

Therefore, convolutional models are more suitable for time-series anomaly detection.

---

## Technologies

- Python
- NumPy / SciPy
- Matplotlib
- Keras 3 with JAX backend
- Google Colab (for training)

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

## Live Project

GitHub Pages:  
https://hasanovmurad.github.io/genesis-oracle/