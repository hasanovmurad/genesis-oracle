import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === AYAR ===
student_id_last_two = 42  # BURAYI değiştir (son 2 hane)

A = 1.0
f0 = student_id_last_two + 1

periods = 100
samples_per_period = 200

R = 1000
C = 1e-6

noise_std = 0.08

output_dir = Path("data")
output_dir.mkdir(exist_ok=True)

# === ZAMAN ===
T = 1 / f0
duration = periods * T
num_samples = periods * samples_per_period

t = np.linspace(0, duration, num_samples)

# === FOURIER KARE DALGA ===
signal = np.zeros_like(t)

for n in range(1, 18, 2):  # 1,3,5,...17
    signal += (4*A/(np.pi*n)) * np.sin(2*np.pi*n*f0*t)

# === RC FILTER ===
filtered = np.zeros_like(t)

for n in range(1, 18, 2):
    omega = 2*np.pi*n*f0
    H = 1 / np.sqrt(1 + (omega*R*C)**2)
    phi = -np.arctan(omega*R*C)

    filtered += (4*A/(np.pi*n)) * H * np.sin(omega*t + phi)

# === NOISE ===
noise = np.random.normal(0, noise_std, size=t.shape)
noisy = filtered + noise

# === ANOMALY ===
anomaly = noisy.copy()

start = int(70 * samples_per_period)
end = int(75 * samples_per_period)

anomaly[start:end] += 3 * np.sin(2*np.pi*40*f0*t[start:end])

# === SAVE DATA ===
np.save("data/rc_signal.npy", anomaly)

# === PLOT ===
plt.figure(figsize=(10,4))

plt.plot(t[0:1000], anomaly[0:1000], label="Normal")
plt.plot(t[start:end], anomaly[start:end], label="Anomaly")

plt.legend()
plt.title("RC Signal with Anomaly")

plt.savefig("data/data_feed.png")
plt.show()

print("Done")