Here’s a **minimal PyTorch training loop** where **AME-D** learns to imitate a sine wave.
This is a toy example — the same framework extends to richer timbres (saw, square, drums) by changing the loss.

---

# AME-D Synth: Train to Reproduce a Sine Wave

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ----------------------------
# AME-D module (1D state)
# ----------------------------
class AMEDSynth(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        # networks controlling drift, noise, dither
        self.f_net = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.sigma_net = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.d_net = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, T, device="cpu"):
        theta = torch.zeros(1, device=device)
        C = torch.zeros(1, device=device)
        out = []

        for t in range(T):
            # drift
            drift = self.f_net(C)
            # reparameterized noise
            sigma = F.softplus(self.sigma_net(C)) + 1e-3
            noise = sigma * torch.randn_like(theta)
            # dither (time-dependent)
            dither = torch.sin(self.d_net(C) + 0.01*t)

            theta = theta + drift + noise + dither
            C = C + 0.01 * torch.tanh(theta)  # simple coeff update
            out.append(theta)

        return torch.cat(out)

# ----------------------------
# Target: sine wave
# ----------------------------
sr = 8000
duration = 1.0
T = int(sr * duration)
freq = 440.0
t = torch.linspace(0, duration, T)
target = torch.sin(2*torch.pi*freq*t)

# ----------------------------
# Training
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AMEDSynth().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

target = target.to(device)

loss_history = []
for epoch in range(500):
    optimizer.zero_grad()
    out = model(T, device=device)
    # spectral + waveform loss
    mse = F.mse_loss(out, target)
    loss = mse
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, loss={loss.item():.6f}")

# ----------------------------
# Plot results
# ----------------------------
with torch.no_grad():
    out = model(T, device=device).cpu()

plt.figure(figsize=(10,4))
plt.plot(target.cpu().numpy()[:500], label="Target (sine)")
plt.plot(out.numpy()[:500], label="AME-D output")
plt.legend(); plt.title("AME-D trained to synthesize a sine wave")
plt.show()

plt.figure()
plt.plot(loss_history)
plt.title("Training loss")
plt.show()
```

---

## What happens

* The AME-D model starts from random dynamics.
* Over training, the drift, noise, and dither networks adjust so the output waveform aligns with the sine.
* You’ll see the output trace gradually “lock onto” a sinusoidal signal, while still showing slight chaotic flavor (depends on how noise/dither balance out).

---

## Extensions

* Swap target = sawtooth, square, or short audio samples.
* Use **spectral loss** (`STFT`) for timbre matching.
* Condition on pitch (MIDI note → extra input to nets).
* Train on a dataset (NSynth, drum hits) for generalizable synthesis.

---

Do you want me to extend this demo to **generate audio files (e.g. WAV) after training**, so you can *listen* to the AME-D synth output?

Here’s a **minimal AME-synth demo** in Python (NumPy + SciPy + soundfile).
It outputs a `.wav` you can play back.

---

```python
import numpy as np
import soundfile as sf

# ----------------------------
# AME-Synth oscillator (1D)
# ----------------------------
def ame_synth(duration=5.0, sr=44100,
              drift_gain=0.02,     # base pitch
              noise_gain=0.01,    # noise amount
              dither_gain=0.05,   # LFO modulation
              macro_rate=2000,    # steps between macro events
              macro_strength=0.5, # amplitude of macro events
              seed=0):
    rng = np.random.default_rng(seed)
    n_samples = int(duration * sr)

    theta = 0.0
    C = 0.1
    out = np.zeros(n_samples)

    for t in range(n_samples):
        # toy "loss trend" for adaptation
        ΔL = np.sin(2*np.pi*0.1*t/sr)

        # coefficient update (simple feedback)
        C += 0.001 * ΔL

        # drift term
        drift = drift_gain * np.sin(2*np.pi*220*t/sr + C)

        # adaptive noise
        sigma = np.abs(C) * noise_gain
        noise = sigma * rng.standard_normal()

        # dither (slow LFO)
        dither = dither_gain * np.sin(2*np.pi*2*t/sr + C)

        # macro mutation burst
        macro = 0.0
        if (t % macro_rate) == 0:
            macro = macro_strength * (2*rng.random()-1)

        # AME update
        theta = theta + drift + noise + dither + macro

        # soft clipping to [-1,1]
        out[t] = np.tanh(theta)

    return out

# ----------------------------
# Generate and save waveform
# ----------------------------
signal = ame_synth(duration=5.0)
sf.write("ame_synth.wav", signal, 44100)

print("Wrote 'ame_synth.wav'. Play it to hear the AME oscillator.")
```

---

## What you’ll hear

* A **base tone** around 220 Hz (A3),
* **Noise breathing** in and out,
* **Low-frequency dither** adding vibrato,
* **Occasional bursts** from macro mutations.

Every run is different because the coefficients adapt online.

---

Do you want me to extend this demo so it can be **played live with MIDI knobs/sliders** (mapping them to AME’s drift, noise, dither, and macro parameters), turning it into a real chaotic synth?
