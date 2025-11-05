#!/usr/bin/env python3
import numpy as np
import h5py
from scipy.integrate import solve_ivp
from scipy import signal
from pathlib import Path

MASS, SPRING_K, DAMPING_C = 1.0, 1.0, 0.1
SAMPLE_RATE, DURATION = 100.0, 10.0
DT = 1.0 / SAMPLE_RATE
T_EVAL = np.arange(0, DURATION, DT)
N_SAMPLES = len(T_EVAL)

def dynamics(t, state, u_func, m, k, c):
    x, v = state
    return [v, (u_func(t) - c * v - k * x) / m]

def simulate(u_func, x0=0.0, v0=0.0):
    sol = solve_ivp(
        lambda t, s: dynamics(t, s, u_func, MASS, SPRING_K, DAMPING_C),
        (0.0, DURATION), [x0, v0], t_eval=T_EVAL, method='RK45',
        rtol=1e-6, atol=1e-9
    )
    return sol.y[0], sol.y[1], np.array([u_func(t) for t in sol.t])

def step(step_time=2.0, amp=1.0):
    return lambda t: amp if t >= step_time else 0.0

def sine(freq=1.0, amp=1.0, phase=0.0):
    return lambda t: amp * np.sin(2 * np.pi * freq * t + phase)

def chirp(f0=0.1, f1=5.0, amp=1.0):
    return lambda t: amp * np.sin(2 * np.pi * (f0 + (f1 - f0) * t / DURATION) * t)

def square_wave(freq=0.5, amp=1.0, duty=0.5):
    return lambda t: amp * signal.square(2 * np.pi * freq * t, duty=duty)

def random_signal(seed=42, amp=1.0, cutoff=5.0):
    rng = np.random.RandomState(seed)
    noise = rng.randn(N_SAMPLES)
    sos = signal.butter(4, cutoff, fs=SAMPLE_RATE, output='sos')
    filtered = signal.sosfilt(sos, noise)
    filtered = filtered / np.max(np.abs(filtered)) * amp
    return lambda t: filtered[min(int(t * SAMPLE_RATE), len(filtered) - 1)]

def ramp(start_time=1.0, slope=0.5):
    return lambda t: 0.0 if t < start_time else slope * (t - start_time)

def pulse_train(freq=1.0, amp=1.0, width=0.1):
    period = 1.0 / freq
    return lambda t: amp if (t % period) < width else 0.0

def multisine(freqs=[0.5, 1.5, 3.0], amps=[0.5, 0.3, 0.2]):
    return lambda t: sum(a * np.sin(2 * np.pi * f * t) for f, a in zip(freqs, amps))

def get_trajectories():
    trajs = []
    for i, (st, a) in enumerate([(1.0, 1.0), (2.0, 2.0), (3.0, -1.5), (1.5, 0.5), (4.0, -0.8)]):
        trajs.append(('step', i, step(st, a)))
    for i, (f, a) in enumerate([(0.5, 1.0), (1.0, 1.5), (2.0, 0.8), (0.3, 1.2), (1.5, 1.0)]):
        trajs.append(('sine', i, sine(f, a)))
    for i, (f0, f1, a) in enumerate([(0.1, 5.0, 1.0), (0.5, 3.0, 1.2), (0.2, 4.0, 0.8), (1.0, 5.0, 1.0), (0.1, 2.0, 1.5)]):
        trajs.append(('chirp', i, chirp(f0, f1, a)))
    for i, (f, a, d) in enumerate([(0.5, 1.0, 0.5), (1.0, 1.5, 0.3), (0.3, 1.2, 0.7), (1.5, 0.8, 0.5), (0.8, 1.0, 0.4)]):
        trajs.append(('square', i, square_wave(f, a, d)))
    for i in range(50):
        trajs.append(('random', i, random_signal(42 + i * 10, 0.8 + i * 0.1, 3.0 + i * 0.5)))
    for i, (st, sl) in enumerate([(1.0, 0.5), (2.0, -0.3), (0.5, 0.8), (1.5, 0.2), (3.0, -0.5)]):
        trajs.append(('ramp', i, ramp(st, sl)))
    for i, (f, a, w) in enumerate([(1.0, 1.5, 0.1), (0.5, 2.0, 0.2), (2.0, 1.0, 0.05), (1.5, 1.2, 0.15), (0.8, 1.8, 0.12)]):
        trajs.append(('pulse', i, pulse_train(f, a, w)))
    for i, (fs, as_) in enumerate([
        ([0.5, 1.5, 3.0], [0.5, 0.3, 0.2]),
        ([0.3, 1.0, 2.5], [0.6, 0.25, 0.15]),
        ([1.0, 2.0, 4.0], [0.4, 0.35, 0.25]),
        ([0.8, 1.8, 3.5], [0.5, 0.3, 0.2]),
        ([0.4, 1.2, 2.8], [0.55, 0.28, 0.17])
    ]):
        trajs.append(('multisine', i, multisine(fs, as_)))
    return trajs

def save_h5(filename, u, x, v):
    with h5py.File(filename, 'w') as f:
        f.create_dataset('u', data=u.astype(np.float32))
        f.create_dataset('x', data=x.astype(np.float32))
        f.create_dataset('v', data=v.astype(np.float32))
        f.attrs['mass'] = MASS
        f.attrs['spring_constant'] = SPRING_K
        f.attrs['damping_coefficient'] = DAMPING_C
        f.attrs['dt'] = DT
        f.attrs['sampling_rate'] = SAMPLE_RATE
        f.attrs['duration'] = DURATION
        f.attrs['n_samples'] = N_SAMPLES
        f.attrs['x0'] = 0.0
        f.attrs['v0'] = 0.0

def generate_dataset():
    base = Path("dataset")
    (base / "train").mkdir(parents=True, exist_ok=True)
    (base / "valid").mkdir(exist_ok=True)
    (base / "test").mkdir(exist_ok=True)

    trajs = get_trajectories()
    np.random.seed(42)
    indices = np.random.permutation(len(trajs))

    n_train = int(len(trajs) * 0.7)
    n_valid = int(len(trajs) * 0.15)

    splits = {
        'train': indices[:n_train],
        'valid': indices[n_train:n_train + n_valid],
        'test': indices[n_train + n_valid:]
    }

    for split, idxs in splits.items():
        for i, idx in enumerate(idxs):
            traj_type, traj_idx, u_func = trajs[idx]
            x, v, u = simulate(u_func)
            filename = base / split / f"trajectory_{traj_type}_{traj_idx:03d}.h5"
            save_h5(str(filename), u, x, v)
            print(f"[{split}] {i+1}/{len(idxs)} {filename.name}")

    print(f"\nGenerated {len(trajs)} trajectories: {n_train} train, {n_valid} valid, {len(splits['test'])} test")

if __name__ == "__main__":
    generate_dataset()
