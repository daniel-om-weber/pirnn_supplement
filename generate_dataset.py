#!/usr/bin/env python3
"""
Mass-Spring-Damper System Dataset Generator

Simulates a forced mass-spring-damper system with various input trajectories
and saves the results as HDF5 files for physics-informed neural network training.
"""

import numpy as np
import h5py
from scipy.integrate import solve_ivp
from scipy import signal
from pathlib import Path
from typing import Tuple, Callable, Optional
import os


# ============================================================================
# CONFIGURATION
# ============================================================================

# Physical parameters
MASS = 1.0  # kg
SPRING_CONSTANT = 1.0  # N/m
DAMPING_COEFFICIENT = 0.1  # N*s/m

# Simulation parameters
SAMPLING_RATE = 100.0  # Hz
DURATION = 10.0  # seconds
DT = 1.0 / SAMPLING_RATE  # time step
T_SPAN = (0.0, DURATION)
T_EVAL = np.arange(0, DURATION, DT)
N_SAMPLES = len(T_EVAL)

# Dataset parameters
OUTPUT_DIR = "dataset"
N_TRAJECTORIES_PER_TYPE = 50

# Initial conditions
VARY_INITIAL_CONDITIONS = False  # Set to False for all trajectories to start at rest
X0_RANGE = (-0.5, 0.5)  # position range when varying
V0_RANGE = (-0.5, 0.5)  # velocity range when varying


# ============================================================================
# SYSTEM DYNAMICS
# ============================================================================

def mass_spring_damper_dynamics(t, state, u_func, m, k, c):
    """
    Mass-spring-damper system dynamics: m*a + c*v + k*x = u(t)
    
    Args:
        t: current time
        state: [position, velocity]
        u_func: forcing function u(t)
        m: mass
        k: spring constant
        c: damping coefficient
    
    Returns:
        [velocity, acceleration]
    """
    x, v = state
    u = u_func(t)
    a = (u - c * v - k * x) / m
    return [v, a]


def simulate_system(u_func: Callable, x0: float = 0.0, v0: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate the mass-spring-damper system with given forcing function.
    
    Args:
        u_func: forcing function u(t)
        x0: initial position
        v0: initial velocity
    
    Returns:
        (t, x, v, u): time array, position, velocity, and input arrays
    """
    initial_state = [x0, v0]
    
    sol = solve_ivp(
        fun=lambda t, state: mass_spring_damper_dynamics(t, state, u_func, MASS, SPRING_CONSTANT, DAMPING_COEFFICIENT),
        t_span=T_SPAN,
        y0=initial_state,
        t_eval=T_EVAL,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )
    
    t = sol.t
    x = sol.y[0]
    v = sol.y[1]
    u = np.array([u_func(ti) for ti in t])
    
    return t, x, v, u


# ============================================================================
# INPUT TRAJECTORY GENERATORS
# ============================================================================

def generate_step_trajectory(step_time: float = 2.0, amplitude: float = 1.0) -> Callable:
    """Step function at specified time."""
    def u_func(t):
        return amplitude if t >= step_time else 0.0
    return u_func


def generate_sine_trajectory(frequency: float = 1.0, amplitude: float = 1.0, phase: float = 0.0) -> Callable:
    """Sinusoidal input."""
    def u_func(t):
        return amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return u_func


def generate_chirp_trajectory(f0: float = 0.1, f1: float = 5.0, amplitude: float = 1.0) -> Callable:
    """Chirp signal (frequency sweep)."""
    def u_func(t):
        return amplitude * np.sin(2 * np.pi * (f0 + (f1 - f0) * t / DURATION) * t)
    return u_func


def generate_square_wave_trajectory(frequency: float = 0.5, amplitude: float = 1.0, duty_cycle: float = 0.5) -> Callable:
    """Square wave."""
    def u_func(t):
        return amplitude * signal.square(2 * np.pi * frequency * t, duty=duty_cycle)
    return u_func


def generate_random_trajectory(seed: int = 42, amplitude: float = 1.0, cutoff_freq: float = 5.0) -> Callable:
    """Band-limited random signal."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(N_SAMPLES)
    
    # Low-pass filter
    sos = signal.butter(4, cutoff_freq, fs=SAMPLING_RATE, output='sos')
    filtered = signal.sosfilt(sos, noise)
    filtered = filtered / np.max(np.abs(filtered)) * amplitude
    
    def u_func(t):
        idx = int(t * SAMPLING_RATE)
        if idx >= len(filtered):
            idx = len(filtered) - 1
        return filtered[idx]
    return u_func


def generate_ramp_trajectory(start_time: float = 1.0, slope: float = 0.5) -> Callable:
    """Ramp (linear increase/decrease)."""
    def u_func(t):
        if t < start_time:
            return 0.0
        else:
            return slope * (t - start_time)
    return u_func


def generate_pulse_train_trajectory(frequency: float = 1.0, amplitude: float = 1.0, pulse_width: float = 0.1) -> Callable:
    """Pulse train with specified width."""
    period = 1.0 / frequency
    def u_func(t):
        t_mod = t % period
        return amplitude if t_mod < pulse_width else 0.0
    return u_func


def generate_multisine_trajectory(frequencies: list = [0.5, 1.5, 3.0], amplitudes: list = [0.5, 0.3, 0.2]) -> Callable:
    """Multi-sine (sum of sinusoids)."""
    def u_func(t):
        result = 0.0
        for f, a in zip(frequencies, amplitudes):
            result += a * np.sin(2 * np.pi * f * t)
        return result
    return u_func


# ============================================================================
# TRAJECTORY DEFINITIONS
# ============================================================================

def get_all_trajectories():
    """Define all trajectory types and their variations."""
    trajectories = []
    
    # Step functions
    for i, (step_time, amplitude) in enumerate([
        (1.0, 1.0), (2.0, 2.0), (3.0, -1.5), (1.5, 0.5), (4.0, -0.8)
    ]):
        trajectories.append(('step', i, generate_step_trajectory(step_time, amplitude)))
    
    # Sinusoidal
    for i, (freq, amp) in enumerate([
        (0.5, 1.0), (1.0, 1.5), (2.0, 0.8), (0.3, 1.2), (1.5, 1.0)
    ]):
        trajectories.append(('sine', i, generate_sine_trajectory(freq, amp)))
    
    # Chirp
    for i, (f0, f1, amp) in enumerate([
        (0.1, 5.0, 1.0), (0.5, 3.0, 1.2), (0.2, 4.0, 0.8), (1.0, 5.0, 1.0), (0.1, 2.0, 1.5)
    ]):
        trajectories.append(('chirp', i, generate_chirp_trajectory(f0, f1, amp)))
    
    # Square waves
    for i, (freq, amp, duty) in enumerate([
        (0.5, 1.0, 0.5), (1.0, 1.5, 0.3), (0.3, 1.2, 0.7), (1.5, 0.8, 0.5), (0.8, 1.0, 0.4)
    ]):
        trajectories.append(('square', i, generate_square_wave_trajectory(freq, amp, duty)))
    
    # Random
    for i in range(N_TRAJECTORIES_PER_TYPE):
        seed = 42 + i * 10
        amp = 0.8 + i * 0.1
        cutoff = 3.0 + i * 0.5
        trajectories.append(('random', i, generate_random_trajectory(seed, amp, cutoff)))
    
    # Ramps
    for i, (start, slope) in enumerate([
        (1.0, 0.5), (2.0, -0.3), (0.5, 0.8), (1.5, 0.2), (3.0, -0.5)
    ]):
        trajectories.append(('ramp', i, generate_ramp_trajectory(start, slope)))
    
    # Pulse trains
    for i, (freq, amp, width) in enumerate([
        (1.0, 1.5, 0.1), (0.5, 2.0, 0.2), (2.0, 1.0, 0.05), (1.5, 1.2, 0.15), (0.8, 1.8, 0.12)
    ]):
        trajectories.append(('pulse', i, generate_pulse_train_trajectory(freq, amp, width)))
    
    # Multi-sine
    for i, (freqs, amps) in enumerate([
        ([0.5, 1.5, 3.0], [0.5, 0.3, 0.2]),
        ([0.3, 1.0, 2.5], [0.6, 0.25, 0.15]),
        ([1.0, 2.0, 4.0], [0.4, 0.35, 0.25]),
        ([0.8, 1.8, 3.5], [0.5, 0.3, 0.2]),
        ([0.4, 1.2, 2.8], [0.55, 0.28, 0.17])
    ]):
        trajectories.append(('multisine', i, generate_multisine_trajectory(freqs, amps)))
    
    return trajectories


# ============================================================================
# DATA EXPORT
# ============================================================================

def save_trajectory_to_hdf5(filename: str, u: np.ndarray, x: np.ndarray, v: np.ndarray, 
                            x0: float, v0: float):
    """
    Save trajectory data to HDF5 file.
    
    Args:
        filename: output file path
        u: input signal
        x: position trajectory
        v: velocity trajectory
        x0: initial position
        v0: initial velocity
    """
    with h5py.File(filename, 'w') as f:
        # Save datasets
        f.create_dataset('u', data=u.astype(np.float32))
        f.create_dataset('x', data=x.astype(np.float32))
        f.create_dataset('v', data=v.astype(np.float32))
        
        # Save metadata as attributes
        f.attrs['mass'] = MASS
        f.attrs['spring_constant'] = SPRING_CONSTANT
        f.attrs['damping_coefficient'] = DAMPING_COEFFICIENT
        f.attrs['dt'] = DT
        f.attrs['sampling_rate'] = SAMPLING_RATE
        f.attrs['duration'] = DURATION
        f.attrs['n_samples'] = N_SAMPLES
        f.attrs['x0'] = x0
        f.attrs['v0'] = v0


# ============================================================================
# MAIN GENERATION LOOP
# ============================================================================

def generate_dataset():
    """Generate the complete dataset."""
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    print(f"Generating mass-spring-damper dataset")
    print(f"=" * 60)
    print(f"Physical parameters:")
    print(f"  Mass (m):              {MASS} kg")
    print(f"  Spring constant (k):   {SPRING_CONSTANT} N/m")
    print(f"  Damping coeff. (c):    {DAMPING_COEFFICIENT} N*s/m")
    print(f"\nSimulation parameters:")
    print(f"  Sampling rate:         {SAMPLING_RATE} Hz")
    print(f"  Duration:              {DURATION} s")
    print(f"  Time step:             {DT} s")
    print(f"  Number of samples:     {N_SAMPLES}")
    print(f"\nDataset parameters:")
    print(f"  Output directory:      {OUTPUT_DIR}")
    print(f"  Vary initial conds:    {VARY_INITIAL_CONDITIONS}")
    if VARY_INITIAL_CONDITIONS:
        print(f"  x0 range:              {X0_RANGE}")
        print(f"  v0 range:              {V0_RANGE}")
    print(f"=" * 60)
    print()
    
    # Get all trajectories
    trajectories = get_all_trajectories()
    
    # Generate dataset
    rng = np.random.RandomState(42)
    trajectory_count = 0
    
    for traj_type, traj_idx, u_func in trajectories:
        # Set initial conditions
        if VARY_INITIAL_CONDITIONS:
            x0 = rng.uniform(*X0_RANGE)
            v0 = rng.uniform(*V0_RANGE)
        else:
            x0, v0 = 0.0, 0.0
        
        # Simulate
        t, x, v, u = simulate_system(u_func, x0, v0)
        
        # Save to HDF5
        filename = output_path / f"trajectory_{traj_type}_{traj_idx:03d}.h5"
        save_trajectory_to_hdf5(str(filename), u, x, v, x0, v0)
        
        trajectory_count += 1
        print(f"[{trajectory_count:3d}/{len(trajectories)}] Generated {filename.name} "
              f"(x0={x0:+.3f}, v0={v0:+.3f})")
    
    print()
    print(f"=" * 60)
    print(f"Dataset generation complete!")
    print(f"Total trajectories: {trajectory_count}")
    print(f"Output directory: {output_path.absolute()}")
    print(f"=" * 60)


if __name__ == "__main__":
    generate_dataset()

