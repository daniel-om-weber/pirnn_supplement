# Mass-Spring-Damper Dataset Generator

Physics-based dataset generator for training physics-informed RNNs using the tsfast library.

## Overview

Simulates a forced mass-spring-damper system with various input trajectories and saves each simulation as a separate HDF5 file.

## System Dynamics

The mass-spring-damper system follows the equation:

```
m*ẍ + c*ẋ + k*x = u(t)
```

where:
- `m`: mass
- `c`: damping coefficient
- `k`: spring constant
- `x`: position
- `u(t)`: forcing input

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Simply run the script:

```bash
python generate_dataset.py
```

The script will generate a `dataset/` folder containing HDF5 files for each trajectory.

## Configuration

Edit the configuration section at the top of `generate_dataset.py` to customize:

- **Physical parameters**: `MASS`, `SPRING_CONSTANT`, `DAMPING_COEFFICIENT`
- **Simulation parameters**: `SAMPLING_RATE`, `DURATION`
- **Dataset parameters**: `OUTPUT_DIR`, `N_TRAJECTORIES_PER_TYPE`
- **Initial conditions**: `VARY_INITIAL_CONDITIONS`, `X0_RANGE`, `V0_RANGE`

## Trajectory Types

The dataset includes 8 types of input trajectories:

1. **Step functions**: Various amplitudes and timing
2. **Sinusoidal**: Various frequencies and amplitudes
3. **Chirp signals**: Frequency sweeps
4. **Square waves**: Various frequencies and duty cycles
5. **Random/noise**: Band-limited random inputs
6. **Ramps**: Linear increase/decrease
7. **Pulse trains**: Periodic pulses with varying widths
8. **Multi-sine**: Sum of multiple sinusoids

## Output Format

Each HDF5 file contains:

- `u`: Input force signal (float32)
- `x`: Position trajectory (float32)
- `v`: Velocity trajectory (float32)

And metadata attributes:
- `mass`, `spring_constant`, `damping_coefficient`
- `dt`, `sampling_rate`, `duration`, `n_samples`
- `x0`, `v0`: Initial conditions

## Example Usage with Python

```python
import h5py
import numpy as np

# Load a trajectory
with h5py.File('dataset/trajectory_sine_000.h5', 'r') as f:
    u = f['u'][:]
    x = f['x'][:]
    v = f['v'][:]
    
    # Access metadata
    m = f.attrs['mass']
    k = f.attrs['spring_constant']
    c = f.attrs['damping_coefficient']
    dt = f.attrs['dt']
```

