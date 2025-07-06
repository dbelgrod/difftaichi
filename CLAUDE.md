# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DiffTaichi is a differentiable programming framework for physical simulation. It's a collection of example simulators that demonstrate how to use the Taichi programming language for differentiable physics simulations. The project enables gradient-based optimization of physical systems, allowing neural network controllers to be trained via gradient descent rather than reinforcement learning.

## Key Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run any example simulator
python3 examples/<simulator_name>.py

# Common simulators:
python3 examples/diffmpm.py      # Elastic object simulation
python3 examples/liquid.py       # 3D fluid simulation
python3 examples/billiards.py    # Billiard ball simulation
python3 examples/rigid_body.py   # Rigid body simulation
```

## Architecture Overview

### Core Pattern
All simulators follow this structure:
1. **Field Allocation**: Use `ti.field()` to allocate memory for simulation state (positions, velocities, grids)
2. **Kernels**: Define physics computations with `@ti.kernel` decorator
3. **Forward Simulation**: Implement timestep updates
4. **Loss Function**: Define optimization objectives
5. **Backward Pass**: Taichi automatically computes gradients

### Key Technical Details
- **Language**: Python 3 with Taichi kernels
- **GPU Acceleration**: Most examples use `ti.cuda` backend
- **Differentiation**: Automatic via Taichi's AD system
- **State Storage**: Taichi fields store particle/grid data
- **Visualization**: Uses matplotlib or cv2 for rendering

### Common Code Patterns

```python
# Field allocation pattern
x = ti.Vector.field(2, dtype=ti.f32, shape=n_particles, needs_grad=True)
v = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)

# Kernel pattern
@ti.kernel
def substep():
    for i in x:
        # Physics computation
        v[i] += dt * force
        x[i] += dt * v[i]

# Loss computation pattern
@ti.kernel
def compute_loss():
    for i in range(n_particles):
        loss[None] += (x[i] - target[i]).norm()

# Optimization loop
with ti.ad.Tape(loss=loss):
    for s in range(steps):
        substep()
    compute_loss()
```

## Important Notes

- **No Test Suite**: This is a research example repository without formal tests
- **Standalone Examples**: Each simulator is self-contained
- **Taichi Version**: May require specific Taichi versions for compatibility
- **GPU Memory**: Large simulations may require significant GPU memory
- **Gradient Computation**: Memory usage scales with simulation length due to AD tape

## Common Issues

1. **CUDA Errors**: Ensure CUDA toolkit is installed for GPU examples
2. **Memory Issues**: Reduce particle count or simulation steps if OOM
3. **Import Errors**: Install Taichi with `pip install taichi`
4. **Performance**: Use `ti.cuda` backend for best performance