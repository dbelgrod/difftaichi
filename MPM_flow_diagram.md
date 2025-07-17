# Material Point Method (MPM) Flow Diagram

## Overview
MPM is a hybrid particle-grid method that combines the advantages of both Lagrangian (particle) and Eulerian (grid) approaches for simulating deformable materials.

## Core MPM Cycle (One Timestep)

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INITIALIZATION                              │
│  • Particles store: position (x), velocity (v), deformation (F,C)   │
│  • Background grid: temporary storage for forces and mass           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     1. CLEAR GRID (clear_grid)                      │
│  • Reset grid velocities to 0                                       │
│  • Reset grid masses to 0                                           │
│  • Grid is just temporary scratch space                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│              2. COMPUTE ACTUATION (compute_actuation)               │
│  • Neural network generates forces for actuators                    │
│  • Uses sinusoidal basis: Σ weights[i,j] * sin(ωt + phase)         │
│  • Output: actuation forces for this timestep                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│               3. PARTICLE TO GRID (p2g)                             │
│  • Transfer particle properties to grid nodes                       │
│  • Each particle affects 3x3x3 = 27 nearby grid nodes              │
│  • Transfer: mass, momentum, stress forces                          │
│  • Uses B-spline interpolation weights                              │
│                                                                      │
│    Particle ●  →→→  Grid nodes ▫▫▫                                 │
│              ╲  │  ╱            ▫▫▫                                 │
│               ╲ │ ╱             ▫▫▫                                 │
│                ╲│╱                                                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    4. GRID OPERATIONS (grid_op)                     │
│  • Update grid velocities: v = momentum / mass                      │
│  • Apply external forces (gravity): v.y -= dt * g                   │
│  • Apply boundary conditions (walls, floor)                         │
│  • Collision detection and response                                 │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    5. GRID TO PARTICLE (g2p)                        │
│  • Transfer updated velocities back to particles                    │
│  • Update particle positions: x += dt * v                           │
│  • Update velocity gradient (C matrix) for APIC                     │
│  • Each particle samples from same 27 grid nodes                    │
│                                                                      │
│    Grid nodes ▫▫▫  →→→  Particle ●                                 │
│               ▫▫▫  │  ╱           ╲                                 │
│               ▫▫▫  │ ╱             ╲ (new position)                 │
│                    │╱                                                │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                          [REPEAT FOR NEXT TIMESTEP]
```

## Differentiable Simulation Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FORWARD SIMULATION                            │
│                                                                      │
│  main() ──► forward() ──► for each timestep:                       │
│                           1. clear_grid()                            │
│                           2. compute_actuation(t)                    │
│                           3. p2g(t)                                  │
│                           4. grid_op()                               │
│                           5. g2p(t)                                  │
│                           └─► Update particle states                 │
│                                                                      │
│             After all timesteps:                                     │
│             └─► compute_x_avg() ──► compute_loss()                 │
│                 (average position)   (distance objective)            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BACKWARD PROPAGATION                            │
│                                                                      │
│  main() ──► backward() ──► for each timestep (REVERSE):            │
│                            1. Re-run p2g(t) and grid_op()           │
│                               (need grid state for gradients)        │
│                            2. g2p.grad(t)                            │
│                            3. grid_op.grad()                         │
│                            4. p2g.grad(t)                            │
│                            5. compute_actuation.grad(t)              │
│                            └─► Accumulate gradients                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        OPTIMIZATION STEP                             │
│                                                                      │
│  main() ──► learn() ──► Update neural network:                     │
│                         • weights -= learning_rate * weights.grad    │
│                         • bias -= learning_rate * bias.grad          │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Concepts Explained

### 1. **Particles vs Grid**
- **Particles**: Store the actual material (position, velocity, deformation)
- **Grid**: Temporary computational mesh for solving physics equations
- Think of it like: particles are the "real" material, grid is just for calculations

### 2. **Why Transfer Between Them?**
- Particles can move anywhere (good for large deformations)
- Grid provides regular structure (good for computing forces/collisions)
- MPM combines both advantages

### 3. **The Transfer Process**
- Each particle "spreads" its properties to nearby grid nodes
- Grid nodes collect contributions from all nearby particles
- After grid update, particles "gather" new velocities from grid

### 4. **Neural Network Control**
- The `weights` and `bias` form a simple neural network
- Input: time-based sinusoidal features
- Output: actuation forces for robot muscles
- Optimization finds weights that make robot move rightward

### 5. **Differentiable Simulation**
- Every operation can compute gradients
- Allows backpropagation through entire physics simulation
- Enables learning controllers via gradient descent (not RL)

## Data Flow Summary

```
Neural Network ──► Actuation Forces
                         │
                         ▼
Particles ──► Grid ──► Physics Update ──► Grid ──► Particles
   │                                                    │
   └────────────────── One Timestep ───────────────────┘
                         │
                    (Repeat 512x)
                         │
                         ▼
                   Final Position ──► Loss
                         │
                   (Backpropagate)
                         │
                         ▼
                Update Neural Network
```

This creates a differentiable physics simulator where a neural network learns to control a soft robot to achieve objectives!