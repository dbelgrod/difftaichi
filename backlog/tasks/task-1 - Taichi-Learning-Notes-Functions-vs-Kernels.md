---
id: task-1
title: Taichi Learning Notes - Functions vs Kernels
status: To Do
assignee: []
created_date: '2025-07-17'
updated_date: '2025-07-17'
labels: []
dependencies: []
---

## Description

Personal learning notes to understand the key differences between Taichi functions and kernels, and how they relate to CUDA programming concepts

## Acceptance Criteria

- [ ] Document function vs kernel differences
- [ ] Note GPU execution patterns
- [ ] Compare with CUDA programming model

## Implementation Notes

### Key differences between @ti.func and @ti.kernel:

**Official Taichi Documentation Note:**
For users familiar with CUDA programming, the ti.func in Taichi is equivalent to the __device__ in CUDA, while the ti.kernel in Taichi corresponds to the __global__ in CUDA.

1. **Kernels (@ti.kernel)**:
   - Entry points for parallel computation (like CUDA kernels)
   - Called from Python host code
   - Outermost loops automatically parallelized
   - Cannot be called from other kernels or functions
   - Equivalent to __global__ functions in CUDA

2. **Functions (@ti.func)**:
   - Helper functions that run on GPU/device
   - Can only be called from kernels or other functions
   - Cannot be called directly from Python
   - Equivalent to __device__ functions in CUDA
   - Cannot call kernels (one-way relationship)

3. **Execution Model**:
   - Python → Kernel → Function (allowed)
   - Function → Kernel (NOT allowed)
   - Function → Function (allowed)
   - Kernel → Kernel (NOT allowed)

This mirrors CUDA's model where device functions can't launch kernels, maintaining a clear host-to-device execution flow.

---

### Metal vs Vulkan on macOS:

1. **Metal**:
   - Apple's proprietary graphics API (like DirectX for Windows)
   - Native to macOS/iOS, deeply integrated with Apple hardware
   - Better performance on Apple devices due to direct hardware optimization
   - Required for Apple-specific features (ProRes, Neural Engine, etc.)
   - Fully supported and actively developed by Apple

2. **Vulkan**:
   - Cross-platform graphics API (works on Windows, Linux, Android, etc.)
   - NOT natively supported on macOS - Apple deprecated OpenGL/Vulkan support
   - Can run via MoltenVK (translates Vulkan → Metal), but with overhead
   - Better for cross-platform development
   - Apple chose not to support it to push Metal adoption

**Unity PolySpatial:**
- Specifically designed for Apple Vision Pro
- Uses RealityKit as the rendering backend (built on Metal)
- Converts Unity content to RealityKit's native format
- Does NOT use traditional Unity rendering pipelines
- Requires Metal/RealityKit for spatial computing features
- No Vulkan option for Apple Vision Pro development

---

### Damping in Physics Simulations:

Damping is a force that opposes motion and removes energy from the system - like friction or resistance that slows things down.

In cloth simulation, there are two types of damping:

1. **Drag damping** (e.g., drag_damping = 1):
   - Air resistance that slows down the entire cloth
   - Applied as: v[i] *= ti.exp(-drag_damping * dt)
   - Reduces all velocities exponentially
   - Without it, the cloth would oscillate forever

2. **Dashpot damping** (e.g., dashpot_damping = 1e4):
   - Internal friction between connected points
   - Acts like shock absorbers between mass points
   - Prevents springs from bouncing endlessly
   - Applied to spring connections

Real-world analogy:
- No damping = Pendulum in vacuum (swings forever)
- With damping = Pendulum in air (gradually stops)

The exponential formula ti.exp(-drag_damping * dt) ensures physical accuracy - it models how real drag forces work, where resistance is proportional to velocity.

---

### Material Point Method (MPM) Flow Diagram

#### Overview
MPM is a hybrid particle-grid method that combines the advantages of both Lagrangian (particle) and Eulerian (grid) approaches for simulating deformable materials.

#### Core MPM Cycle (One Timestep)

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

#### Differentiable Simulation Flow

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

#### Key Concepts Explained

1. **Particles vs Grid**
   - **Particles**: Store the actual material (position, velocity, deformation)
   - **Grid**: Temporary computational mesh for solving physics equations
   - Think of it like: particles are the "real" material, grid is just for calculations

2. **Why Transfer Between Them?**
   - Particles can move anywhere (good for large deformations)
   - Grid provides regular structure (good for computing forces/collisions)
   - MPM combines both advantages

3. **The Transfer Process**
   - Each particle "spreads" its properties to nearby grid nodes
   - Grid nodes collect contributions from all nearby particles
   - After grid update, particles "gather" new velocities from grid

4. **Neural Network Control**
   - The `weights` and `bias` form a simple neural network
   - Input: time-based sinusoidal features
   - Output: actuation forces for robot muscles
   - Optimization finds weights that make robot move rightward

5. **Differentiable Simulation**
   - Every operation can compute gradients
   - Allows backpropagation through entire physics simulation
   - Enables learning controllers via gradient descent (not RL)

#### Data Flow Summary

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

#### Function/Kernel connections in elastic_cube.py:

**The Core MPM Loop (each timestep):**
1. **clear_grid()** - Resets the background grid
2. **compute_actuation()** - Neural network generates muscle forces
3. **p2g()** - Particles transfer their mass/momentum to grid
4. **grid_op()** - Apply physics (gravity, collisions) on grid
5. **g2p()** - Grid transfers updated velocities back to particles

**The Learning Process:**
- **forward()** - Runs simulation for 512 timesteps
- **compute_loss()** - Measures how far robot moved (wants rightward motion)
- **backward()** - Computes gradients through entire simulation
- **learn()** - Updates neural network weights

The key insight of MPM is that it uses both particles (which can move freely) and a grid (which makes physics calculations easier). The particles are the "real" material, while the grid is just temporary scratch space for computations.