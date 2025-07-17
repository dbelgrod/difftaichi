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

Key differences between @ti.func and @ti.kernel:

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

Key differences between @ti.func and @ti.kernel:

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

**Metal vs Vulkan on macOS:**

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

Key differences between @ti.func and @ti.kernel:

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

**Metal vs Vulkan on macOS:**

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

**Damping in Physics Simulations:**

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
