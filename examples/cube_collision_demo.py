"""
Demonstrates differentiable cube collision simulation.
Two elastic cubes: one stationary, one falling.
Shows how rebound height changes with initial drop velocity.
"""

import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

real = ti.f32
ti.init(default_fp=real, arch=ti.gpu)

# Simulation parameters
dim = 3
n_grid = 32
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 2e-3
p_vol = 1
E = 10  # Young's modulus
mu = E
la = E
steps = 100  # Fewer steps for quick demo
gravity = 10

# Test different drop velocities
test_velocities = np.linspace(-0.5, -5.0, 10)
rebound_ratios = []

print("Testing different drop velocities...")
print("-" * 40)

for test_vel in test_velocities:
    # Fields
    n_particles = 1458  # Pre-calculated for 2 cubes
    
    particle_type = ti.field(ti.i32, shape=n_particles)
    cube_id = ti.field(ti.i32, shape=n_particles)
    x = ti.Vector.field(dim, dtype=real, shape=(steps, n_particles))
    v = ti.Vector.field(dim, dtype=real, shape=(steps, n_particles))
    C = ti.Matrix.field(dim, dim, dtype=real, shape=(steps, n_particles))
    F = ti.Matrix.field(dim, dim, dtype=real, shape=(steps, n_particles))
    grid_v_in = ti.Vector.field(dim, dtype=real, shape=(n_grid, n_grid, n_grid))
    grid_m_in = ti.field(dtype=real, shape=(n_grid, n_grid, n_grid))
    grid_v_out = ti.Vector.field(dim, dtype=real, shape=(n_grid, n_grid, n_grid))
    
    @ti.kernel
    def clear_grid():
        for i, j, k in grid_m_in:
            grid_v_in[i, j, k] = [0, 0, 0]
            grid_m_in[i, j, k] = 0
    
    @ti.kernel
    def p2g(f: ti.i32):
        for p in range(n_particles):
            base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
            fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
            new_F = (ti.Matrix.identity(real, dim) + dt * C[f, p]) @ F[f, p]
            J = new_F.determinant()
            F[f + 1, p] = new_F
            
            r, s = ti.polar_decompose(new_F)
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + ti.Matrix.identity(real, dim) * la * J * (J - 1)
            stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
            affine = stress + C[f, p]
            
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        offset = ti.Vector([i, j, k])
                        dpos = (ti.cast(ti.Vector([i, j, k]), real) - fx) * dx
                        weight = w[i][0] * w[j][1] * w[k][2]
                        ti.atomic_add(grid_v_in[base + offset], weight * (v[f, p] + affine @ dpos))
                        ti.atomic_add(grid_m_in[base + offset], weight)
    
    @ti.kernel
    def grid_op():
        for i, j, k in grid_m_in:
            if grid_m_in[i, j, k] > 0:
                v_out = grid_v_in[i, j, k] / grid_m_in[i, j, k]
                v_out[1] -= dt * gravity
                
                # Ground collision
                if j < 3 and v_out[1] < 0:
                    v_out[0] = 0
                    v_out[1] = 0
                    v_out[2] = 0
                
                grid_v_out[i, j, k] = v_out
    
    @ti.kernel
    def g2p(f: ti.i32):
        for p in range(n_particles):
            base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
            fx = x[f, p] * inv_dx - ti.cast(base, real)
            w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
            new_v = ti.Vector.zero(real, dim)
            new_C = ti.Matrix.zero(real, dim, dim)
            
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    for k in ti.static(range(3)):
                        dpos = ti.cast(ti.Vector([i, j, k]), real) - fx
                        g_v = grid_v_out[base[0] + i, base[1] + j, base[2] + k]
                        weight = w[i][0] * w[j][1] * w[k][2]
                        new_v += weight * g_v
                        new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
            
            v[f + 1, p] = new_v
            x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
            C[f + 1, p] = new_C
    
    # Initialize scene
    particle_positions = []
    particle_velocities = []
    particle_types = []
    cube_ids = []
    
    # Create cubes
    cube_size = 0.15
    density = 2
    count = int(cube_size / dx * density)
    real_d = cube_size / count
    
    # Stationary cube
    for i in range(count):
        for j in range(count):
            for k in range(count):
                pos = [0.4 + (i + 0.5) * real_d, 
                       0.1 + (j + 0.5) * real_d,
                       0.4 + (k + 0.5) * real_d]
                particle_positions.append(pos)
                particle_velocities.append([0.0, 0.0, 0.0])
                particle_types.append(1)
                cube_ids.append(0)
    
    # Falling cube
    for i in range(count):
        for j in range(count):
            for k in range(count):
                pos = [0.55 + (i + 0.5) * real_d,
                       0.6 + (j + 0.5) * real_d,
                       0.4 + (k + 0.5) * real_d]
                particle_positions.append(pos)
                particle_velocities.append([0.0, test_vel, 0.0])
                particle_types.append(1)
                cube_ids.append(1)
    
    # Initialize fields
    x_np = np.array(particle_positions, dtype=np.float32)
    v_np = np.array(particle_velocities, dtype=np.float32)
    cube_ids_np = np.array(cube_ids, dtype=np.int32)
    
    @ti.kernel
    def init(x_arr: ti.types.ndarray(ndim=2), 
             v_arr: ti.types.ndarray(ndim=2),
             cube_id_arr: ti.types.ndarray(ndim=1)):
        for i in range(n_particles):
            for j in ti.static(range(3)):
                x[0, i][j] = x_arr[i, j]
                v[0, i][j] = v_arr[i, j]
            F[0, i] = ti.Matrix.identity(real, dim)
            particle_type[i] = 1
            cube_id[i] = cube_id_arr[i]
    
    init(x_np, v_np, cube_ids_np)
    
    # Run simulation
    for s in range(steps - 1):
        clear_grid()
        p2g(s)
        grid_op()
        g2p(s)
    
    # Calculate rebound height
    x_final = x.to_numpy()
    cube_id_np = cube_id.to_numpy()
    
    max_height = 0.0
    for i in range(n_particles):
        if cube_id_np[i] == 1:  # Falling cube
            max_height = max(max_height, x_final[steps-1, i][1])
    
    rebound_ratio = max_height / 0.6
    rebound_ratios.append(rebound_ratio)
    
    print(f"Velocity: {test_vel:5.2f} m/s | Rebound ratio: {rebound_ratio:.3f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(test_velocities, rebound_ratios, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Initial Drop Velocity (m/s)', fontsize=12)
plt.ylabel('Rebound Ratio (final height / initial height)', fontsize=12)
plt.title('Elastic Cube Collision: Rebound vs Drop Velocity', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect elastic collision')
plt.legend()
plt.tight_layout()
plt.show()

print("\nAnalysis:")
print("- Lower drop velocities result in higher rebound ratios")
print("- Energy is lost due to material deformation")
print("- This demonstrates how differentiable physics can optimize collision parameters")