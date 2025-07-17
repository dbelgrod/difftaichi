import taichi as ti
import argparse
import os
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import time

real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True, device_memory_GB=3.5)

dim = 3
# this will be overwritten
n_particles = 0
n_solid_particles = 0
n_actuators = 0
n_grid = 32  # Reduced grid size for faster computation
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 2e-3
p_vol = 1
E = 10
# TODO: update
mu = E
la = E
max_steps = 256
steps = 128  # Shorter simulation for collision
gravity = 10
use_apic = False

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

actuator_id = ti.field(ti.i32)
particle_type = ti.field(ti.i32)
x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

loss = scalar()

# For tracking which cube each particle belongs to
cube_id = ti.field(ti.i32)

# For tracking maximum height of falling cube
max_height = scalar()
initial_drop_height = scalar()

# Optimizable parameters
drop_offset_x = scalar()  # Horizontal offset for drop
drop_offset_z = scalar()  # Depth offset for drop
initial_velocity_x = scalar()  # Initial horizontal velocity
initial_velocity_y = scalar()  # Initial downward velocity


def allocate_fields():
    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type, cube_id)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.ijk, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(loss, max_height, initial_drop_height)
    ti.root.place(drop_offset_x, drop_offset_z, initial_velocity_x, initial_velocity_y)
    
    ti.root.lazy_grad()


@ti.func
def zero_vec():
    return [0.0, 0.0, 0.0]


@ti.func
def zero_matrix():
    return [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]


@ti.kernel
def clear_grid():
    for i, j, k in grid_m_in:
        grid_v_in[i, j, k] = [0, 0, 0]
        grid_m_in[i, j, k] = 0
        grid_v_in.grad[i, j, k] = [0, 0, 0]
        grid_m_in.grad[i, j, k] = 0
        grid_v_out.grad[i, j, k] = [0, 0, 0]


@ti.kernel
def clear_particle_grad():
    # for all time steps and all particles
    for f, i in x:
        x.grad[f, i] = zero_vec()
        v.grad[f, i] = zero_vec()
        C.grad[f, i] = zero_matrix()
        F.grad[f, i] = zero_matrix()


@ti.kernel
def p2g(f: ti.i32):
    for p in range(0, n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=dim, val=1) + dt * C[f, p]) @ F[f, p]
        J = (new_F).determinant()
        if particle_type[p] == 0:  # fluid
            sqrtJ = ti.sqrt(J)
            # TODO: need pow(x, 1/3)
            new_F = ti.Matrix([[sqrtJ, 0, 0], [0, sqrtJ, 0], [0, 0, 1]])

        F[f + 1, p] = new_F

        cauchy = ti.Matrix(zero_matrix())
        mass = 0.0
        ident = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        if particle_type[p] == 0:
            mass = 4
            cauchy = ti.Matrix(ident) * (J - 1) * E
        else:
            mass = 1
            cauchy = mu * (new_F @ new_F.transpose()) + ti.Matrix(ident) * (
                la * ti.log(J) - mu)
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    offset = ti.Vector([i, j, k])
                    dpos = (ti.cast(ti.Vector([i, j, k]), real) - fx) * dx
                    weight = w[i][0] * w[j][1] * w[k][2]
                    ti.atomic_add(grid_v_in[base + offset], weight * (mass * v[f, p] + affine @ dpos))
                    ti.atomic_add(grid_m_in[base + offset], weight * mass)


bound = 3
coeff = 1.5


@ti.kernel
def grid_op():
    for i, j, k in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j, k] + 1e-10)
        v_out = inv_m * grid_v_in[i, j, k]
        v_out[1] -= dt * gravity

        if i < bound and v_out[0] < 0:
            v_out[0] = 0
            v_out[1] = 0
            v_out[2] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
            v_out[1] = 0
            v_out[2] = 0

        if k < bound and v_out[2] < 0:
            v_out[0] = 0
            v_out[1] = 0
            v_out[2] = 0
        if k > n_grid - bound and v_out[2] > 0:
            v_out[0] = 0
            v_out[1] = 0
            v_out[2] = 0

        if j < bound and v_out[1] < 0:
            v_out[0] = 0
            v_out[1] = 0
            v_out[2] = 0
            normal = ti.Vector([0.0, 1.0, 0.0])
            lsq = (normal**2).sum()
            if lsq > 0.5:
                if ti.static(coeff < 0):
                    v_out[0] = 0
                    v_out[1] = 0
                    v_out[2] = 0
                else:
                    lin = v_out.dot(normal)
                    if lin < 0:
                        vit = v_out - lin * normal
                        lit = vit.norm() + 1e-10
                        if lit + coeff * lin <= 0:
                            v_out[0] = 0
                            v_out[1] = 0
                            v_out[2] = 0
                        else:
                            v_out = (1 + coeff * lin / lit) * vit
        if j > n_grid - bound and v_out[1] > 0:
            v_out[0] = 0
            v_out[1] = 0
            v_out[2] = 0

        grid_v_out[i, j, k] = v_out


@ti.kernel
def g2p(f: ti.i32):
    for p in range(0, n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector(zero_vec())
        new_C = ti.Matrix(zero_matrix())

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


@ti.kernel
def compute_max_height():
    # Sum height of all falling cube particles
    # Avoid conditionals for autodiff
    for i in range(n_particles):
        # Use multiplication to mask particles
        is_falling_cube = (cube_id[i] == 1)
        ti.atomic_add(max_height[None], is_falling_cube * x[steps - 1, i][1])


@ti.kernel
def compute_loss():
    # Maximize rebound height = minimize negative height
    # max_height contains sum, so divide by number of falling cube particles
    # Using approximate count based on reduced density
    n_falling = ti.cast(n_particles / 2, ti.f32)
    avg_height = max_height[None] / n_falling
    rebound_ratio = avg_height / initial_drop_height[None]
    loss[None] = -rebound_ratio  # Negative because we minimize loss


def forward(total_steps=steps):
    # simulation
    for s in range(total_steps - 1):
        clear_grid()
        p2g(s)
        grid_op()
        g2p(s)
    
    # Reset max_height before computing
    max_height[None] = 0.0
    compute_max_height()
    compute_loss()
    return loss[None]


def backward():
    clear_particle_grad()
    
    compute_loss.grad()
    compute_max_height.grad()
    for s in reversed(range(steps - 1)):
        # Since we do not store the grid history (to save space), we redo p2g and grid op
        clear_grid()
        p2g(s)
        grid_op()
        
        g2p.grad(s)
        grid_op.grad()
        p2g.grad(s)


class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.v = []
        self.actuator_id = []
        self.particle_type = []
        self.cube_ids = []
        self.offset_x = 0
        self.offset_y = 0
        self.offset_z = 0

    def add_cube(self, x, y, z, size, cube_idx, initial_velocity=None):
        global n_particles
        density = 2  # Reduced density for fewer particles
        count = int(size / dx * density)
        real_d = size / count
        
        for i in range(count):
            for j in range(count):
                for k in range(count):
                    pos = [
                        x + (i + 0.5) * real_d + self.offset_x,
                        y + (j + 0.5) * real_d + self.offset_y,
                        z + (k + 0.5) * real_d + self.offset_z
                    ]
                    self.x.append(pos)
                    
                    # Set initial velocity if provided
                    if initial_velocity is not None:
                        self.v.append(initial_velocity)
                    else:
                        self.v.append([0.0, 0.0, 0.0])
                    
                    self.actuator_id.append(-1)  # No actuation
                    self.particle_type.append(1)  # Solid
                    self.cube_ids.append(cube_idx)
                    self.n_particles += 1
                    self.n_solid_particles += 1

    def set_offset(self, x, y, z):
        self.offset_x = x
        self.offset_y = y
        self.offset_z = z

    def finalize(self):
        global n_particles, n_solid_particles
        n_particles = self.n_particles
        n_solid_particles = max(self.n_solid_particles, 1)
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)


def two_cubes_collision(scene, drop_x_offset, drop_z_offset, init_vx, init_vy):
    cube_size = 0.15
    
    # Stationary cube on ground
    scene.add_cube(0.4, 0.1, 0.4, cube_size, cube_idx=0, initial_velocity=None)
    
    # Falling cube with offset and initial velocity
    drop_height = 0.6
    drop_x = 0.4 + drop_x_offset
    drop_z = 0.4 + drop_z_offset
    
    scene.add_cube(drop_x, drop_height, drop_z, cube_size, cube_idx=1, 
                   initial_velocity=[init_vx, init_vy, 0.0])
    
    return drop_height


@ti.kernel
def init(x_: ti.types.ndarray(element_dim=1), v_: ti.types.ndarray(element_dim=1),
         actuator_id_arr: ti.types.ndarray(), particle_type_arr: ti.types.ndarray(),
         cube_id_arr: ti.types.ndarray()):
    for i in range(n_particles):
        x[0, i] = x_[i]
        v[0, i] = v_[i]
        F[0, i] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        actuator_id[i] = actuator_id_arr[i]
        particle_type[i] = particle_type_arr[i]
        cube_id[i] = cube_id_arr[i]


@ti.kernel
def update_optimizable_params():
    # Update initial velocities and positions based on optimizable parameters
    for i in range(n_particles):
        if cube_id[i] == 1:  # Falling cube
            # Update position with offsets
            x[0, i][0] += drop_offset_x[None]
            x[0, i][2] += drop_offset_z[None]
            
            # Update velocity
            v[0, i][0] = initial_velocity_x[None]
            v[0, i][1] = initial_velocity_y[None]


@ti.kernel
def learn(learning_rate: ti.f32):
    # Update optimizable parameters using gradients
    drop_offset_x[None] -= learning_rate * drop_offset_x.grad[None]
    drop_offset_z[None] -= learning_rate * drop_offset_z.grad[None]
    initial_velocity_x[None] -= learning_rate * initial_velocity_x.grad[None]
    initial_velocity_y[None] -= learning_rate * initial_velocity_y.grad[None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=100)
    options = parser.parse_args()
    
    # Create base scene
    scene = Scene()
    scene.set_offset(0.0, 0.0, 0.0)
    
    # Create collision scene with initial parameters
    drop_height = two_cubes_collision(scene, 0.15, 0.0, 0.0, -1.0)
    
    scene.finalize()
    allocate_fields()
    
    # Initialize optimizable parameters AFTER fields are allocated
    drop_offset_x[None] = 0.0
    drop_offset_z[None] = 0.0
    initial_velocity_x[None] = 0.0
    initial_velocity_y[None] = -1.0  # Small initial downward velocity
    initial_drop_height[None] = drop_height
    
    # Initialize particle data
    init(np.array(scene.x, dtype=np.float32),
         np.array(scene.v, dtype=np.float32),
         np.array(scene.actuator_id, dtype=np.int32),
         np.array(scene.particle_type, dtype=np.int32),
         np.array(scene.cube_ids, dtype=np.int32))
    
    losses = []
    rebound_heights = []
    
    for iter in range(options.iters):
        t = time.time()
        
        # Update particle positions/velocities based on current parameters
        update_optimizable_params()
        
        ti.ad.clear_all_gradients()
        l = forward()
        losses.append(l)
        
        # Track rebound height
        rebound_heights.append(max_height[None])
        
        loss.grad[None] = 1
        backward()
        
        per_iter_time = time.time() - t
        print(f'i={iter}, loss={l:.4f}, rebound_height={max_height[None]:.4f}, '
              f'drop_offset=({drop_offset_x[None]:.3f}, {drop_offset_z[None]:.3f}), '
              f'init_vel=({initial_velocity_x[None]:.3f}, {initial_velocity_y[None]:.3f}), '
              f'per iter {per_iter_time:.2f}s')
        
        learning_rate = 0.1
        learn(learning_rate)
        
        if iter % 20 == 19:
            print('Writing particle data to disk...')
            # visualize
            forward()
            x_ = x.to_numpy()
            v_ = v.to_numpy()
            cube_id_ = cube_id.to_numpy()
            
            folder = f'cube_collision/iter{iter:04d}/'
            os.makedirs(folder, exist_ok=True)
            
            for s in range(7, steps, 4):
                xs, ys, zs = [], [], []
                us, vs, ws = [], [], []
                cs = []
                
                for i in range(n_particles):
                    xs.append(x_[s, i][0])
                    ys.append(x_[s, i][1])
                    zs.append(x_[s, i][2])
                    us.append(v_[s, i][0])
                    vs.append(v_[s, i][1])
                    ws.append(v_[s, i][2])
                    
                    # Color by cube
                    if cube_id_[i] == 0:
                        # Stationary cube - blue
                        r, g, b = 0.2, 0.2, 0.8
                    else:
                        # Falling cube - red
                        r, g, b = 0.8, 0.2, 0.2
                    
                    cs.append(ti.rgb_to_hex((r, g, b)))
                
                data = np.array(xs + ys + zs + us + vs + ws + cs,
                                dtype=np.float32)
                fn = f'{folder}/{s:04}.bin'
                data.tofile(open(fn, 'wb'))
                print('.', end='')
            print()
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(losses)
    ax1.set_title("Optimization of Rebound Height")
    ax1.set_ylabel("Loss (negative rebound ratio)")
    ax1.set_xlabel("Gradient Descent Iterations")
    
    ax2.plot(rebound_heights)
    ax2.axhline(y=initial_drop_height[None], color='r', linestyle='--', label='Initial drop height')
    ax2.set_title("Rebound Height Evolution")
    ax2.set_ylabel("Maximum Height")
    ax2.set_xlabel("Gradient Descent Iterations")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal optimized parameters:")
    print(f"Drop offset X: {drop_offset_x[None]:.4f}")
    print(f"Drop offset Z: {drop_offset_z[None]:.4f}")
    print(f"Initial velocity X: {initial_velocity_x[None]:.4f}")
    print(f"Initial velocity Y: {initial_velocity_y[None]:.4f}")
    print(f"Final rebound height: {max_height[None]:.4f}")
    print(f"Rebound ratio: {max_height[None]/initial_drop_height[None]:.4f}")


if __name__ == '__main__':
    main()