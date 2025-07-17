import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import time

real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True)

dim = 3
n_particles = 0
n_solid_particles = 0
n_grid = 32
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 2e-3
p_vol = 1
E = 10
mu = E
la = E
steps = 128
gravity = 10

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

particle_type = ti.field(ti.i32)
cube_id = ti.field(ti.i32)
x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

loss = scalar()
x_init = vec()  # Store initial positions
v_init = vec()  # Store initial velocities

# Optimizable parameters
drop_velocity_y = scalar()

def allocate_fields():
    ti.root.dense(ti.i, n_particles).place(particle_type, cube_id)
    ti.root.dense(ti.i, n_particles).place(x_init, v_init)
    ti.root.dense(ti.k, steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.ijk, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(loss, drop_velocity_y, avg_height, n_falling)
    ti.root.lazy_grad()

@ti.func
def zero_vec():
    return [0.0, 0.0, 0.0]

@ti.func
def zero_matrix():
    return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

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
    for f, i in x:
        x.grad[f, i] = zero_vec()
        v.grad[f, i] = zero_vec()
        C.grad[f, i] = zero_matrix()
        F.grad[f, i] = zero_matrix()

@ti.kernel
def reset_particles():
    # Reset particles to initial state with current drop velocity
    for i in range(n_particles):
        x[0, i] = x_init[i]
        v[0, i] = v_init[i]
        # Update falling cube velocity
        if cube_id[i] == 1:
            v[0, i][1] = drop_velocity_y[None]
        F[0, i] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        C[0, i] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

@ti.kernel
def p2g(f: ti.i32):
    for p in range(0, n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=dim, val=1) + dt * C[f, p]) @ F[f, p]
        J = (new_F).determinant()
        F[f + 1, p] = new_F
        
        cauchy = ti.Matrix(zero_matrix())
        mass = 1
        ident = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        cauchy = mu * (new_F @ new_F.transpose()) + ti.Matrix(ident) * (la * ti.log(J) - mu)
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

@ti.kernel
def grid_op():
    for i, j, k in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j, k] + 1e-10)
        v_out = inv_m * grid_v_in[i, j, k]
        v_out[1] -= dt * gravity
        
        # Boundary conditions
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
        if k < bound and v_out[2] < 0:
            v_out[2] = 0
        if k > n_grid - bound and v_out[2] > 0:
            v_out[2] = 0
        if j < bound and v_out[1] < 0:
            v_out[0] = 0
            v_out[1] = 0
            v_out[2] = 0
        if j > n_grid - bound and v_out[1] > 0:
            v_out[1] = 0
            
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

# Split into two kernels for autodiff compatibility
avg_height = scalar()
n_falling = scalar()

@ti.kernel 
def compute_avg_height():
    # Sum heights of falling cube
    for i in range(n_particles):
        # Multiply by mask to avoid if statement
        is_falling = ti.cast(cube_id[i] == 1, ti.f32)
        avg_height[None] += is_falling * x[steps - 1, i][1]
        n_falling[None] += is_falling

@ti.kernel
def compute_loss():
    # Normalize and compute loss
    loss[None] = -avg_height[None] / (n_falling[None] * 0.6)

def forward():
    reset_particles()
    for s in range(steps - 1):
        clear_grid()
        p2g(s)
        grid_op()
        g2p(s)
    
    # Reset accumulators
    avg_height[None] = 0.0
    n_falling[None] = 0.0
    
    compute_avg_height()
    compute_loss()
    return loss[None]

def backward():
    clear_particle_grad()
    
    compute_loss.grad()
    compute_avg_height.grad()
    
    for s in reversed(range(steps - 1)):
        clear_grid()
        p2g(s)
        grid_op()
        g2p.grad(s)
        grid_op.grad()
        p2g.grad(s)
    
    reset_particles.grad()

@ti.kernel
def learn(lr: ti.f32):
    drop_velocity_y[None] -= lr * drop_velocity_y.grad[None]

class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.v = []
        self.particle_type = []
        self.cube_ids = []
        
    def add_cube(self, x, y, z, size, cube_idx, initial_velocity=None):
        global n_particles
        density = 2
        count = int(size / dx * density)
        real_d = size / count
        
        for i in range(count):
            for j in range(count):
                for k in range(count):
                    pos = [x + (i + 0.5) * real_d, y + (j + 0.5) * real_d, z + (k + 0.5) * real_d]
                    self.x.append(pos)
                    
                    if initial_velocity is not None:
                        self.v.append(initial_velocity)
                    else:
                        self.v.append([0.0, 0.0, 0.0])
                    
                    self.particle_type.append(1)
                    self.cube_ids.append(cube_idx)
                    self.n_particles += 1
                    self.n_solid_particles += 1
    
    def finalize(self):
        global n_particles, n_solid_particles
        n_particles = self.n_particles
        n_solid_particles = max(self.n_solid_particles, 1)
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)

@ti.kernel
def init(x_: ti.types.ndarray(element_dim=1), v_: ti.types.ndarray(element_dim=1),
         particle_type_arr: ti.types.ndarray(), cube_id_arr: ti.types.ndarray()):
    for i in range(n_particles):
        x_init[i] = x_[i]
        v_init[i] = v_[i]
        particle_type[i] = particle_type_arr[i]
        cube_id[i] = cube_id_arr[i]

def main():
    # Create scene
    scene = Scene()
    
    # Stationary cube on ground
    scene.add_cube(0.4, 0.1, 0.4, 0.15, cube_idx=0)
    
    # Falling cube with offset
    scene.add_cube(0.55, 0.6, 0.4, 0.15, cube_idx=1, initial_velocity=[0.0, -2.0, 0.0])
    
    scene.finalize()
    allocate_fields()
    
    # Initialize
    init(np.array(scene.x, dtype=np.float32),
         np.array(scene.v, dtype=np.float32),
         np.array(scene.particle_type, dtype=np.int32),
         np.array(scene.cube_ids, dtype=np.int32))
    
    # Initialize optimizable parameter
    drop_velocity_y[None] = -2.0
    
    losses = []
    velocities = []
    
    print("Starting optimization...")
    for iter in range(50):
        t = time.time()
        
        ti.ad.clear_all_gradients()
        l = forward()
        losses.append(l)
        velocities.append(drop_velocity_y[None])
        
        loss.grad[None] = 1
        backward()
        
        print(f'iter={iter}, loss={l:.4f}, drop_velocity={drop_velocity_y[None]:.4f}, '
              f'grad={drop_velocity_y.grad[None]:.4f}, time={time.time()-t:.2f}s')
        
        learn(0.5)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    ax1.plot(losses)
    ax1.set_title("Loss Evolution (Negative Rebound Height)")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Iterations")
    
    ax2.plot(velocities)
    ax2.set_title("Drop Velocity Evolution")
    ax2.set_ylabel("Initial Y Velocity")
    ax2.set_xlabel("Iterations")
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal optimized drop velocity: {drop_velocity_y[None]:.4f}")

if __name__ == '__main__':
    main()