"""
Interactive cube collision simulation with Taichi GGUI.
Controls:
- Space: Restart simulation
- Up/Down arrows: Adjust drop velocity
- Left/Right arrows: Adjust horizontal offset
- Q: Quit
"""

import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

# Simulation parameters
dim = 3
n_grid = 32
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 2e-3
p_vol = 1
E = 10
mu = E
la = E
max_steps = 256
gravity = 10

# Particle data
n_particles = 1458  # Pre-calculated for 2 cubes
particle_type = ti.field(ti.i32, shape=n_particles)
cube_id = ti.field(ti.i32, shape=n_particles)
x = ti.Vector.field(dim, dtype=ti.f32, shape=(max_steps, n_particles))
v = ti.Vector.field(dim, dtype=ti.f32, shape=(max_steps, n_particles))
C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(max_steps, n_particles))
F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(max_steps, n_particles))

# Grid data
grid_v_in = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
grid_m_in = ti.field(dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
grid_v_out = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))

# Rendering data
positions = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
colors = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)

# Control parameters
drop_velocity = ti.field(ti.f32, shape=())
drop_offset_x = ti.field(ti.f32, shape=())
current_step = ti.field(ti.i32, shape=())

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
        new_F = (ti.Matrix.identity(ti.f32, dim) + dt * C[f, p]) @ F[f, p]
        J = new_F.determinant()
        F[f + 1, p] = new_F
        
        r, s = ti.polar_decompose(new_F)
        cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + ti.Matrix.identity(ti.f32, dim) * la * J * (J - 1)
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + C[f, p]
        
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    offset = ti.Vector([i, j, k])
                    dpos = (ti.cast(ti.Vector([i, j, k]), ti.f32) - fx) * dx
                    weight = w[i][0] * w[j][1] * w[k][2]
                    ti.atomic_add(grid_v_in[base + offset], weight * (v[f, p] + affine @ dpos))
                    ti.atomic_add(grid_m_in[base + offset], weight)

@ti.kernel
def grid_op():
    for i, j, k in grid_m_in:
        if grid_m_in[i, j, k] > 0:
            v_out = grid_v_in[i, j, k] / grid_m_in[i, j, k]
            v_out[1] -= dt * gravity
            
            # Boundary conditions
            if i < 3 and v_out[0] < 0:
                v_out[0] = 0
            if i > n_grid - 3 and v_out[0] > 0:
                v_out[0] = 0
            if k < 3 and v_out[2] < 0:
                v_out[2] = 0
            if k > n_grid - 3 and v_out[2] > 0:
                v_out[2] = 0
            if j < 3 and v_out[1] < 0:
                v_out[0] = 0
                v_out[1] = 0
                v_out[2] = 0
            if j > n_grid - 3 and v_out[1] > 0:
                v_out[1] = 0
                
            grid_v_out[i, j, k] = v_out

@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.f32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(ti.f32, dim)
        new_C = ti.Matrix.zero(ti.f32, dim, dim)
        
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    dpos = ti.cast(ti.Vector([i, j, k]), ti.f32) - fx
                    g_v = grid_v_out[base[0] + i, base[1] + j, base[2] + k]
                    weight = w[i][0] * w[j][1] * w[k][2]
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        
        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C

@ti.kernel
def init_particles(x_arr: ti.types.ndarray(ndim=2), 
                   v_arr: ti.types.ndarray(ndim=2),
                   cube_id_arr: ti.types.ndarray(ndim=1)):
    for i in range(n_particles):
        for j in ti.static(range(3)):
            x[0, i][j] = x_arr[i, j]
            v[0, i][j] = v_arr[i, j]
        # Apply drop velocity and offset to falling cube
        if cube_id_arr[i] == 1:
            x[0, i][0] += drop_offset_x[None]
            v[0, i][1] = drop_velocity[None]
        F[0, i] = ti.Matrix.identity(ti.f32, dim)
        particle_type[i] = 1
        cube_id[i] = cube_id_arr[i]
        
        # Set colors
        if cube_id[i] == 0:
            colors[i] = [0.2, 0.3, 0.8]  # Blue for stationary
        else:
            colors[i] = [0.8, 0.3, 0.2]  # Red for falling

@ti.kernel
def update_render_data(step: ti.i32):
    for i in range(n_particles):
        positions[i] = x[step, i]

def create_scene():
    particle_positions = []
    particle_velocities = []
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
                cube_ids.append(0)
    
    # Falling cube (base position, will be offset in kernel)
    for i in range(count):
        for j in range(count):
            for k in range(count):
                pos = [0.4 + (i + 0.5) * real_d,
                       0.6 + (j + 0.5) * real_d,
                       0.4 + (k + 0.5) * real_d]
                particle_positions.append(pos)
                particle_velocities.append([0.0, 0.0, 0.0])
                cube_ids.append(1)
    
    return (np.array(particle_positions, dtype=np.float32),
            np.array(particle_velocities, dtype=np.float32),
            np.array(cube_ids, dtype=np.int32))

def simulate_step():
    step = current_step[None]
    if step < max_steps - 1:
        clear_grid()
        p2g(step)
        grid_op()
        g2p(step)
        current_step[None] = step + 1

def main():
    # Initialize scene
    x_np, v_np, cube_ids_np = create_scene()
    
    # Set initial parameters
    drop_velocity[None] = -2.0
    drop_offset_x[None] = 0.15
    current_step[None] = 0
    
    # Create window
    window = ti.ui.Window("Cube Collision Simulation", (1024, 768))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    
    # Set up camera
    camera.position(2.0, 1.5, 2.0)
    camera.lookat(0.5, 0.3, 0.5)
    camera.up(0, 1, 0)
    
    # Initialize particles
    init_particles(x_np, v_np, cube_ids_np)
    
    # Simulation control
    paused = False
    frame = 0
    
    while window.running:
        # Handle input
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.SPACE:
                # Reset simulation
                current_step[None] = 0
                init_particles(x_np, v_np, cube_ids_np)
            elif window.event.key == 'q':
                break
        
        # Continuous key handling
        if window.is_pressed(ti.ui.UP):
            drop_velocity[None] -= 0.05
            print(f"Drop velocity: {drop_velocity[None]:.2f}")
        if window.is_pressed(ti.ui.DOWN):
            drop_velocity[None] += 0.05
            print(f"Drop velocity: {drop_velocity[None]:.2f}")
        if window.is_pressed(ti.ui.LEFT):
            drop_offset_x[None] -= 0.005
            print(f"Drop offset X: {drop_offset_x[None]:.3f}")
        if window.is_pressed(ti.ui.RIGHT):
            drop_offset_x[None] += 0.005
            print(f"Drop offset X: {drop_offset_x[None]:.3f}")
        
        # Update simulation
        if not paused and current_step[None] < max_steps - 1:
            simulate_step()
        
        # Update render data
        update_render_data(current_step[None])
        
        # Render
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
        scene.particles(positions, radius=0.005, color=(0.5, 0.5, 0.5))
        
        # Add ground plane visualization
        scene.mesh(ti.Vector.field(3, dtype=ti.f32, shape=4),
                   indices=ti.field(dtype=ti.i32, shape=6))
        
        canvas.scene(scene)
        
        # Display info
        window.GUI.begin("Controls", 0.02, 0.02, 0.3, 0.2)
        window.GUI.text(f"Step: {current_step[None]}/{max_steps}")
        window.GUI.text(f"Drop velocity: {drop_velocity[None]:.2f} m/s")
        window.GUI.text(f"Drop offset X: {drop_offset_x[None]:.3f} m")
        window.GUI.text("Space: Reset | Arrows: Adjust")
        window.GUI.text("Q: Quit")
        window.GUI.end()
        
        window.show()
        frame += 1

if __name__ == '__main__':
    main()