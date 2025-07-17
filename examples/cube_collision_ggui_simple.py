"""
Simple cube collision simulation with Taichi GGUI.
Reduced particle count for better performance.
Controls:
- Space: Restart simulation
- Up/Down arrows: Adjust drop velocity
- Q: Quit
"""

import taichi as ti
import numpy as np

# Use CPU for better stability, change to ti.gpu if you have a good GPU
ti.init(arch=ti.cpu)

# Simulation parameters
dim = 3
n_grid = 24  # Reduced grid size
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 2e-3
p_vol = 1
E = 10
mu = E
la = E
max_steps = 1000  # Extended simulation to see multiple bounces
gravity = 10

# Particle data - will be set after scene creation
n_particles = 0
particle_type = None
cube_id = None
x = None
v = None
C = None
F = None

# Grid data
grid_v_in = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
grid_m_in = ti.field(dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
grid_v_out = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))

# Control parameters
drop_velocity = ti.field(ti.f32, shape=())
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
                v_out = [0, 0, 0]  # Full stop on ground
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
        # Apply drop velocity to falling cube
        if cube_id_arr[i] == 1:
            v[0, i][1] = drop_velocity[None]
        F[0, i] = ti.Matrix.identity(ti.f32, dim)
        particle_type[i] = 1
        cube_id[i] = cube_id_arr[i]

@ti.kernel
def update_positions(positions: ti.template(), step: ti.i32):
    for i in range(n_particles):
        positions[i] = x[step, i]

def create_scene():
    particle_positions = []
    particle_velocities = []
    cube_ids = []
    
    # Create cubes with 5x5x5 particles
    cube_size = 0.15
    count = 5  # Exactly 5x5x5 particles per cube
    real_d = cube_size / count
    
    # Stationary cube centered at x=0.4
    for i in range(count):
        for j in range(count):
            for k in range(count):
                pos = [0.4 + (i + 0.5) * real_d, 
                       0.1 + (j + 0.5) * real_d,
                       0.4 + (k + 0.5) * real_d]
                particle_positions.append(pos)
                particle_velocities.append([0.0, 0.0, 0.0])
                cube_ids.append(0)
    
    # Falling cube - offset by 10% of cube size for 90% overlap
    # 10% of 0.15 = 0.015, so offset by 0.015
    x_offset = 0.015
    for i in range(count):
        for j in range(count):
            for k in range(count):
                pos = [0.4 + x_offset + (i + 0.5) * real_d,
                       0.5 + (j + 0.5) * real_d,
                       0.4 + (k + 0.5) * real_d]
                particle_positions.append(pos)
                particle_velocities.append([0.0, 0.0, 0.0])
                cube_ids.append(1)
    
    return (np.array(particle_positions, dtype=np.float32),
            np.array(particle_velocities, dtype=np.float32),
            np.array(cube_ids, dtype=np.int32),
            len(particle_positions))

def simulate_step():
    step = current_step[None]
    if step < max_steps - 1:
        clear_grid()
        p2g(step)
        grid_op()
        g2p(step)
        current_step[None] = step + 1
    else:
        # Auto-restart when simulation ends
        current_step[None] = 0

def main():
    global n_particles, particle_type, cube_id, x, v, C, F
    
    # Create scene and get particle count
    x_np, v_np, cube_ids_np, n_particles = create_scene()
    print(f"Running with {n_particles} particles")
    
    # Now allocate fields with correct size
    particle_type = ti.field(ti.i32, shape=n_particles)
    cube_id = ti.field(ti.i32, shape=n_particles)
    x = ti.Vector.field(dim, dtype=ti.f32, shape=(max_steps, n_particles))
    v = ti.Vector.field(dim, dtype=ti.f32, shape=(max_steps, n_particles))
    C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(max_steps, n_particles))
    F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(max_steps, n_particles))
    
    # Set initial parameters
    drop_velocity[None] = -2.0
    current_step[None] = 0
    
    # Create window
    window = ti.ui.Window("Simple Cube Collision", (800, 600))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    
    # Set up camera - adjusted for larger view
    camera.position(2.0, 1.5, 2.0)
    camera.lookat(0.5, 0.4, 0.5)
    camera.up(0, 1, 0)
    
    # Initialize particles
    init_particles(x_np, v_np, cube_ids_np)
    
    # Position and color fields for rendering
    positions = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
    colors = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
    
    # Set colors
    @ti.kernel
    def set_colors():
        for i in range(n_particles):
            if cube_id[i] == 0:
                colors[i] = [0.2, 0.3, 0.8]  # Blue
            else:
                colors[i] = [0.8, 0.3, 0.2]  # Red
    
    set_colors()
    
    while window.running:
        # Handle input
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.SPACE:
                # Reset simulation
                current_step[None] = 0
                init_particles(x_np, v_np, cube_ids_np)
                print("Simulation reset")
            elif window.event.key == 'q':
                break
        
        # Continuous key handling
        if window.is_pressed(ti.ui.UP):
            drop_velocity[None] -= 0.1
            print(f"Drop velocity: {drop_velocity[None]:.2f}")
        if window.is_pressed(ti.ui.DOWN):
            drop_velocity[None] += 0.1
            print(f"Drop velocity: {drop_velocity[None]:.2f}")
        
        # Update simulation
        simulate_step()
        
        # Check if we need to reinitialize at loop restart
        if current_step[None] == 0:
            init_particles(x_np, v_np, cube_ids_np)
        
        # Update positions for rendering
        update_positions(positions, current_step[None])
        
        # Render
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        
        scene.ambient_light((0.6, 0.6, 0.6))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
        
        # Draw particles with colors
        scene.particles(positions, radius=0.01, per_vertex_color=colors)
        
        canvas.scene(scene)
        
        # Display info
        window.GUI.begin("Info", 0.02, 0.02, 0.35, 0.18)
        window.GUI.text(f"Step: {current_step[None]}/{max_steps}")
        window.GUI.text(f"Particles: {n_particles}")
        window.GUI.text(f"Drop velocity: {drop_velocity[None]:.2f} m/s")
        window.GUI.text("Auto-looping simulation")
        window.GUI.text("Space: Reset | Up/Down: Velocity | Q: Quit")
        window.GUI.end()
        
        window.show()

if __name__ == '__main__':
    main()