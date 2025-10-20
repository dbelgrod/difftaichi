"""
3D Tree wind simulation using Taichi MPM.
A tree with stiff trunk, bendable branches, and soft leaves that sway in the wind.
Controls:
- Space: Restart simulation
- Up/Down arrows: Adjust drop velocity (legacy, kept for compatibility)
- Left/Right arrows: Adjust wind strength
- Q: Quit
"""

import taichi as ti
import numpy as np
import math
from tree_controls import MaterialPropertyController, RecordingManager

# Use CPU for better stability, change to ti.gpu if you have a good GPU
ti.init(arch=ti.cpu)

# Simulation parameters
dim = 3
n_grid = 28  # Slightly increased grid size
dx = 1 / n_grid
inv_dx = 1 / dx
base_dt = 5e-4  # Much smaller base time step for stability
dt = ti.field(ti.f32, shape=())  # Adaptive time step
p_vol = 1
buffer_size = 1000  # Circular buffer size for history
# gravity and damping_factor are now controlled by MaterialPropertyController
max_velocity = 5.0  # Lower velocity clamping threshold

# Actual step counter (can grow infinitely)
actual_step_count = ti.field(ti.i32, shape=())

# Material properties are now controlled by MaterialPropertyController
# This allows real-time adjustment via GUI sliders
material_controller = None  # Will be initialized in main()

# Stiffness ramping for stability
stiffness_ramp_factor = ti.field(ti.f32, shape=())
ramp_steps = 100  # Steps to reach full stiffness

# Particle data - will be set after scene creation
n_particles = 0
particle_type = None
part_id = None  # 0=trunk, 1=branches, 2=leaves
x = None
v = None
C = None
F = None

# Grid data
grid_v_in = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
grid_m_in = ti.field(dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
grid_v_out = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))

# Control parameters
drop_velocity = ti.field(ti.f32, shape=())  # Legacy, kept for compatibility
# wind_strength is now in material_controller

@ti.kernel
def clear_grid():
    for i, j, k in grid_m_in:
        grid_v_in[i, j, k] = [0, 0, 0]
        grid_m_in[i, j, k] = 0

@ti.kernel
def p2g(f: ti.i32, f_next: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.identity(ti.f32, dim) + dt[None] * C[f, p]) @ F[f, p]
        J = new_F.determinant()
        # Clamp Jacobian to prevent extreme deformations
        if J < 0.4:
            new_F = new_F * ti.pow(0.4 / J, 1.0/3.0)
        elif J > 2.5:
            new_F = new_F * ti.pow(2.5 / J, 1.0/3.0)
        F[f_next, p] = new_F
        
        r, s = ti.polar_decompose(new_F)
        
        # Different material properties based on part type
        # Read from controller for real-time adjustment
        E = material_controller.E_wood[None] * stiffness_ramp_factor[None]
        mu = material_controller.mu_wood[None] * stiffness_ramp_factor[None]
        la = material_controller.la_wood[None] * stiffness_ramp_factor[None]
        if part_id[p] == 2:  # Leaves
            E = material_controller.E_leaf[None]
            mu = material_controller.mu_leaf[None]
            la = material_controller.la_leaf[None]
        elif part_id[p] == 1:  # Branches (almost as stiff as trunk)
            E = material_controller.E_wood[None] * 0.9 * stiffness_ramp_factor[None]
            mu = material_controller.mu_wood[None] * 0.9 * stiffness_ramp_factor[None]
            la = material_controller.la_wood[None] * 0.9 * stiffness_ramp_factor[None]
        
        cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + ti.Matrix.identity(ti.f32, dim) * la * J * (J - 1)
        stress = -(dt[None] * p_vol * 4 * inv_dx * inv_dx) * cauchy
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
            v_out[1] -= dt[None] * material_controller.gravity[None]

            # Add wind force - oscillating with time
            # Wind primarily in x direction, with some variation
            wind_factor = ti.sin(ti.cast(actual_step_count[None], ti.f32) * 0.01)  # Smooth oscillation
            wind_x = material_controller.wind_strength[None] * wind_factor
            
            # Wind affects higher parts more (based on y coordinate)
            # And affects leaves more than branches, branches more than trunk
            height_factor = ti.cast(j, ti.f32) / ti.cast(n_grid, ti.f32)  # 0 at bottom, 1 at top
            
            # Apply wind force (increased effect)
            v_out[0] += dt[None] * wind_x * height_factor * 5.0

            # Apply numerical damping
            v_out *= material_controller.damping_factor[None]
            
            # Velocity clamping to prevent explosions
            v_mag = v_out.norm()
            if v_mag > max_velocity:
                v_out = v_out * (max_velocity / v_mag)
            
            # Boundary conditions
            if i < 3 and v_out[0] < 0:
                v_out[0] = 0
            if i > n_grid - 3 and v_out[0] > 0:
                v_out[0] = 0
            if k < 3 and v_out[2] < 0:
                v_out[2] = 0
            if k > n_grid - 3 and v_out[2] > 0:
                v_out[2] = 0
            
            # Ground boundary - anchor trunk base
            if j < 3 and v_out[1] < 0:
                v_out = [0, 0, 0]  # Full stop on ground
            if j > n_grid - 3 and v_out[1] > 0:
                v_out[1] = 0
                
            grid_v_out[i, j, k] = v_out

@ti.kernel
def g2p(f: ti.i32, f_next: ti.i32):
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

        # Additional wind effect directly on particles (especially leaves)
        if part_id[p] == 2:  # Leaves get extra wind
            wind_factor = ti.sin(ti.cast(actual_step_count[None], ti.f32) * 0.01)
            new_v[0] += material_controller.wind_strength[None] * wind_factor * 0.5

        # Velocity clamping per particle
        v_mag = new_v.norm()
        if v_mag > max_velocity:
            new_v = new_v * (max_velocity / v_mag)

        # Anchor trunk base particles
        if part_id[p] == 0 and x[f, p][1] < 0.1:  # Trunk particles near ground
            new_v = ti.Vector([0, 0, 0])

        v[f_next, p] = new_v
        x[f_next, p] = x[f, p] + dt[None] * v[f_next, p]
        C[f_next, p] = new_C

@ti.kernel
def init_particles(x_arr: ti.types.ndarray(ndim=2),
                   v_arr: ti.types.ndarray(ndim=2),
                   part_id_arr: ti.types.ndarray(ndim=1)):
    for i in range(n_particles):
        for j in ti.static(range(3)):
            x[0, i][j] = x_arr[i, j]
            v[0, i][j] = v_arr[i, j]
        F[0, i] = ti.Matrix.identity(ti.f32, dim)
        C[0, i] = ti.Matrix.zero(ti.f32, dim, dim)  # Initialize affine velocity to zero
        particle_type[i] = 1
        part_id[i] = part_id_arr[i]

@ti.kernel
def update_positions(positions: ti.template(), step: ti.i32):
    for i in range(n_particles):
        positions[i] = x[step, i]

def create_scene():
    particle_positions = []
    particle_velocities = []
    part_ids = []
    
    # Tree structure parameters
    trunk_width = 0.1
    trunk_height = 0.5
    trunk_depth = 0.1
    
    # Trunk - tall rectangular box
    # Base at y=0.05, centered at x=0.5, z=0.5
    trunk_particles_x = 5
    trunk_particles_y = 20
    trunk_particles_z = 5
    
    dx_trunk = trunk_width / trunk_particles_x
    dy_trunk = trunk_height / trunk_particles_y
    dz_trunk = trunk_depth / trunk_particles_z
    
    # Create trunk particles
    for i in range(trunk_particles_x):
        for j in range(trunk_particles_y):
            for k in range(trunk_particles_z):
                pos = [0.5 - trunk_width/2 + (i + 0.5) * dx_trunk,
                       0.05 + (j + 0.5) * dy_trunk,
                       0.5 - trunk_depth/2 + (k + 0.5) * dz_trunk]
                particle_positions.append(pos)
                particle_velocities.append([0.0, 0.0, 0.0])
                part_ids.append(0)  # Trunk
    
    # Branches - 4 branches at different heights and angles
    branch_width = 0.05
    branch_length = 0.2
    branch_depth = 0.05
    branch_particles = 3  # particles per dimension for branches
    
    # Branch positions and angles
    branches = [
        # (height_on_trunk, angle_from_trunk, side)
        (0.3, 45, 'left'),
        (0.35, 30, 'right'),
        (0.4, 40, 'front'),
        (0.45, 35, 'back'),
    ]
    
    for branch_height, angle, side in branches:
        # Calculate branch start position on trunk
        start_x = 0.5
        start_y = 0.05 + branch_height
        start_z = 0.5
        
        # Adjust starting position based on side
        if side == 'left':
            start_x -= trunk_width/2
            dir_x = -math.cos(math.radians(angle))
            dir_z = 0
        elif side == 'right':
            start_x += trunk_width/2
            dir_x = math.cos(math.radians(angle))
            dir_z = 0
        elif side == 'front':
            start_z -= trunk_depth/2
            dir_x = 0
            dir_z = -math.cos(math.radians(angle))
        else:  # back
            start_z += trunk_depth/2
            dir_x = 0
            dir_z = math.cos(math.radians(angle))
        
        dir_y = math.sin(math.radians(angle))
        
        # Create branch particles
        for i in range(branch_particles):
            for j in range(branch_particles):
                for k in range(int(branch_particles * 4)):  # Longer in branch direction
                    # Position along branch
                    t = k / (branch_particles * 4 - 1)
                    pos = [start_x + dir_x * branch_length * t + (i - branch_particles/2) * branch_width/branch_particles,
                           start_y + dir_y * branch_length * t + (j - branch_particles/2) * branch_depth/branch_particles,
                           start_z + dir_z * branch_length * t + (j - branch_particles/2) * branch_depth/branch_particles]
                    particle_positions.append(pos)
                    particle_velocities.append([0.0, 0.0, 0.0])
                    part_ids.append(1)  # Branch
        
        # Add leaves at branch ends
        # 3 leaf clusters per branch
        leaf_radius = 0.03
        leaf_particles_per_cluster = 25
        
        for leaf_idx in range(3):
            # Position leaves around branch end
            leaf_angle = leaf_idx * 120  # Spread leaves around branch
            leaf_offset_x = math.cos(math.radians(leaf_angle)) * leaf_radius
            leaf_offset_z = math.sin(math.radians(leaf_angle)) * leaf_radius
            
            center_x = start_x + dir_x * branch_length + leaf_offset_x
            center_y = start_y + dir_y * branch_length + (leaf_idx - 1) * 0.02
            center_z = start_z + dir_z * branch_length + leaf_offset_z
            
            # Create spherical cluster of leaf particles
            for _ in range(leaf_particles_per_cluster):
                # Random position within sphere
                theta = np.random.random() * 2 * np.pi
                phi = np.random.random() * np.pi
                r = np.random.random() ** (1/3) * leaf_radius
                
                pos = [center_x + r * np.sin(phi) * np.cos(theta),
                       center_y + r * np.cos(phi),
                       center_z + r * np.sin(phi) * np.sin(theta)]
                particle_positions.append(pos)
                particle_velocities.append([0.0, 0.0, 0.0])
                part_ids.append(2)  # Leaf
    
    return (np.array(particle_positions, dtype=np.float32),
            np.array(particle_velocities, dtype=np.float32),
            np.array(part_ids, dtype=np.int32),
            len(particle_positions))

@ti.kernel
def calculate_adaptive_timestep():
    """Calculate timestep based on CFL condition for current stiffness"""
    # Wave speed c = sqrt(E/rho), assuming rho=1000 kg/m^3
    max_E = material_controller.E_wood[None] * stiffness_ramp_factor[None]
    wave_speed = ti.sqrt(max_E / 1000.0)
    # CFL condition: dt < dx / c * safety_factor
    safety_factor = 0.2  # More conservative
    max_dt = dx / wave_speed * safety_factor
    # Clamp to reasonable range
    dt[None] = ti.min(max_dt, base_dt)

def update_stiffness_ramp():
    """Gradually increase stiffness to avoid sudden shocks"""
    if actual_step_count[None] < ramp_steps:
        stiffness_ramp_factor[None] = (actual_step_count[None] + 1) / ramp_steps
    else:
        stiffness_ramp_factor[None] = 1.0

def simulate_step():
    """Advance simulation by one step using circular buffer"""
    step = actual_step_count[None]
    buffer_idx = step % buffer_size
    buffer_idx_next = (step + 1) % buffer_size

    update_stiffness_ramp()
    calculate_adaptive_timestep()
    clear_grid()
    p2g(buffer_idx, buffer_idx_next)
    grid_op()
    g2p(buffer_idx, buffer_idx_next)
    actual_step_count[None] = step + 1

def main():
    global n_particles, particle_type, part_id, x, v, C, F, material_controller

    # Initialize material property controller
    material_controller = MaterialPropertyController()
    material_controller.init_controls("normal")  # Start with normal preset

    # Initialize recording manager
    recording_manager = RecordingManager()

    # Create scene and get particle count
    x_np, v_np, part_ids_np, n_particles = create_scene()
    print(f"Running tree simulation with {n_particles} particles")
    
    # Now allocate fields with circular buffer
    particle_type = ti.field(ti.i32, shape=n_particles)
    part_id = ti.field(ti.i32, shape=n_particles)
    x = ti.Vector.field(dim, dtype=ti.f32, shape=(buffer_size, n_particles))
    v = ti.Vector.field(dim, dtype=ti.f32, shape=(buffer_size, n_particles))
    C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(buffer_size, n_particles))
    F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(buffer_size, n_particles))

    # Set initial parameters
    drop_velocity[None] = -2.0  # Legacy parameter
    actual_step_count[None] = 0
    stiffness_ramp_factor[None] = 0.01  # Start with very low stiffness
    dt[None] = base_dt  # Initialize timestep
    
    # Create window
    window = ti.ui.Window("Tree Wind Simulation", (800, 600))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    
    # Set up camera - adjusted for tree view
    camera.position(2.0, 1.5, 2.0)
    camera.lookat(0.5, 0.4, 0.5)
    camera.up(0, 1, 0)
    
    # Initialize particles
    init_particles(x_np, v_np, part_ids_np)
    
    # Position and color fields for rendering
    positions = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
    colors = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
    
    # Set colors based on part type
    @ti.kernel
    def set_colors():
        for i in range(n_particles):
            if part_id[i] == 0:  # Trunk
                colors[i] = [0.5, 0.3, 0.1]  # Brown
            elif part_id[i] == 1:  # Branches
                colors[i] = [0.5, 0.3, 0.1]  # Brown
            else:  # Leaves
                colors[i] = [0.1, 0.5, 0.1]  # Green
    
    set_colors()
    
    while window.running:
        # Handle input
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.SPACE:
                # Reset simulation
                actual_step_count[None] = 0
                stiffness_ramp_factor[None] = 0.01
                init_particles(x_np, v_np, part_ids_np)
                print("Simulation reset")
            elif window.event.key == 'q':
                break
            elif window.event.key == ti.ui.LEFT:
                new_strength = max(0.0, material_controller.wind_strength[None] - 0.1)
                material_controller.wind_strength[None] = new_strength
                print(f"Wind strength: {new_strength:.2f}")
            elif window.event.key == ti.ui.RIGHT:
                new_strength = min(5.0, material_controller.wind_strength[None] + 0.1)
                material_controller.wind_strength[None] = new_strength
                print(f"Wind strength: {new_strength:.2f}")

        # Continuous key handling
        if window.is_pressed(ti.ui.UP):
            drop_velocity[None] -= 0.1
            print(f"Drop velocity: {drop_velocity[None]:.2f}")
        if window.is_pressed(ti.ui.DOWN):
            drop_velocity[None] += 0.1
            print(f"Drop velocity: {drop_velocity[None]:.2f}")
        if window.is_pressed(ti.ui.LEFT):
            material_controller.wind_strength[None] = max(0, material_controller.wind_strength[None] - 0.01)
        if window.is_pressed(ti.ui.RIGHT):
            material_controller.wind_strength[None] = min(5.0, material_controller.wind_strength[None] + 0.01)
        
        # Check if reset button was pressed
        if material_controller.check_and_clear_reset_flag():
            actual_step_count[None] = 0
            stiffness_ramp_factor[None] = 0.01
            init_particles(x_np, v_np, part_ids_np)
            print("Simulation reset")

        # Update simulation
        simulate_step()

        # Update positions for rendering (use current buffer index)
        buffer_idx = actual_step_count[None] % buffer_size
        update_positions(positions, buffer_idx)
        
        # Render
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        
        scene.ambient_light((0.6, 0.6, 0.6))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
        
        # Draw particles with colors
        scene.particles(positions, radius=0.01, per_vertex_color=colors)
        
        canvas.scene(scene)
        
        # Display info panel
        window.GUI.begin("Info", 0.02, 0.02, 0.35, 0.34)
        window.GUI.text(f"Step: {actual_step_count[None]}")
        window.GUI.text(f"Particles: {n_particles}")
        window.GUI.text(f"Buffer: {buffer_size}")
        window.GUI.text(f"Wind: {material_controller.wind_strength[None]:.2f}")
        wind_dir = "→" if math.sin(actual_step_count[None] * 0.01) > 0 else "←"
        window.GUI.text(f"Wind dir: {wind_dir}")
        window.GUI.text(f"Wood E: {material_controller.E_wood[None] * stiffness_ramp_factor[None]:.0f}")
        window.GUI.text(f"Leaf E: {material_controller.E_leaf[None]:.1f}")
        window.GUI.text(f"Gravity: {material_controller.gravity[None]:.1f}")
        window.GUI.text(f"Damping: {material_controller.damping_factor[None]:.3f}")
        window.GUI.text(f"Timestep: {dt[None]:.5f}s")
        window.GUI.text("")
        window.GUI.text("Infinite simulation")
        window.GUI.text("Space: Reset | Arrows: Wind")
        window.GUI.text("Up/Down: Drop vel | Q: Quit")
        window.GUI.end()

        # Display material controls panel
        material_controller.render_controls(window)

        # Display recording panel
        recording_manager.render_ui(window)

        # Render frame first
        window.show()

        # Capture frame AFTER rendering (correct timing)
        if recording_manager.is_recording[None]:
            recording_manager.capture_frame(window)

if __name__ == '__main__':
    main()