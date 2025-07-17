"""
3D Tree wind simulation using Taichi MPM with high stiffness support.
This version demonstrates handling very stiff materials (E=1000+) without numerical instability.
Controls:
- Space: Restart simulation
- Up/Down arrows: Adjust drop velocity (legacy, kept for compatibility)
- Left/Right arrows: Adjust wind strength
- 1/2/3: Set stiffness to 500/1000/2000
- Q: Quit
"""

import taichi as ti
import numpy as np
import math

# Use CPU for better stability, change to ti.gpu if you have a good GPU
ti.init(arch=ti.cpu)

# Simulation parameters
dim = 3
n_grid = 24  # Reduced grid size
dx = 1 / n_grid
inv_dx = 1 / dx
base_dt = 2e-3  # Base time step
dt = ti.field(ti.f32, shape=())  # Adaptive time step
p_vol = 1
max_steps = 1000  # Extended simulation to see continuous wind effects
gravity = 10
damping_factor = 0.99  # Numerical damping
max_velocity = 10.0  # Velocity clamping threshold

# Material properties for different tree parts
# Trunk and branches - stiff wood (can now handle higher values)
E_wood_base = ti.field(ti.f32, shape=())  # Adjustable stiffness
mu_wood_base = ti.field(ti.f32, shape=())
la_wood_base = ti.field(ti.f32, shape=())

# Leaves - soft and bendable
E_leaf = 1.0
mu_leaf = 1.0
la_leaf = 1.0

# Stiffness ramping for stability
stiffness_ramp_factor = ti.field(ti.f32, shape=())
ramp_steps = 200  # More steps for higher stiffness

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
wind_strength = ti.field(ti.f32, shape=())
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
        new_F = (ti.Matrix.identity(ti.f32, dim) + dt[None] * C[f, p]) @ F[f, p]
        J = new_F.determinant()
        
        # Clamp Jacobian to prevent extreme deformations
        if J < 0.3:
            J = 0.3
            new_F = new_F * ti.pow(0.3 / new_F.determinant(), 1.0/3.0)
        elif J > 3.0:
            J = 3.0
            new_F = new_F * ti.pow(3.0 / new_F.determinant(), 1.0/3.0)
        
        F[f + 1, p] = new_F
        
        r, s = ti.polar_decompose(new_F)
        
        # Different material properties based on part type
        E = E_wood_base[None] * stiffness_ramp_factor[None]
        mu = mu_wood_base[None] * stiffness_ramp_factor[None]
        la = la_wood_base[None] * stiffness_ramp_factor[None]
        if part_id[p] == 2:  # Leaves
            E = E_leaf
            mu = mu_leaf
            la = la_leaf
        elif part_id[p] == 1:  # Branches (almost as stiff as trunk)
            E = E_wood_base[None] * 0.9 * stiffness_ramp_factor[None]
            mu = mu_wood_base[None] * 0.9 * stiffness_ramp_factor[None]
            la = la_wood_base[None] * 0.9 * stiffness_ramp_factor[None]
        
        cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + ti.Matrix.identity(ti.f32, dim) * la * J * (J - 1)
        stress = -(dt[None] * p_vol * 4 * inv_dx * inv_dx) * cauchy
        
        # Clamp stress to prevent numerical issues
        stress_norm = stress.norm()
        max_stress = 1e4
        if stress_norm > max_stress:
            stress = stress * (max_stress / stress_norm)
        
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
            v_out[1] -= dt[None] * gravity
            
            # Add wind force - oscillating with time
            # Wind primarily in x direction, with some variation
            wind_factor = ti.sin(ti.cast(current_step[None], ti.f32) * 0.01)  # Smooth oscillation
            wind_x = wind_strength[None] * wind_factor
            
            # Wind affects higher parts more (based on y coordinate)
            # And affects leaves more than branches, branches more than trunk
            height_factor = ti.cast(j, ti.f32) / ti.cast(n_grid, ti.f32)  # 0 at bottom, 1 at top
            
            # Apply wind force (increased effect)
            v_out[0] += dt[None] * wind_x * height_factor * 5.0
            
            # Apply numerical damping
            v_out *= damping_factor
            
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
        
        # Additional wind effect directly on particles (especially leaves)
        if part_id[p] == 2:  # Leaves get extra wind
            wind_factor = ti.sin(ti.cast(current_step[None], ti.f32) * 0.01)
            new_v[0] += wind_strength[None] * wind_factor * 0.5
        
        # Velocity clamping per particle
        v_mag = new_v.norm()
        if v_mag > max_velocity:
            new_v = new_v * (max_velocity / v_mag)
        
        # Anchor trunk base particles
        if part_id[p] == 0 and x[f, p][1] < 0.1:  # Trunk particles near ground
            new_v = ti.Vector([0, 0, 0])
        
        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt[None] * v[f + 1, p]
        C[f + 1, p] = new_C

@ti.kernel
def init_particles(x_arr: ti.types.ndarray(ndim=2), 
                   v_arr: ti.types.ndarray(ndim=2),
                   part_id_arr: ti.types.ndarray(ndim=1)):
    for i in range(n_particles):
        for j in ti.static(range(3)):
            x[0, i][j] = x_arr[i, j]
            v[0, i][j] = v_arr[i, j]
        F[0, i] = ti.Matrix.identity(ti.f32, dim)
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

def calculate_adaptive_timestep():
    """Calculate timestep based on CFL condition for current stiffness"""
    # Wave speed c = sqrt(E/rho), assuming rho=1000 kg/m^3
    max_E = E_wood_base[None] * stiffness_ramp_factor[None]
    wave_speed = ti.sqrt(max_E / 1000.0)
    # CFL condition: dt < dx / c * safety_factor
    safety_factor = 0.3  # Extra conservative for high stiffness
    max_dt = dx / wave_speed * safety_factor
    # Clamp to reasonable range
    dt[None] = min(max_dt, base_dt)

def update_stiffness_ramp():
    """Gradually increase stiffness to avoid sudden shocks"""
    if current_step[None] < ramp_steps:
        # Use smoother ramp function for high stiffness
        t = current_step[None] / ramp_steps
        # Smoothstep function for smoother transition
        stiffness_ramp_factor[None] = t * t * (3.0 - 2.0 * t)
    else:
        stiffness_ramp_factor[None] = 1.0

def set_stiffness(E_value):
    """Set the base stiffness values"""
    E_wood_base[None] = E_value
    mu_wood_base[None] = E_value
    la_wood_base[None] = E_value
    # Reset simulation with new stiffness
    current_step[None] = 0
    stiffness_ramp_factor[None] = 0.01
    print(f"Stiffness set to E={E_value}")

def simulate_step():
    step = current_step[None]
    if step < max_steps - 1:
        update_stiffness_ramp()
        calculate_adaptive_timestep()
        clear_grid()
        p2g(step)
        grid_op()
        g2p(step)
        current_step[None] = step + 1
    else:
        # Auto-restart when simulation ends
        current_step[None] = 0
        stiffness_ramp_factor[None] = 0.01  # Reset ramp

def main():
    global n_particles, particle_type, part_id, x, v, C, F
    
    # Create scene and get particle count
    x_np, v_np, part_ids_np, n_particles = create_scene()
    print(f"Running high-stiffness tree simulation with {n_particles} particles")
    print("Press 1/2/3 to set stiffness to 500/1000/2000")
    
    # Now allocate fields with correct size
    particle_type = ti.field(ti.i32, shape=n_particles)
    part_id = ti.field(ti.i32, shape=n_particles)
    x = ti.Vector.field(dim, dtype=ti.f32, shape=(max_steps, n_particles))
    v = ti.Vector.field(dim, dtype=ti.f32, shape=(max_steps, n_particles))
    C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(max_steps, n_particles))
    F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(max_steps, n_particles))
    
    # Set initial parameters
    drop_velocity[None] = -2.0  # Legacy parameter
    wind_strength[None] = 0.0  # Start with no wind to see tree standing straight
    current_step[None] = 0
    stiffness_ramp_factor[None] = 0.01  # Start with very low stiffness
    dt[None] = base_dt  # Initialize timestep
    set_stiffness(500)  # Start with moderate stiffness
    
    # Create window
    window = ti.ui.Window("High-Stiffness Tree Wind Simulation", (800, 600))
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
                current_step[None] = 0
                stiffness_ramp_factor[None] = 0.01
                init_particles(x_np, v_np, part_ids_np)
                print("Simulation reset")
            elif window.event.key == 'q':
                break
            elif window.event.key == ti.ui.LEFT:
                new_strength = max(0.0, wind_strength[None] - 0.1)
                wind_strength[None] = new_strength
                print(f"Wind strength: {new_strength:.2f}")
            elif window.event.key == ti.ui.RIGHT:
                new_strength = min(2.0, wind_strength[None] + 0.1)
                wind_strength[None] = new_strength
                print(f"Wind strength: {new_strength:.2f}")
            elif window.event.key == '1':
                set_stiffness(500)
                init_particles(x_np, v_np, part_ids_np)
            elif window.event.key == '2':
                set_stiffness(1000)
                init_particles(x_np, v_np, part_ids_np)
            elif window.event.key == '3':
                set_stiffness(2000)
                init_particles(x_np, v_np, part_ids_np)
        
        # Continuous key handling
        if window.is_pressed(ti.ui.UP):
            drop_velocity[None] -= 0.1
            print(f"Drop velocity: {drop_velocity[None]:.2f}")
        if window.is_pressed(ti.ui.DOWN):
            drop_velocity[None] += 0.1
            print(f"Drop velocity: {drop_velocity[None]:.2f}")
        if window.is_pressed(ti.ui.LEFT):
            wind_strength[None] = max(0, wind_strength[None] - 0.1)
            print(f"Wind strength: {wind_strength[None]:.2f}")
        if window.is_pressed(ti.ui.RIGHT):
            wind_strength[None] = min(2.0, wind_strength[None] + 0.1)
            print(f"Wind strength: {wind_strength[None]:.2f}")
        
        # Update simulation
        simulate_step()
        
        # Check if we need to reinitialize at loop restart
        if current_step[None] == 0:
            stiffness_ramp_factor[None] = 0.01
            init_particles(x_np, v_np, part_ids_np)
        
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
        window.GUI.begin("Info", 0.02, 0.02, 0.40, 0.32)
        window.GUI.text(f"Step: {current_step[None]}/{max_steps}")
        window.GUI.text(f"Particles: {n_particles}")
        window.GUI.text(f"Wind strength: {wind_strength[None]:.2f} (0.0-2.0)")
        wind_dir = "→" if math.sin(current_step[None] * 0.01) > 0 else "←"
        window.GUI.text(f"Wind direction: {wind_dir}")
        window.GUI.text(f"Stiffness: {E_wood_base[None] * stiffness_ramp_factor[None]:.1f}/{E_wood_base[None]:.1f}")
        window.GUI.text(f"Timestep: {dt[None]:.6f}s")
        window.GUI.text("Auto-looping simulation")
        window.GUI.text("Space: Reset | Left/Right: Wind")
        window.GUI.text("1/2/3: Stiffness 500/1000/2000")
        window.GUI.text("Up/Down: Drop vel | Q: Quit")
        window.GUI.end()
        
        window.show()

if __name__ == '__main__':
    main()