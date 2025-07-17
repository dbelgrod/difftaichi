"""
3D Tree wind simulation using Taichi MPM with robust high stiffness support.
This version uses advanced stability techniques to handle very stiff materials.
Controls:
- Space: Restart simulation
- Left/Right arrows: Adjust wind strength
- 1/2/3: Set stiffness to 100/500/1000
- Q: Quit
"""

import taichi as ti
import numpy as np
import math

# Use CPU for better stability, change to ti.gpu if you have a good GPU
ti.init(arch=ti.cpu)

# Simulation parameters
dim = 3
n_grid = 32  # Increased grid resolution for better accuracy
dx = 1 / n_grid
inv_dx = 1 / dx
base_dt = 3e-4  # Much smaller base timestep
dt = ti.field(ti.f32, shape=())  # Adaptive time step
substeps = ti.field(ti.i32, shape=())  # Dynamic substeps
p_vol = 1
max_steps = 2000  # Extended simulation
gravity = 10
damping_factor = 0.995  # Stronger damping
max_velocity = 5.0  # More conservative velocity limit

# Material properties for different tree parts
E_wood_base = ti.field(ti.f32, shape=())  # Adjustable stiffness
mu_wood_base = ti.field(ti.f32, shape=())
la_wood_base = ti.field(ti.f32, shape=())

# Leaves - very soft and light
E_leaf = 0.5
mu_leaf = 0.5
la_leaf = 0.5

# Stiffness ramping for stability
stiffness_ramp_factor = ti.field(ti.f32, shape=())
ramp_steps = 300  # More gradual ramp

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
wind_strength = ti.field(ti.f32, shape=())
current_step = ti.field(ti.i32, shape=())

@ti.kernel
def clear_grid():
    for i, j, k in grid_m_in:
        grid_v_in[i, j, k] = [0, 0, 0]
        grid_m_in[i, j, k] = 0

@ti.kernel
def compute_adaptive_timestep_and_substeps():
    """Compute timestep and substeps based on current stiffness"""
    # Get maximum stiffness in the system
    max_E = E_wood_base[None] * stiffness_ramp_factor[None]
    
    # Wave speed c = sqrt(E/rho), assuming rho=1000 kg/m^3
    wave_speed = ti.sqrt(max_E / 1000.0)
    
    # CFL condition with very conservative safety factor
    safety_factor = 0.1  # Very conservative for high stiffness
    max_dt = dx / wave_speed * safety_factor
    
    # Compute substeps if needed
    if max_dt < base_dt:
        substeps[None] = ti.cast(ti.ceil(base_dt / max_dt), ti.i32)
        dt[None] = base_dt / ti.cast(substeps[None], ti.f32)
    else:
        substeps[None] = 1
        dt[None] = base_dt

@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        
        # Update deformation gradient with current timestep
        dt_local = dt[None]
        new_F = (ti.Matrix.identity(ti.f32, dim) + dt_local * C[f, p]) @ F[f, p]
        
        # Clamp deformation gradient determinant
        J = new_F.determinant()
        J_min = 0.4
        J_max = 2.5
        if J < J_min:
            new_F = new_F * ti.pow(J_min / J, 1.0/3.0)
            J = J_min
        elif J > J_max:
            new_F = new_F * ti.pow(J_max / J, 1.0/3.0)
            J = J_max
        
        F[f + 1, p] = new_F
        
        # Polar decomposition for stress calculation
        r, s = ti.polar_decompose(new_F)
        
        # Material properties with ramping
        ramp = stiffness_ramp_factor[None]
        E = E_wood_base[None] * ramp
        mu = mu_wood_base[None] * ramp
        la = la_wood_base[None] * ramp
        
        if part_id[p] == 2:  # Leaves
            E = E_leaf
            mu = mu_leaf
            la = la_leaf
        elif part_id[p] == 1:  # Branches
            E = E_wood_base[None] * 0.8 * ramp
            mu = mu_wood_base[None] * 0.8 * ramp
            la = la_wood_base[None] * 0.8 * ramp
        
        # Neo-Hookean constitutive model
        cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + ti.Matrix.identity(ti.f32, dim) * la * J * (J - 1)
        
        # Stress with conservative dt scaling
        stress = -(dt_local * p_vol * 4 * inv_dx * inv_dx) * cauchy
        
        # Advanced stress limiting
        stress_norm = stress.norm()
        max_stress = 5e3 / (1.0 + ramp * 10.0)  # Adaptive stress limit
        if stress_norm > max_stress:
            stress = stress * (max_stress / stress_norm)
        
        affine = stress + C[f, p]
        
        # Scatter to grid
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    offset = ti.Vector([i, j, k])
                    dpos = (ti.cast(ti.Vector([i, j, k]), ti.f32) - fx) * dx
                    weight = w[i][0] * w[j][1] * w[k][2]
                    # Reduce mass contribution for leaves (lighter particles)
                    mass_scale = 1.0
                    if part_id[p] == 2:  # Leaves have less mass
                        mass_scale = 0.3
                    ti.atomic_add(grid_v_in[base + offset], weight * (v[f, p] + affine @ dpos))
                    ti.atomic_add(grid_m_in[base + offset], weight * mass_scale)

@ti.kernel
def grid_op():
    dt_local = dt[None]
    for i, j, k in grid_m_in:
        if grid_m_in[i, j, k] > 0:
            # Semi-implicit velocity update
            v_in = grid_v_in[i, j, k] / grid_m_in[i, j, k]
            
            # Apply gravity
            v_out = v_in
            v_out[1] -= dt_local * gravity
            
            # Wind force with height scaling
            wind_time = ti.cast(current_step[None], ti.f32) * dt_local
            wind_factor = ti.sin(wind_time * 3.0)  # Faster oscillation
            wind_x = wind_strength[None] * wind_factor
            
            height_factor = ti.cast(j, ti.f32) / ti.cast(n_grid, ti.f32)
            v_out[0] += dt_local * wind_x * height_factor * 3.0
            
            # Apply strong damping for stability
            v_out *= damping_factor
            
            # Aggressive velocity clamping
            v_mag = v_out.norm()
            if v_mag > max_velocity:
                v_out = v_out * (max_velocity / v_mag)
            
            # Boundary conditions with extra damping
            boundary_damping = 0.5
            if i < 3:
                v_out[0] = ti.max(0.0, v_out[0]) * boundary_damping
            if i > n_grid - 3:
                v_out[0] = ti.min(0.0, v_out[0]) * boundary_damping
            if k < 3:
                v_out[2] = ti.max(0.0, v_out[2]) * boundary_damping
            if k > n_grid - 3:
                v_out[2] = ti.min(0.0, v_out[2]) * boundary_damping
            
            # Ground boundary - strong anchoring
            if j < 3:
                if v_out[1] < 0:
                    v_out = [0, 0, 0]
                else:
                    v_out *= 0.1  # Strong damping near ground
            if j > n_grid - 3 and v_out[1] > 0:
                v_out[1] = 0
                
            grid_v_out[i, j, k] = v_out

@ti.kernel
def g2p(f: ti.i32):
    dt_local = dt[None]
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.f32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector.zero(ti.f32, dim)
        new_C = ti.Matrix.zero(ti.f32, dim, dim)
        
        # Gather from grid
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    dpos = ti.cast(ti.Vector([i, j, k]), ti.f32) - fx
                    g_v = grid_v_out[base[0] + i, base[1] + j, base[2] + k]
                    weight = w[i][0] * w[j][1] * w[k][2]
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        
        # Extra wind on leaves (lighter = more affected)
        if part_id[p] == 2:
            wind_time = ti.cast(current_step[None], ti.f32) * dt_local
            wind_factor = ti.sin(wind_time * 3.0)
            new_v[0] += wind_strength[None] * wind_factor * 0.5
            new_v[1] += wind_strength[None] * ti.abs(wind_factor) * 0.1  # Slight lift
        
        # Velocity limiting
        v_mag = new_v.norm()
        if v_mag > max_velocity:
            new_v = new_v * (max_velocity / v_mag)
        
        # Strong anchoring for trunk base
        if part_id[p] == 0 and x[f, p][1] < 0.12:  # Deeper anchoring
            anchor_factor = (0.12 - x[f, p][1]) / 0.12  # Gradual anchoring
            new_v *= (1.0 - anchor_factor * 0.95)
        
        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt_local * v[f + 1, p]
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
    """Create a simpler tree with fewer particles for stability"""
    particle_positions = []
    particle_velocities = []
    part_ids = []
    
    # Tree structure parameters
    trunk_width = 0.08
    trunk_height = 0.4
    trunk_depth = 0.08
    
    # Trunk - fewer particles
    trunk_particles_x = 4
    trunk_particles_y = 16
    trunk_particles_z = 4
    
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
    
    # Fewer branches
    branch_width = 0.04
    branch_length = 0.15
    branch_depth = 0.04
    branch_particles = 2  # Fewer particles
    
    # Only 2 main branches
    branches = [
        (0.3, 40, 'left'),
        (0.35, 35, 'right'),
    ]
    
    for branch_height, angle, side in branches:
        start_x = 0.5
        start_y = 0.05 + branch_height
        start_z = 0.5
        
        if side == 'left':
            start_x -= trunk_width/2
            dir_x = -math.cos(math.radians(angle))
            dir_z = 0
        else:
            start_x += trunk_width/2
            dir_x = math.cos(math.radians(angle))
            dir_z = 0
        
        dir_y = math.sin(math.radians(angle))
        
        # Create branch particles
        for i in range(branch_particles):
            for j in range(branch_particles):
                for k in range(int(branch_particles * 3)):  # Shorter branches
                    t = k / (branch_particles * 3 - 1)
                    pos = [start_x + dir_x * branch_length * t + (i - branch_particles/2) * branch_width/branch_particles,
                           start_y + dir_y * branch_length * t + (j - branch_particles/2) * branch_depth/branch_particles,
                           start_z + dir_z * branch_length * t + (j - branch_particles/2) * branch_depth/branch_particles]
                    particle_positions.append(pos)
                    particle_velocities.append([0.0, 0.0, 0.0])
                    part_ids.append(1)  # Branch
        
        # Fewer leaves
        leaf_radius = 0.025
        leaf_particles_per_cluster = 15
        
        for leaf_idx in range(2):  # Only 2 leaf clusters per branch
            leaf_angle = leaf_idx * 180
            leaf_offset_x = math.cos(math.radians(leaf_angle)) * leaf_radius
            leaf_offset_z = math.sin(math.radians(leaf_angle)) * leaf_radius
            
            center_x = start_x + dir_x * branch_length + leaf_offset_x
            center_y = start_y + dir_y * branch_length
            center_z = start_z + dir_z * branch_length + leaf_offset_z
            
            # Create leaf particles
            for _ in range(leaf_particles_per_cluster):
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

def update_stiffness_ramp():
    """Smoother stiffness ramping"""
    if current_step[None] < ramp_steps:
        t = current_step[None] / ramp_steps
        # Smooth cubic ramp
        stiffness_ramp_factor[None] = t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)
    else:
        stiffness_ramp_factor[None] = 1.0

def set_stiffness(E_value):
    """Set the base stiffness values"""
    E_wood_base[None] = E_value
    mu_wood_base[None] = E_value
    la_wood_base[None] = E_value
    current_step[None] = 0
    stiffness_ramp_factor[None] = 0.001
    print(f"Stiffness set to E={E_value}")

def simulate_step():
    step = current_step[None]
    if step < max_steps - 1:
        update_stiffness_ramp()
        compute_adaptive_timestep_and_substeps()
        
        # Perform substeps
        for _ in range(substeps[None]):
            clear_grid()
            p2g(step)
            grid_op()
            g2p(step)
        
        current_step[None] = step + 1
    else:
        # Auto-restart
        current_step[None] = 0
        stiffness_ramp_factor[None] = 0.001

def main():
    global n_particles, particle_type, part_id, x, v, C, F
    
    # Create scene
    x_np, v_np, part_ids_np, n_particles = create_scene()
    print(f"Running robust tree simulation with {n_particles} particles")
    print("Press 1/2/3 to set stiffness to 100/500/1000")
    
    # Allocate fields
    particle_type = ti.field(ti.i32, shape=n_particles)
    part_id = ti.field(ti.i32, shape=n_particles)
    x = ti.Vector.field(dim, dtype=ti.f32, shape=(max_steps, n_particles))
    v = ti.Vector.field(dim, dtype=ti.f32, shape=(max_steps, n_particles))
    C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(max_steps, n_particles))
    F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(max_steps, n_particles))
    
    # Initialize
    wind_strength[None] = 0.0
    current_step[None] = 0
    stiffness_ramp_factor[None] = 0.001
    dt[None] = base_dt
    substeps[None] = 1
    set_stiffness(1000)  # Start with high stiffness
    
    # Create window
    window = ti.ui.Window("Robust Tree Wind Simulation", (800, 600))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    
    camera.position(1.5, 1.0, 1.5)
    camera.lookat(0.5, 0.3, 0.5)
    camera.up(0, 1, 0)
    
    # Initialize particles
    init_particles(x_np, v_np, part_ids_np)
    
    # Rendering fields
    positions = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
    colors = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
    
    @ti.kernel
    def set_colors():
        for i in range(n_particles):
            if part_id[i] == 0:  # Trunk
                colors[i] = [0.4, 0.25, 0.1]
            elif part_id[i] == 1:  # Branches
                colors[i] = [0.45, 0.3, 0.15]
            else:  # Leaves
                colors[i] = [0.2, 0.6, 0.2]
    
    set_colors()
    
    while window.running:
        # Handle input
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.SPACE:
                current_step[None] = 0
                stiffness_ramp_factor[None] = 0.001
                init_particles(x_np, v_np, part_ids_np)
                print("Simulation reset")
            elif e.key == 'q':
                break
            elif e.key == ti.ui.LEFT:
                wind_strength[None] = max(0.0, wind_strength[None] - 0.1)
                print(f"Wind strength: {wind_strength[None]:.2f}")
            elif e.key == ti.ui.RIGHT:
                wind_strength[None] = min(1.0, wind_strength[None] + 0.1)
                print(f"Wind strength: {wind_strength[None]:.2f}")
            elif e.key == '1':
                set_stiffness(100)
                init_particles(x_np, v_np, part_ids_np)
            elif e.key == '2':
                set_stiffness(500)
                init_particles(x_np, v_np, part_ids_np)
            elif e.key == '3':
                set_stiffness(1000)
                init_particles(x_np, v_np, part_ids_np)
        
        # Update simulation
        simulate_step()
        
        # Reinitialize if needed
        if current_step[None] == 0:
            stiffness_ramp_factor[None] = 0.001
            init_particles(x_np, v_np, part_ids_np)
        
        # Update positions
        update_positions(positions, current_step[None])
        
        # Render
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(pos=(0.5, 1.0, 0.5), color=(0.8, 0.8, 0.8))
        
        scene.particles(positions, radius=0.012, per_vertex_color=colors)
        
        canvas.scene(scene)
        
        # Display info
        window.GUI.begin("Info", 0.02, 0.02, 0.42, 0.35)
        window.GUI.text(f"Step: {current_step[None]}/{max_steps}")
        window.GUI.text(f"Particles: {n_particles}")
        window.GUI.text(f"Grid: {n_grid}x{n_grid}x{n_grid}")
        window.GUI.text(f"Wind: {wind_strength[None]:.2f} (0.0-1.0)")
        wind_dir = "→" if math.sin(current_step[None] * dt[None] * 3.0) > 0 else "←"
        window.GUI.text(f"Wind direction: {wind_dir}")
        window.GUI.text(f"Stiffness: {E_wood_base[None] * stiffness_ramp_factor[None]:.1f}/{E_wood_base[None]:.0f}")
        window.GUI.text(f"Timestep: {dt[None]:.6f}s")
        window.GUI.text(f"Substeps: {substeps[None]}")
        window.GUI.text("Space: Reset | Left/Right: Wind")
        window.GUI.text("1/2/3: Stiffness 100/500/1000 | Q: Quit")
        window.GUI.end()
        
        window.show()

if __name__ == '__main__':
    main()