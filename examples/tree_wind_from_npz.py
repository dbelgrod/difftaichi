"""
3D Tree wind simulation using Taichi MPM with tree loaded from NPZ file.
Loads tree point cloud from tree-gen folder and applies MPM physics simulation.

Usage:
    python tree_wind_from_npz.py [path_to_npz_file]
    python tree_wind_from_npz.py ../tree-gen/tree_mpm_optimized.npz
    python tree_wind_from_npz.py ../tree-gen/realistic_tree.npz

Controls:
- Space: Restart simulation
- Left/Right arrows: Adjust wind strength
- Q: Quit
"""

import taichi as ti
import numpy as np
import math
import sys
import os

# Use CPU for better stability, change to ti.gpu if you have a good GPU
ti.init(arch=ti.cpu)

# Simulation parameters
dim = 3
n_grid = 28  # Grid size
dx = 1 / n_grid
inv_dx = 1 / dx
base_dt = 5e-4  # Base time step for stability
dt = ti.field(ti.f32, shape=())  # Adaptive time step
p_vol = 1
max_steps = 1000  # Extended simulation to see continuous wind effects
gravity = 10
damping_factor = 0.995  # Numerical damping
max_velocity = 5.0  # Velocity clamping threshold

# Material properties for different tree parts
# Wood (trunk and branches) - very stiff, almost rigid
E_wood = 8000.0  # Very high stiffness - tree structure is almost rigid
mu_wood = 8000.0
la_wood = 8000.0

# Leaves - very soft and flexible
E_leaf = 0.5  # Very soft - will naturally deform and potentially break off
mu_leaf = 0.5
la_leaf = 0.5

# Stiffness ramping for stability
stiffness_ramp_factor = ti.field(ti.f32, shape=())
ramp_steps = 100  # Steps to reach full stiffness

# Particle data - will be set after loading NPZ
n_particles = 0
particle_type = None
material_id = None  # 0=wood (trunk/branches), 1=leaves
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
def p2g(f: ti.i32):
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

        F[f + 1, p] = new_F

        r, s = ti.polar_decompose(new_F)

        # Different material properties based on material type
        E = E_wood * stiffness_ramp_factor[None]
        mu = mu_wood * stiffness_ramp_factor[None]
        la = la_wood * stiffness_ramp_factor[None]

        if material_id[p] == 1:  # Leaves
            E = E_leaf
            mu = mu_leaf
            la = la_leaf

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
            v_out[1] -= dt[None] * gravity

            # Add wind force - oscillating with time
            wind_factor = ti.sin(ti.cast(current_step[None], ti.f32) * 0.01)
            wind_x = wind_strength[None] * wind_factor

            # Wind affects higher parts more
            height_factor = ti.cast(j, ti.f32) / ti.cast(n_grid, ti.f32)

            # Apply wind force (affects all particles through grid)
            # Stiff wood resists, soft leaves move easily
            v_out[0] += dt[None] * wind_x * height_factor * 10.0

            # Apply numerical damping
            v_out *= damping_factor

            # Velocity clamping
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

            # Ground boundary
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

        # Velocity clamping per particle
        v_mag = new_v.norm()
        if v_mag > max_velocity:
            new_v = new_v * (max_velocity / v_mag)

        # Anchor trunk base particles (wood particles near ground)
        if material_id[p] == 0 and x[f, p][1] < 0.15:
            new_v = ti.Vector([0, 0, 0])

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt[None] * v[f + 1, p]
        C[f + 1, p] = new_C

@ti.kernel
def init_particles(x_arr: ti.types.ndarray(ndim=2),
                   v_arr: ti.types.ndarray(ndim=2),
                   material_id_arr: ti.types.ndarray(ndim=1)):
    for i in range(n_particles):
        for j in ti.static(range(3)):
            x[0, i][j] = x_arr[i, j]
            v[0, i][j] = v_arr[i, j]
        F[0, i] = ti.Matrix.identity(ti.f32, dim)
        particle_type[i] = 1
        material_id[i] = material_id_arr[i]

@ti.kernel
def update_positions(positions: ti.template(), step: ti.i32):
    for i in range(n_particles):
        positions[i] = x[step, i]

def load_scene_from_npz(npz_path):
    """
    Load tree point cloud from NPZ file and prepare for simulation.

    Material IDs in NPZ:
      0 = wood (trunk/branches) - very stiff, almost rigid (E=8000)
      1 = leaves - very soft and flexible (E=0.5)

    Physics-based behavior:
      - Wind affects all particles through grid
      - Stiff wood resists deformation
      - Soft leaves deform easily and can naturally break off under high stress

    Coordinate system: Converts from tree-gen (Z-up) to MPM (Y-up)

    Returns (positions, velocities, material_ids, n_particles)
    """
    print(f"Loading tree from {npz_path}...")

    # Load NPZ file
    data = np.load(npz_path, allow_pickle=True)

    # Extract data
    positions = data['positions'].astype(np.float32)
    material_ids = data['material_ids'].astype(np.int32)

    # Convert from tree-gen coordinate system (Z-up) to MPM (Y-up)
    # Swap Y and Z axes: (X, Y, Z)_tree-gen → (X, Z, Y)_MPM
    positions = positions[:, [0, 2, 1]].copy()

    n = len(positions)
    print(f"Loaded {n} particles")

    # Count materials
    n_wood = np.sum(material_ids == 0)
    n_leaves = np.sum(material_ids == 1)
    print(f"  Wood particles: {n_wood}")
    print(f"  Leaf particles: {n_leaves}")

    # Normalize positions to fit in simulation domain [0, 1]
    # Calculate bounding box
    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)
    extent = max_pos - min_pos

    print(f"  Original bounding box: min={min_pos}, max={max_pos}")
    print(f"  Extent: {extent}")

    # Scale to fit in ~80% of domain (leaving margin for boundaries)
    target_size = 0.8
    scale_factor = target_size / extent.max()

    # Center the tree and scale
    centered = positions - min_pos  # Move to origin
    scaled = centered * scale_factor  # Scale to target size

    # Position tree: center in X-Z, base at y=0.05
    offset = np.array([0.5, 0.05, 0.5], dtype=np.float32)
    # Calculate X-Z center of current positions
    xz_center = np.array([scaled[:, 0].mean(), 0, scaled[:, 2].mean()], dtype=np.float32)
    xz_offset = np.array([0.5, 0, 0.5], dtype=np.float32) - xz_center

    # Apply final positioning
    normalized_positions = scaled.copy()
    normalized_positions[:, 0] += xz_offset[0]
    normalized_positions[:, 1] += offset[1]  # Base at 0.05
    normalized_positions[:, 2] += xz_offset[2]

    print(f"  Normalized bounding box: min={normalized_positions.min(axis=0)}, max={normalized_positions.max(axis=0)}")

    # Initialize velocities to zero
    velocities = np.zeros_like(positions)

    return normalized_positions, velocities, material_ids, n

@ti.kernel
def calculate_adaptive_timestep():
    """Calculate timestep based on CFL condition for current stiffness"""
    max_E = E_wood * stiffness_ramp_factor[None]
    wave_speed = ti.sqrt(max_E / 1000.0)
    safety_factor = 0.2
    max_dt = dx / wave_speed * safety_factor
    dt[None] = ti.min(max_dt, base_dt)

def update_stiffness_ramp():
    """Gradually increase stiffness to avoid sudden shocks"""
    if current_step[None] < ramp_steps:
        stiffness_ramp_factor[None] = (current_step[None] + 1) / ramp_steps
    else:
        stiffness_ramp_factor[None] = 1.0

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
        stiffness_ramp_factor[None] = 0.01

def main():
    global n_particles, particle_type, material_id, x, v, C, F

    # Parse command-line arguments
    if len(sys.argv) > 1:
        npz_path = sys.argv[1]
    else:
        # Default to MPM optimized tree (relative to repo root)
        # Try both relative paths (from examples/ and from repo root)
        candidates = [
            '../../tree-gen/tree_mpm_optimized.npz',  # From examples/
            '../tree-gen/tree_mpm_optimized.npz',     # From repo root
        ]
        npz_path = None
        for path in candidates:
            if os.path.exists(path):
                npz_path = path
                break
        if npz_path is None:
            npz_path = candidates[0]  # Default to first option

    # Check if file exists
    if not os.path.exists(npz_path):
        print(f"Error: File not found: {npz_path}")
        print("\nUsage: python tree_wind_from_npz.py [path_to_npz_file]")
        print("Example: python tree_wind_from_npz.py ../tree-gen/tree_mpm_optimized.npz")
        sys.exit(1)

    # Load scene from NPZ
    x_np, v_np, material_ids_np, n_particles = load_scene_from_npz(npz_path)
    print(f"\nRunning tree simulation with {n_particles} particles")

    # Allocate fields with correct size
    particle_type = ti.field(ti.i32, shape=n_particles)
    material_id = ti.field(ti.i32, shape=n_particles)
    x = ti.Vector.field(dim, dtype=ti.f32, shape=(max_steps, n_particles))
    v = ti.Vector.field(dim, dtype=ti.f32, shape=(max_steps, n_particles))
    C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(max_steps, n_particles))
    F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(max_steps, n_particles))

    # Set initial parameters
    wind_strength[None] = 0.0  # Start with no wind
    current_step[None] = 0
    stiffness_ramp_factor[None] = 0.01  # Start with very low stiffness
    dt[None] = base_dt

    # Create window
    window = ti.ui.Window("Tree Wind Simulation (from NPZ)", (800, 600))
    canvas = window.get_canvas()
    scene = window.get_scene()
    camera = ti.ui.Camera()

    # Set up camera
    camera.position(2.0, 1.5, 2.0)
    camera.lookat(0.5, 0.4, 0.5)
    camera.up(0, 1, 0)

    # Initialize particles
    init_particles(x_np, v_np, material_ids_np)

    # Position and color fields for rendering
    positions = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
    colors = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)

    # Set colors based on material type
    @ti.kernel
    def set_colors():
        for i in range(n_particles):
            if material_id[i] == 0:  # Wood (trunk/branches)
                colors[i] = [0.5, 0.3, 0.1]  # Brown
            else:  # Leaves
                colors[i] = [0.1, 0.5, 0.1]  # Green

    set_colors()

    print("\nControls:")
    print("  Space: Reset simulation")
    print("  Left/Right arrows: Adjust wind strength")
    print("  Q: Quit")
    print("\nStarting simulation...")

    while window.running:
        # Handle input
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.SPACE:
                # Reset simulation
                current_step[None] = 0
                stiffness_ramp_factor[None] = 0.01
                init_particles(x_np, v_np, material_ids_np)
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

        # Continuous key handling
        if window.is_pressed(ti.ui.LEFT):
            wind_strength[None] = max(0, wind_strength[None] - 0.01)
        if window.is_pressed(ti.ui.RIGHT):
            wind_strength[None] = min(2.0, wind_strength[None] + 0.01)

        # Update simulation
        simulate_step()

        # Check if we need to reinitialize at loop restart
        if current_step[None] == 0:
            stiffness_ramp_factor[None] = 0.01
            init_particles(x_np, v_np, material_ids_np)

        # Update positions for rendering
        update_positions(positions, current_step[None])

        # Render
        camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)

        scene.ambient_light((0.6, 0.6, 0.6))
        scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))

        # Draw particles with colors
        scene.particles(positions, radius=0.008, per_vertex_color=colors)

        canvas.scene(scene)

        # Display info
        window.GUI.begin("Info", 0.02, 0.02, 0.35, 0.28)
        window.GUI.text(f"Step: {current_step[None]}/{max_steps}")
        window.GUI.text(f"Particles: {n_particles}")
        window.GUI.text(f"Wind strength: {wind_strength[None]:.2f}")
        wind_dir = "→" if math.sin(current_step[None] * 0.01) > 0 else "←"
        window.GUI.text(f"Wind direction: {wind_dir}")
        window.GUI.text(f"Wood E: {E_wood * stiffness_ramp_factor[None]:.0f}")
        window.GUI.text(f"Leaf E: {E_leaf:.1f}")
        window.GUI.text("Space: Reset | Left/Right: Wind | Q: Quit")
        window.GUI.end()

        window.show()

if __name__ == '__main__':
    main()
