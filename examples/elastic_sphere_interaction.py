import taichi as ti
import numpy as np
import math

# Initialize Taichi
real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True)

# Simulation parameters
dim = 3
n_particles = 0  # Will be set after sphere generation
n_grid = 64
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 5e-4  # Smaller timestep for stability
p_vol = 1
E = 50  # Young's modulus for elastic material (softer = more deformable without fracturing)
mu = E * 0.5  # Shear modulus
la = E * 0.5  # Lame's first parameter
max_steps = 2048
gravity = 0.0  # Zero gravity

# Sphere parameters
sphere_center = [0.5, 0.5, 0.5]
sphere_radius = 0.12
particle_spacing = 1.0 * dx  # Reduce particle count for better performance

# Cube parameters
cube_half_extent = 0.03  # Smaller cube for gentler collisions

# Damping for stability
damping = 0.98  # Stronger velocity damping to prevent fracturing

# Field definitions
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

# Particle fields
particle_type = ti.field(ti.i32)
x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

# Cube state
cube_center = vec()
cube_velocity = vec()
cube_prev_center = vec()
cube_prev_velocity = vec()

# Visualization
screen_res = 800
screen = ti.field(dtype=ti.f32, shape=(screen_res, screen_res, 3))

def generate_sphere_particles():
    """Generate particles in a sphere pattern"""
    global n_particles
    positions = []
    
    # Calculate bounds for particle generation
    min_coord = int((sphere_center[0] - sphere_radius) / particle_spacing)
    max_coord = int((sphere_center[0] + sphere_radius) / particle_spacing) + 1
    
    for i in range(min_coord, max_coord):
        for j in range(min_coord, max_coord):
            for k in range(min_coord, max_coord):
                pos = np.array([i, j, k]) * particle_spacing
                if np.linalg.norm(pos - sphere_center) <= sphere_radius:
                    # Add small random jitter
                    pos += (np.random.rand(3) - 0.5) * 0.05 * dx
                    positions.append(pos)
    
    n_particles = len(positions)
    print(f"Generated {n_particles} particles for sphere")
    return np.array(positions, dtype=np.float32)

def allocate_fields():
    """Allocate Taichi fields after n_particles is known"""
    ti.root.dense(ti.i, n_particles).place(particle_type)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.ijk, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(cube_center, cube_velocity, cube_prev_center, cube_prev_velocity)
    ti.root.lazy_grad()

@ti.kernel
def init_particles(positions: ti.types.ndarray(element_dim=1)):
    """Initialize particle states"""
    for i in range(n_particles):
        x[0, i] = positions[i]
        v[0, i] = [0, 0, 0]
        F[0, i] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        C[0, i] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        particle_type[i] = 1  # Elastic solid
    
    # Initialize cube at starting position
    cube_center[None] = [0.5, 0.5, 0.2]
    cube_velocity[None] = [0, 0, 0]
    cube_prev_center[None] = cube_center[None]
    cube_prev_velocity[None] = [0, 0, 0]

@ti.kernel
def clear_grid():
    for i, j, k in grid_m_in:
        grid_v_in[i, j, k] = [0, 0, 0]
        grid_m_in[i, j, k] = 0
        grid_v_out[i, j, k] = [0, 0, 0]

@ti.kernel
def p2g(f: ti.i32):
    """Particle to grid transfer with APIC"""
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        
        # Update deformation gradient
        new_F = (ti.Matrix.diag(dim=dim, val=1) + dt * C[f, p]) @ F[f, p]
        
        # Clamp singular values to prevent inversion and fracturing
        U, sig, V = ti.svd(new_F)
        for d in ti.static(range(3)):
            sig[d, d] = ti.max(0.6, ti.min(1.5, sig[d, d]))  # Tighter bounds to prevent fracturing
        new_F = U @ sig @ V.transpose()
        
        J = new_F.determinant()
        F[f + 1, p] = new_F
        
        # Neo-Hookean elasticity
        r, s = ti.polar_decompose(new_F)
        cauchy = mu * (new_F @ new_F.transpose()) + ti.Matrix.identity(real, dim) * (la * ti.log(J) - mu)
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + C[f, p]
        
        # Scatter to grid with APIC
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    offset = ti.Vector([i, j, k])
                    dpos = (ti.cast(ti.Vector([i, j, k]), real) - fx) * dx
                    weight = w[i][0] * w[j][1] * w[k][2]
                    ti.atomic_add(grid_v_in[base + offset], weight * (v[f, p] + affine @ dpos))
                    ti.atomic_add(grid_m_in[base + offset], weight)

bound = 3

@ti.kernel
def grid_op():
    """Grid operations including boundary conditions"""
    for i, j, k in grid_m_in:
        if grid_m_in[i, j, k] > 0:
            inv_m = 1 / grid_m_in[i, j, k]
            v_out = inv_m * grid_v_in[i, j, k]
            
            # No gravity in this simulation
            # v_out[1] -= dt * gravity
            
            # Boundary conditions
            if i < bound and v_out[0] < 0:
                v_out[0] = 0
            if i > n_grid - bound and v_out[0] > 0:
                v_out[0] = 0
            if j < bound and v_out[1] < 0:
                v_out[1] = 0
            if j > n_grid - bound and v_out[1] > 0:
                v_out[1] = 0
            if k < bound and v_out[2] < 0:
                v_out[2] = 0
            if k > n_grid - bound and v_out[2] > 0:
                v_out[2] = 0
            
            grid_v_out[i, j, k] = v_out

@ti.func
def cube_sdf(pos):
    """Signed distance function for cube"""
    offset = ti.abs(pos - cube_center[None]) - cube_half_extent
    return ti.max(ti.max(offset[0], offset[1]), offset[2])

@ti.func
def cube_sdf_normal(pos):
    """Normal vector at surface of cube (gradient of SDF)"""
    offset = pos - cube_center[None]
    abs_offset = ti.abs(offset) - cube_half_extent
    
    # Find which face we're closest to
    normal = ti.Vector([0.0, 0.0, 0.0])
    max_penetration = ti.max(ti.max(abs_offset[0], abs_offset[1]), abs_offset[2])
    
    # Use the face with maximum penetration
    if abs_offset[0] == max_penetration:
        normal[0] = 1.0 if offset[0] > 0 else -1.0
    elif abs_offset[1] == max_penetration:
        normal[1] = 1.0 if offset[1] > 0 else -1.0
    else:
        normal[2] = 1.0 if offset[2] > 0 else -1.0
    
    return normal

@ti.kernel
def grid_cube_contact():
    """Velocity-based contact using SDF"""
    for i, j, k in grid_v_out:
        if grid_m_in[i, j, k] > 0:
            node_pos = ti.Vector([i, j, k]) * dx + 0.5 * dx
            
            # Check SDF
            d = cube_sdf(node_pos)
            
            if d < 0:  # Inside cube
                n = cube_sdf_normal(node_pos)
                v_rel = grid_v_out[i, j, k] - cube_velocity[None]
                vn = v_rel.dot(n)
                
                if vn < 0:  # Approaching
                    # Remove normal component of relative velocity
                    grid_v_out[i, j, k] -= vn * n
                    
                    # Add some restitution (reduced to prevent bouncing/fracturing)
                    restitution = 0.1
                    grid_v_out[i, j, k] -= restitution * vn * n
                    
                    # Add a small amount of cube velocity to push particles gently
                    grid_v_out[i, j, k] += 0.1 * cube_velocity[None]

@ti.kernel
def g2p(f: ti.i32):
    """Grid to particle transfer with APIC"""
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        
        new_v = ti.Vector([0.0, 0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    dpos = ti.cast(ti.Vector([i, j, k]), real) - fx
                    g_v = grid_v_out[base[0] + i, base[1] + j, base[2] + k]
                    weight = w[i][0] * w[j][1] * w[k][2]
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        
        # Apply damping
        new_v *= damping
        
        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C

def update_cube_from_mouse(gui):
    """Update cube position based on mouse input with filtered velocity"""
    mouse_pos = gui.get_cursor_pos()
    
    # Map mouse position to 3D world coordinates
    # Mouse X -> World X, Mouse Y -> World Z (top-down view)
    target_center = [mouse_pos[0], 0.5, mouse_pos[1]]
    
    # Smooth cube movement with interpolation
    smoothing = 0.15  # Lower = smoother movement
    current = cube_center[None]
    new_center = [
        current[0] + (target_center[0] - current[0]) * smoothing,
        current[1] + (target_center[1] - current[1]) * smoothing,
        current[2] + (target_center[2] - current[2]) * smoothing
    ]
    
    # Calculate raw velocity
    raw_velocity = (new_center - cube_prev_center[None]) / dt
    
    # Filter velocity to reduce jitter
    alpha = 0.3  # Velocity filter parameter
    prev_vel = cube_prev_velocity[None]
    filtered_velocity = prev_vel + alpha * (raw_velocity - prev_vel)
    
    # Cap maximum velocity to prevent instability
    max_velocity = 2.0 * cube_half_extent / dt
    velocity_magnitude = math.sqrt(filtered_velocity[0]**2 + filtered_velocity[1]**2 + filtered_velocity[2]**2)
    if velocity_magnitude > max_velocity:
        scale = max_velocity / velocity_magnitude
        filtered_velocity = [filtered_velocity[0] * scale, filtered_velocity[1] * scale, filtered_velocity[2] * scale]
    
    # Update cube state
    cube_prev_center[None] = cube_center[None]
    cube_prev_velocity[None] = filtered_velocity
    cube_center[None] = new_center
    cube_velocity[None] = filtered_velocity

@ti.kernel
def render_particles(f: ti.i32):
    """Render particles to screen buffer"""
    # Clear screen
    for i, j in ti.ndrange(screen_res, screen_res):
        screen[i, j, 0] = 0.1
        screen[i, j, 1] = 0.1
        screen[i, j, 2] = 0.1
    
    # Draw particles (orthographic projection, XZ plane)
    for p in range(n_particles):
        pos = x[f, p]
        screen_x = int(pos[0] * screen_res)
        screen_y = int(pos[2] * screen_res)
        
        # Draw with some radius for visibility
        radius = 2
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                px = screen_x + dx
                py = screen_y + dy
                if 0 <= px < screen_res and 0 <= py < screen_res and dx*dx + dy*dy <= radius*radius:
                    # Color based on deformation
                    F_p = F[f, p]
                    J = F_p.determinant()
                    # Blue to red based on compression/extension
                    red = ti.min(1.0, ti.max(0.0, 2.0 * (J - 0.9)))
                    blue = ti.min(1.0, ti.max(0.0, 2.0 * (1.1 - J)))
                    screen[px, py, 0] = red
                    screen[px, py, 1] = 0.2
                    screen[px, py, 2] = blue

def draw_cube(gui):
    """Draw cube outline on GUI"""
    cube_pos = cube_center[None]
    # Project to 2D (XZ plane)
    center_2d = (cube_pos[0], cube_pos[2])
    size_2d = cube_half_extent * 2
    
    # Draw cube as white rectangle
    gui.rect(
        (center_2d[0] - cube_half_extent, center_2d[1] - cube_half_extent),
        (center_2d[0] + cube_half_extent, center_2d[1] + cube_half_extent),
        radius=2,
        color=0xFFFFFF
    )

def main():
    # Generate sphere particles
    particle_positions = generate_sphere_particles()
    
    # Allocate fields
    allocate_fields()
    
    # Initialize simulation
    init_particles(particle_positions)
    
    # Create GUI
    gui = ti.GUI("Elastic Sphere Interaction", res=screen_res)
    
    frame = 0
    while gui.running:
        # Update cube from mouse
        update_cube_from_mouse(gui)
        
        # Substeps (more with smaller dt)
        for s in range(20):
            clear_grid()
            p2g(frame)
            grid_op()
            grid_cube_contact()
            g2p(frame)
            frame = (frame + 1) % (max_steps - 1)
        
        # Render
        render_particles(frame)
        gui.set_image(screen)
        draw_cube(gui)
        
        # Show info
        gui.text(content=f"Particles: {n_particles}", pos=(0.05, 0.95), color=0xFFFFFF)
        gui.text(content="Move mouse to control white cube", pos=(0.05, 0.05), color=0xFFFFFF)
        
        gui.show()

if __name__ == '__main__':
    main()