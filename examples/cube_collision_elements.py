"""
Simple cube collision simulation with Taichi Elements and GGUI.
Two elastic cubes colliding with each other.
Controls:
- Space: Restart simulation
- Up/Down arrows: Adjust drop velocity
- Q: Quit
"""

import taichi as ti
import numpy as np
import sys
import os

# Add the taichi_elements path - assumes it's in the same parent directory
taichi_elements_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'taichi_elements')
if os.path.exists(taichi_elements_path):
    sys.path.insert(0, taichi_elements_path)
else:
    print("Error: taichi_elements not found at", taichi_elements_path)
    print("Please ensure taichi_elements is cloned to ~/Repos/taichi_elements")
    sys.exit(1)

try:
    from engine.mpm_solver import MPMSolver
except ImportError:
    print("Error: Could not import MPMSolver from taichi_elements")
    print("Make sure taichi_elements is properly installed")
    sys.exit(1)

# Use CPU for better stability
ti.init(arch=ti.cpu)

# Control parameters
drop_velocity = ti.field(ti.f32, shape=())
current_step = ti.field(ti.i32, shape=())
drop_velocity[None] = -2.0
current_step[None] = 0
max_steps = 1000

def create_scene():
    # Create solver
    mpm = MPMSolver(res=(32, 32, 32), size=1, max_num_particles=2 ** 15, use_ggui=True)
    
    # Stationary cube (blue, elastic)
    mpm.add_cube(lower_corner=[0.4, 0.1, 0.4],
                 cube_size=[0.15, 0.15, 0.15],
                 material=MPMSolver.material_elastic,
                 velocity=[0, 0, 0])
    
    # Falling cube (red, elastic) - slightly offset for 90% overlap
    mpm.add_cube(lower_corner=[0.415, 0.5, 0.4],
                 cube_size=[0.15, 0.15, 0.15],
                 material=MPMSolver.material_elastic,
                 velocity=[0, drop_velocity[None], 0])
    
    # Set gravity
    mpm.set_gravity((0, -10, 0))
    
    return mpm

# Material colors: [water, elastic, snow, sand]
material_type_colors = np.array([
    [0.1, 0.1, 1.0, 0.8],  # water - blue
    [0.8, 0.3, 0.2, 1.0],  # elastic - red (we'll override this per cube)
    [1.0, 1.0, 1.0, 1.0],  # snow - white
    [1.0, 1.0, 0.0, 1.0]   # sand - yellow
])

@ti.kernel
def set_cube_colors(ti_color: ti.template(), ti_material: ti.template(), ti_positions: ti.template()):
    """Color cubes based on their Y position - lower cube blue, upper cube red"""
    for I in ti.grouped(ti_material):
        if ti_material[I] == 1:  # Elastic material
            # Color based on initial Y position
            if ti_positions[I][1] < 0.3:  # Lower cube
                ti_color[I] = ti.Vector([0.2, 0.3, 0.8, 1.0])  # Blue
            else:  # Upper cube
                ti_color[I] = ti.Vector([0.8, 0.3, 0.2, 1.0])  # Red
        elif ti_material[I] == 0:  # Water
            ti_color[I] = ti.Vector([0.1, 0.1, 1.0, 0.8])
        elif ti_material[I] == 2:  # Snow  
            ti_color[I] = ti.Vector([1.0, 1.0, 1.0, 1.0])
        elif ti_material[I] == 3:  # Sand
            ti_color[I] = ti.Vector([1.0, 1.0, 0.0, 1.0])
        else:
            ti_color[I] = ti.Vector([0.5, 0.5, 0.5, 1.0])  # Default gray

# Initialize scene
mpm = create_scene()

# GGUI setup
res = (800, 600)
window = ti.ui.Window("Cube Collision with Taichi Elements", res, vsync=True)
canvas = window.get_canvas()
scene = window.get_scene()
camera = ti.ui.make_camera()
camera.position(2.0, 1.5, 2.0)
camera.lookat(0.5, 0.4, 0.5)
camera.fov(55)
particles_radius = 0.01

def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    
    scene.ambient_light((0.6, 0.6, 0.6))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(1, 1, 1))
    
    # Set colors based on cube position
    set_cube_colors(mpm.color_with_alpha, mpm.material, mpm.x)
    
    # Render particles
    scene.particles(mpm.x, per_vertex_color=mpm.color_with_alpha, radius=particles_radius)
    
    canvas.scene(scene)

def show_info():
    window.GUI.begin("Info", 0.02, 0.02, 0.35, 0.18)
    window.GUI.text(f"Step: {current_step[None]}/{max_steps}")
    window.GUI.text(f"Particles: {mpm.n_particles[None]}")
    window.GUI.text(f"Drop velocity: {drop_velocity[None]:.2f} m/s")
    window.GUI.text("Auto-looping simulation")
    window.GUI.text("Space: Reset | Up/Down: Velocity | Q: Quit")
    window.GUI.end()

print("Starting cube collision simulation...")
print("Controls: Right-click and drag to rotate camera")
print("Space: Reset | Up/Down arrows: Adjust velocity | Q: Quit")

while window.running:
    # Handle input
    if window.get_event(ti.ui.PRESS):
        if window.event.key == ti.ui.SPACE:
            current_step[None] = 0
            mpm = create_scene()  # Reset scene
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
    
    # Simulate step
    if current_step[None] < max_steps:
        mpm.step(4e-3)  # Time step
        current_step[None] += 1
    else:
        # Auto-restart
        current_step[None] = 0
        mpm = create_scene()
    
    render()
    show_info()
    window.show()

print("Simulation ended")