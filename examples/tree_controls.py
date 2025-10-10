"""
Material Property Controller for Tree Wind Simulations.

This module provides an interactive control system for real-time adjustment
of material properties and simulation parameters using Taichi GGUI sliders.

Usage:
    from tree_controls import MaterialPropertyController

    # Initialize controller
    controller = MaterialPropertyController()
    controller.init_controls()

    # In render loop
    controller.render_controls(window)

    # Access current values
    E_wood = controller.E_wood[None]
    wind = controller.wind_strength[None]
"""

import taichi as ti


class MaterialPropertyController:
    """
    Interactive controller for material properties and simulation parameters.

    Provides Taichi fields and GUI sliders for real-time adjustment of:
    - Material stiffness (Young's modulus E)
    - Lamé parameters (mu, lambda)
    - Environmental forces (wind, gravity)
    - Numerical parameters (damping)
    """

    def __init__(self):
        """Initialize Taichi fields for all controllable properties."""
        # Material properties - Wood (trunk/branches)
        self.E_wood = ti.field(ti.f32, shape=())
        self.mu_wood = ti.field(ti.f32, shape=())
        self.la_wood = ti.field(ti.f32, shape=())

        # Material properties - Leaves
        self.E_leaf = ti.field(ti.f32, shape=())
        self.mu_leaf = ti.field(ti.f32, shape=())
        self.la_leaf = ti.field(ti.f32, shape=())

        # Environmental forces
        self.wind_strength = ti.field(ti.f32, shape=())
        self.gravity = ti.field(ti.f32, shape=())

        # Numerical parameters
        self.damping_factor = ti.field(ti.f32, shape=())

        # UI state
        self.show_advanced = False
        self.needs_reset = False

    def init_controls(self, preset="normal"):
        """
        Initialize control values with a preset configuration.

        Args:
            preset: One of "soft", "normal", "stiff", "very_stiff"
        """
        presets = {
            "soft": {
                "E_wood": 100.0,
                "E_leaf": 0.5,
                "wind_strength": 0.5,
                "gravity": 10.0,
                "damping_factor": 0.995
            },
            "normal": {
                "E_wood": 1000.0,
                "E_leaf": 0.5,
                "wind_strength": 0.0,
                "gravity": 10.0,
                "damping_factor": 0.995
            },
            "stiff": {
                "E_wood": 5000.0,
                "E_leaf": 0.5,
                "wind_strength": 0.0,
                "gravity": 10.0,
                "damping_factor": 0.995
            },
            "very_stiff": {
                "E_wood": 8000.0,
                "E_leaf": 0.5,
                "wind_strength": 0.0,
                "gravity": 10.0,
                "damping_factor": 0.995
            }
        }

        config = presets.get(preset, presets["normal"])

        # Set material properties
        self.E_wood[None] = config["E_wood"]
        self.mu_wood[None] = config["E_wood"]  # Simplified: mu = E
        self.la_wood[None] = config["E_wood"]  # Simplified: lambda = E

        self.E_leaf[None] = config["E_leaf"]
        self.mu_leaf[None] = config["E_leaf"]
        self.la_leaf[None] = config["E_leaf"]

        # Set environmental parameters
        self.wind_strength[None] = config["wind_strength"]
        self.gravity[None] = config["gravity"]
        self.damping_factor[None] = config["damping_factor"]

    def render_controls(self, window):
        """
        Render interactive GUI controls for adjusting properties.

        Args:
            window: Taichi GGUI window instance

        Returns:
            bool: True if any slider was changed (may need simulation update)
        """
        changed = False

        # Material Controls Panel
        window.GUI.begin("Material Controls", 0.62, 0.02, 0.36, 0.50)

        # Preset buttons
        window.GUI.text("Presets:")
        if window.GUI.button("Soft"):
            self.init_controls("soft")
            self.needs_reset = True
            changed = True

        window.GUI.text(" ")  # Spacing
        if window.GUI.button("Normal"):
            self.init_controls("normal")
            self.needs_reset = True
            changed = True

        window.GUI.text(" ")  # Spacing
        if window.GUI.button("Stiff"):
            self.init_controls("stiff")
            self.needs_reset = True
            changed = True

        window.GUI.text(" ")  # Spacing
        if window.GUI.button("Very Stiff"):
            self.init_controls("very_stiff")
            self.needs_reset = True
            changed = True

        window.GUI.text("")  # Separator
        window.GUI.text("Material Properties:")

        # Wood stiffness slider
        new_E_wood = window.GUI.slider_float(
            "Wood E",
            self.E_wood[None],
            10.0,
            10000.0
        )
        if abs(new_E_wood - self.E_wood[None]) > 0.1:
            self.E_wood[None] = new_E_wood
            self.mu_wood[None] = new_E_wood  # Keep mu = E
            self.la_wood[None] = new_E_wood  # Keep lambda = E
            changed = True

        # Leaf stiffness slider
        new_E_leaf = window.GUI.slider_float(
            "Leaf E",
            self.E_leaf[None],
            0.1,
            10.0
        )
        if abs(new_E_leaf - self.E_leaf[None]) > 0.01:
            self.E_leaf[None] = new_E_leaf
            self.mu_leaf[None] = new_E_leaf
            self.la_leaf[None] = new_E_leaf
            changed = True

        window.GUI.text("")  # Separator
        window.GUI.text("Environment:")

        # Wind strength slider
        new_wind = window.GUI.slider_float(
            "Wind",
            self.wind_strength[None],
            0.0,
            5.0
        )
        if abs(new_wind - self.wind_strength[None]) > 0.01:
            self.wind_strength[None] = new_wind
            changed = True

        # Gravity slider
        new_gravity = window.GUI.slider_float(
            "Gravity",
            self.gravity[None],
            0.0,
            20.0
        )
        if abs(new_gravity - self.gravity[None]) > 0.1:
            self.gravity[None] = new_gravity
            changed = True

        window.GUI.text("")  # Separator
        window.GUI.text("Simulation:")

        # Damping slider
        new_damping = window.GUI.slider_float(
            "Damping",
            self.damping_factor[None],
            0.90,
            1.0
        )
        if abs(new_damping - self.damping_factor[None]) > 0.001:
            self.damping_factor[None] = new_damping
            changed = True

        window.GUI.text("")  # Separator
        window.GUI.text("Controls:")
        window.GUI.text("Presets: Quick property sets")
        window.GUI.text("Sliders: Fine-tune values")
        window.GUI.text("Arrows: Adjust wind")

        window.GUI.end()

        return changed

    def check_and_clear_reset_flag(self):
        """
        Check if a reset is needed and clear the flag.

        Returns:
            bool: True if reset was requested by preset button
        """
        if self.needs_reset:
            self.needs_reset = False
            return True
        return False

    def get_wood_properties(self):
        """
        Get current wood material properties.

        Returns:
            tuple: (E, mu, lambda) for wood
        """
        return (
            self.E_wood[None],
            self.mu_wood[None],
            self.la_wood[None]
        )

    def get_leaf_properties(self):
        """
        Get current leaf material properties.

        Returns:
            tuple: (E, mu, lambda) for leaves
        """
        return (
            self.E_leaf[None],
            self.mu_leaf[None],
            self.la_leaf[None]
        )

    def get_environment(self):
        """
        Get current environmental parameters.

        Returns:
            tuple: (wind_strength, gravity)
        """
        return (
            self.wind_strength[None],
            self.gravity[None]
        )

    def get_damping(self):
        """
        Get current damping factor.

        Returns:
            float: damping_factor
        """
        return self.damping_factor[None]
