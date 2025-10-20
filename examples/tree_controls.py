"""
Material Property Controller for Tree Wind Simulations.

This module provides an interactive control system for real-time adjustment
of material properties and simulation parameters using Taichi GGUI sliders.

Usage:
    from tree_controls import MaterialPropertyController, RecordingManager

    # Initialize controller
    controller = MaterialPropertyController()
    controller.init_controls()

    # Initialize recording
    recorder = RecordingManager()

    # In render loop
    controller.render_controls(window)
    recorder.render_ui(window)
    if recorder.is_recording[None]:
        recorder.capture_frame(window)

    # Access current values
    E_wood = controller.E_wood[None]
    wind = controller.wind_strength[None]
"""

import taichi as ti
import os
from datetime import datetime


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
        self.reset_requested = False

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
        window.GUI.begin("Material Controls", 0.62, 0.02, 0.36, 0.45)

        # Reset button at top
        window.GUI.text("Simulation Control:")
        if window.GUI.button("Reset Simulation"):
            self.reset_requested = True
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
        window.GUI.text("Reset: Restart from step 0")
        window.GUI.text("Sliders: Adjust properties live")
        window.GUI.text("Arrows: Quick wind adjust")

        window.GUI.end()

        return changed

    def check_and_clear_reset_flag(self):
        """
        Check if a reset is needed and clear the flag.

        Returns:
            bool: True if reset was requested
        """
        if self.reset_requested:
            self.reset_requested = False
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


class RecordingManager:
    """
    Video recording manager for Taichi simulations.

    Captures frames from the simulation window and creates MP4 videos
    at 10 fps using Taichi's VideoManager.
    """

    def __init__(self):
        """Initialize recording manager."""
        self.is_recording = ti.field(ti.i32, shape=())
        self.frame_count = 0
        self.sim_frame_count = 0  # Track simulation frames for skipping
        self.video_manager = None
        self.output_dir = None
        self.is_recording[None] = 0

        # Recording settings
        self.max_frames = 30  # Capture exactly 30 frames (3 seconds at 10 fps)
        self.frame_skip = 6  # Capture every 6th simulation frame (60fps sim -> 10fps video)

    def start_recording(self):
        """
        Start recording video.
        Will capture exactly max_frames frames (30 frames = 3 seconds at 10 fps).
        """
        print(f"[RecordingManager] start_recording() called - will capture {self.max_frames} frames")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"recordings/recording_{timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[RecordingManager] Created directory: {self.output_dir}")

        # Initialize Taichi VideoManager for 10 fps output
        self.video_manager = ti.tools.VideoManager(
            output_dir=self.output_dir,
            framerate=10,
            automatic_build=False
        )
        print("[RecordingManager] VideoManager initialized")

        self.is_recording[None] = 1
        self.frame_count = 0
        self.sim_frame_count = 0
        print(f"[RecordingManager] Recording started - will auto-stop after {self.max_frames} frames")

    def capture_frame(self, window):
        """
        Capture current frame from window with frame skipping.
        Only captures every Nth simulation frame to match target fps.
        Auto-stops after max_frames captured.

        Args:
            window: Taichi GGUI window instance
        """
        if self.is_recording[None]:
            self.sim_frame_count += 1

            # Skip frames to match target fps (capture every 6th frame)
            if self.sim_frame_count % self.frame_skip != 0:
                return

            # Check if we've captured enough frames
            if self.frame_count >= self.max_frames:
                print(f"[RecordingManager] Reached {self.max_frames} frames, auto-stopping...")
                self.stop_recording()
                return

            try:
                if self.frame_count % 10 == 0 or self.frame_count == 0:
                    print(f"[RecordingManager] Capturing frame {self.frame_count + 1}/{self.max_frames}...")

                img = window.get_image_buffer_as_numpy()
                self.video_manager.write_frame(img)
                self.frame_count += 1

                if self.frame_count == 1:
                    print(f"[RecordingManager] First frame captured successfully")
                elif self.frame_count == self.max_frames:
                    print(f"[RecordingManager] Final frame ({self.max_frames}) captured")

            except Exception as e:
                print(f"[RecordingManager ERROR] Failed to capture frame: {e}")
                import traceback
                traceback.print_exc()

    def stop_recording(self):
        """
        Stop recording and create video file.

        Returns:
            str: Path to output directory
        """
        if self.is_recording[None]:
            print(f"Finalizing video with {self.frame_count} frames...")
            self.video_manager.make_video(mp4=True, gif=False)
            self.is_recording[None] = 0

            video_path = os.path.join(self.output_dir, "video.mp4")
            print(f"✓ Video saved: {video_path}")
            print(f"  Frames: {self.frame_count}")
            print(f"  Duration: {self.frame_count / 10:.1f}s at 10fps")

            return self.output_dir
        return None

    def render_ui(self, window):
        """
        Render recording UI controls with stable layout.
        Shows progress during recording (Frame X/30).

        Args:
            window: Taichi GGUI window instance
        """
        try:
            # Recording panel below material controls
            window.GUI.begin("Recording", 0.62, 0.48, 0.36, 0.18)

            # Status line - changes based on recording state
            if self.is_recording[None]:
                window.GUI.text(f"● RECORDING {self.frame_count}/{self.max_frames}")
            else:
                window.GUI.text(f"Capture {self.max_frames} frames (3s @ 10fps)")

            # Always show frame count and duration
            window.GUI.text(f"Frames: {self.frame_count}/{self.max_frames}")
            window.GUI.text(f"Duration: {self.frame_count / 10:.1f}s / {self.max_frames / 10:.1f}s")
            window.GUI.text("")

            # Single button that toggles - stays in same position
            if self.is_recording[None]:
                # Show progress in button text
                button_text = f"Recording... {self.frame_count}/{self.max_frames}"
                if window.GUI.button(button_text):
                    print("[RecordingManager] Stop Recording button clicked")
                    self.stop_recording()
            else:
                if window.GUI.button("Start Recording"):
                    print("[RecordingManager] Start Recording button clicked")
                    self.start_recording()

            window.GUI.end()
        except Exception as e:
            print(f"[RecordingManager ERROR] Failed to render UI: {e}")
            import traceback
            traceback.print_exc()
