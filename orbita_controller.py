"""Controller for the Orbita3D Wrist Actuator"""

import time
import numpy as np
from orbita3d import Orbita3dController


class Orbita:
    TILT_LIMIT = np.deg2rad(15)
    MAX_STEP = np.deg2rad(60)
    MIN_STEP = np.deg2rad(3)
    HOMED_LIMIT = 0.001

    def __init__(self, config):
        try:
            self.controller = Orbita3dController.from_config(config)
        except Exception as e:
            print(f'Error initializing ({e}): Using fake config')
            self.controller = Orbita3dController.from_config('./fake.yaml')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.go_to_sleep()

    def wake_up(self, reset=False):
        """Enables torque"""
        self.controller.enable_torque(reset)
        self.set_rpy_orientation(0.0, 0.0, 0.0)
        time.sleep(0.1)

    def stretch(self, duration=2):
        """Does a simple sinusoidal movement"""
        duration = max(duration, 2)
        t0 = time.time()
        while time.time() - t0 <= duration:
            modulation = min((time.time()-t0), 1) * min(duration-(time.time()-t0), 1)
            roll = np.deg2rad(30) * np.sin(2 * np.pi * (1/duration) * (time.time()-t0)) * modulation
            pitch = np.deg2rad(30) * np.cos(2 * np.pi * (1/duration) * (time.time()-t0)) * modulation
            self.controller.set_target_rpy_orientation((roll, pitch, 0))
            time.sleep(0.001)
        time.sleep(0.5)

    def dance(self, duration=7):
        """Does a little jig"""
        duration = max(duration, 7)
        t0 = time.time()
        while time.time() - t0 <= duration:
            roll = np.deg2rad(10) * np.sin(4 * np.pi * (time.time()-t0))
            pitch = np.deg2rad(10) * np.sin(4 * np.pi * (time.time()-t0))
            yaw = (np.deg2rad(180) * -np.cos(np.pi * (1/duration) * (time.time()-t0))) + np.deg2rad(180)
            self.controller.set_target_rpy_orientation((roll, pitch, yaw))
            time.sleep(0.001)
        time.sleep(0.5)

    def clamp_rpy(self, roll, pitch, yaw, tilt_limit=TILT_LIMIT):
        """
        Reduces roll and pitch magnitude according to the TILT_LIMIT, maintains roll and pitch angle
        Flag is set as final return value as to whether rotation was clamped
        """
        theta_tilt = np.sqrt(roll**2 + pitch**2)
        if theta_tilt <= tilt_limit:
            return roll, pitch, yaw, False
        scale = tilt_limit / theta_tilt
        return roll * scale, pitch * scale, yaw, True

    def set_rpy_orientation(self, roll: float, pitch: float, yaw: float):
        """Sets the clamped roll, pitch, and yaw radian values of the orientation vector such that they cannot exceed the predefined limit"""
        clamped_roll, clamped_pitch, yaw, _ = self.clamp_rpy(roll, pitch, yaw)
        self.controller.set_target_rpy_orientation((clamped_roll, clamped_pitch, yaw))

    def get_rpy_orientation(self):
        """Gets the intrinsic Euler angles representing the end-effector orientation (roll, pitch, yaw)"""
        return self.controller.get_current_rpy_orientation()

    def set_orientation(self, orientation: tuple):
        """Clamps the quaternion representing the end-effector orientation (qx (Roll), qy (Pitch), qz (Yaw), qw)"""
        self.controller.set_target_orientation(orientation)

    def get_orientation(self):
        """Gets the quaternion representing the end-effector orientation (qx (Roll), qy (Pitch), qz (Yaw), qw)"""
        return self.controller.get_current_orientation()

    def home(self):
        """Return to the home (initial) position"""
        # current_yaw = self.get_rpy_orientation()[2]
        # self.controller.set_target_rpy_orientation((0, 0, current_yaw))
        self.controller.set_target_rpy_orientation((0, 0, 0))
        homing = True
        while homing:
            time.sleep(0.5)
            current_rpy_orientation = self.controller.get_current_rpy_orientation()
            if abs(current_rpy_orientation[0]) < self.HOMED_LIMIT and abs(current_rpy_orientation[1]) < self.HOMED_LIMIT:
                homing = False
                time.sleep(0.5)

    def go_to_sleep(self):
        """Homes the device and disables torque"""
        self.home()
        self.controller.disable_torque()
