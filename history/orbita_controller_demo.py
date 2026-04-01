import time
import numpy as np
import math
from orbita3d import Orbita3dController


# orbita = Orbita3dController.from_config("/Users/dweiss/Documents/Grad School/Thesis/Controller/fake.yaml")
orbita = Orbita3dController.from_config("/Users/dweiss/Documents/Grad School/Thesis/Controller/config.yaml")

orbita.enable_torque(True)

time.sleep(3)
orbita.set_target_rpy_orientation((0.0, 0.0, 0.0))
time.sleep(1)

# Do a simple sinusoidal movement
t0 = time.time()
while time.time() - t0 < 30:
    roll = np.deg2rad(15) * np.sin(2 * np.pi * 0.3 * time.time())
    pitch = np.deg2rad(15) * np.cos(2 * np.pi * 0.3 * time.time())
    yaw = np.deg2rad(30) * np.sin(2 * np.pi * 0.3 * time.time())
    # orbita.set_target_velocity()
    orbita.set_target_rpy_orientation((roll, pitch, yaw))
    time.sleep(0.001)

# print("Current Orientation:", orbita.get_current_orientation())
# print("Target Orientation:", orbita.get_target_rpy_orientation())
# print("Target RPY Orientation:", orbita.get_target_rpy_orientation())

orbita.set_target_rpy_orientation((math.radians(0.0), math.radians(0.0), math.radians(0.0)))
time.sleep(3)
orbita.disable_torque()
