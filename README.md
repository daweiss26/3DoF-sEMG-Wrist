# Orbita3D Setup Instructions

---

## 1. Prerequisites

You need the following installed on your system:

* **Rust:** Used for compiling the kinematics SDK
* **Python (v3.9+ Preferred):** Runs the controller
> **Note:** You will need to numpy, mediapipe, and maturin via pip

---

## 2. Installation

Git clone orbita3d_control
In orabita3d_c_api/python/orbita3d/__init__.py, add
    def get_current_rpy_orientation(self) -> Tuple[float, float, float, float]:
        """Get the current rpy orientation of the end-effector.

        Returns:
            The intrinsic Euler representing the end-effector orientation (roll, pitch, yaw).
        """
        rpy = ffi.new("double(*)[3]")
        check(lib.orbita3d_get_current_rpy_orientation(self.uid, rpy))
        return tuple(rpy[0])
In orbita3d_controller/Cargo.toml lines 10-11, swap to build_dynamixel
In orbita3d_controller/src/lib.rs lines 258, remove !self.is_torque_on()? from if statement in enabled_torque
In orbita3d_controller/src/io/poulpe.ts and dynamixel_serial.rs, replace DynamixelSerialIO with DynamixelProtocolHandler, replace device::orbita3d_poulpe with servo::orbita::orbita3d_foc (orbita3d_poulpe -> orbita3d_foc) (MotorValue -> DiskValue), created a 3xf64 target position cache to return because the get_target_position was messed up so it just returned the target_cache, set target_cache on set_target_position(_fb)
Specifically, in dynamixel_serial.rs, in parse_config_file, use the right serial device, is_torque_on should just be returning that read_torque_enable is not equal to 0, read_current_position should be read_present_position, write_target_position should be write_goal_position
Navigate to the orbita3d_c_api directory and run the following command

```bash
pip install --verbose -e .
```

---

## 3. Use

* **orbita.py:** Used to run the project
* **orbita_controller.py:** Creates the Orbita class that is used to issue commands and read data from the device
* **landmarker.py:** Generates landmarks of the hand and elbow provided a live stream video
* **transformer.py:** Calculates the transformation matrix, rpy angles, and quaternions to rotate to the captured wrist angle

---

## 4. Sources

* **[Rustypot](https://github.com/pollen-robotics/rustypot):** Used to configure servos
* **[Orbita3D Control](https://github.com/pollen-robotics/orbita3d_control):** Generates the kinematics SDK
