# Orbita3D Setup Instructions

---

## 1. Prerequisites

You need the following installed on your system:

* **Rust:** Used for compiling the kinematics SDK
* **Python (v3.9+ Preferred):** Runs the controller
> **Note:** You will need to numpy, mediapipe, and maturin via pip

---

## 2. Installation

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
