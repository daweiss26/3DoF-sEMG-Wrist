# Orbita3D Setup Instructions

---

## Prerequisites

You need the following installed on your system:

* **Rust:** Used for compiling the kinematics SDK
* **Python (v3.12 Preferred):** Runs the controller

*Note: These instructions use python and pip commands. You may need to use python3 or pip3 depending on your setup.*

---

## Virtual Environment

It is recommended to create a virtual environment to store compiled packages

*Note: If using PowerShell, you may need to run ```
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process``` first if activation is disabled.*

Create virtual environment (.venv recommended: Other environment names may require file changes such as run_camera_demo.bat)
```
python -m venv .venv
```

Activate virtual environment
```
.venv\Scripts\activate // Windows
source .venv/bin/activate // Mac/Linux
```

---

## Installation

Install the dependencies in the requirements.txt file:
```
pip install -r requirements.txt
```

Then install the Orbita3D package. Inside ```orbita3d_control/orbita3d_c_api```, run:
```
pip install -e .
```

---

## Firmware

Upload the Orbitarduinamixel project to your Generic STM32G4 Series device. Custom-installed libraries should be included in the project.

You may need to add the following .json file for additional boards in the Board Manager:
```
https://github.com/stm32duino/BoardManagerFiles/raw/main/package_stmicroelectronics_index.json
```

The libraries installed:
* **[Simple FOC](https://docs.simplefoc.com/):** by Simplefoc
* **[SimpleFOCDrivers](https://docs.simplefoc.com/):** by Simplefoc
* **[Dynamixel2Arduino](https://github.com/ROBOTIS-GIT/dynamixel2arduino):** by ROBOTIS
* **[Bolder Flight Systems BMI088](https://github.com/bolderflight/bmi088-arduino):** by Brian Taylor

---

## Use

*If using VSCode (recommended), ensure the project contains a ```.vscode``` file with:*
```
{
    "python-envs.defaultEnvManager": "ms-python.python:venv",
    "python.terminal.useEnvFile": true
}
```

If using Windows: running ```run_camera_demo.bat``` should work. *Note: You may need to change the specific virtual environment listed in the .bat file if you are using something different from .venv*

If using Mac/Linux: simply run a demo script from the project root directory such as ```python src/orbita_abh_camera.py```

*Note: If using a demo script, you will need to change the Ability Hand port based upon your device setup*

* **orbita_abh_camera.py:** Uses camera landmarks
* **orbita_emg.py:** Uses muscular signal and a residual TCN model
* **orbita_emg_krr.py:** Uses muscular signal and a kernel ridge regression model
* **task:** Contains landmarker tasks
* **config:** Contains .yaml files for Orbita3D setup. *Contains the serial port to use, which is determined by your specific device.*
* **orbita3d_control:** Contains kinematics for Orbita3D control.
* **firmware:** Contains Arduino code for Orbita3D
* **src/controller:** Contains controller classes for Orbita3D and Ability Hand
* **src/data_collection:** Contains tools for determining electromechanical delay and collecting data
* **src/train:** Contains tools for training muscular signal decoders using collected data
* **src/util:** Contains utilities such as the pose landmarker and coordinate transformer classes

---

## Sources

* **[Orbita3D](https://github.com/pollen-robotics/Orbita_3d_R1):** Original actuator design
* **[Orbita3D Control](https://github.com/pollen-robotics/orbita3d_control):** Generates the kinematics SDK
* **[Rustypot](https://github.com/pollen-robotics/rustypot):** Used to configure servos
* **[Ability Hand API](https://github.com/psyonicinc/ability-hand-api):** Reference on controlling PSYONIC Ability Hand
