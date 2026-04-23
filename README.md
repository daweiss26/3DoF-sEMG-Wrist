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
Or, in your terminal, ```".:./src:./src/controller:./src/util"``` to your PYTHONPATH environment variable, such as with ```export PYTHONPATH=".:./src:./src/controller:./src/util:$PYTHONPATH"``` for Mac/Linux. For Windows, ```run_camera_demo.bat``` should take care of this.

If using Windows: running ```run_camera_demo.bat``` should work. *Note: You may need to change the specific virtual environment listed in the .bat file if you are using something different from .venv*

If using Mac/Linux: simply run a demo script from the project root directory such as ```python src/orbita_abh_camera.py```

*Note: If using a demo script, you will need to change the Ability Hand serial port in ```src/orbita_abh_camera.py``` and the Orbita3D serial port in ```config/default.yaml``` based upon your device setup. You will likely also need to alter the camera index in ```src/orbita_abh_camera.py``` depending on your setup.*

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

---

## Docker

You can also run the camera demo through Docker instead of creating a local Python environment. The Docker image installs the Python dependencies from `requirements.txt`, builds and installs the local `orbita3d_control/orbita3d_c_api` package, and starts `src/orbita_abh_camera.py`.

### Build the image

From the project root, run:
```
docker build -t orbita-camera-demo .
```

### Run the container

The script needs access to your camera and, if you are controlling hardware, your serial devices. It also opens an OpenCV display window, so on Linux you will typically need to share your X11 socket. A common Linux example is:

```
docker run --rm -it \
  --device /dev/video0:/dev/video0 \
  --device /dev/ttyUSB0:/dev/ttyUSB0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  orbita-camera-demo \
  --camera_index 0 \
  --hand_port /dev/ttyUSB0 \
  --orbita_config ./config/default.yaml
```

### Notes

* Update `config/default.yaml` so its `serial_port` matches the Orbita device inside the container, such as `/dev/ttyUSB1` or `/dev/serial/by-id/...`.
* The Ability Hand port is now configurable through `--hand_port`, which is helpful because Linux device names differ from macOS names such as `/dev/cu.usbserial-*`.
* If you only want to test the vision pipeline without connected hardware, you can add `--disable_wrist --disable_hand`.
* If your camera is exposed under a different device path or index, adjust `--device` and `--camera_index` accordingly.
* On macOS or Windows, Docker Desktop does not pass host cameras and serial devices into Linux containers as directly as native Linux does. For full hardware access, running the container on a Linux host is the simplest path.

### Pass different script arguments

Anything placed after the image name is forwarded to `src/orbita_abh_camera.py`. For example:

```
docker run --rm -it \
  --device /dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  orbita-camera-demo \
  --disable_wrist --disable_hand --camera_index 0 --use_elbow
```
