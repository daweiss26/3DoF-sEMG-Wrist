"""
Uses a trained model specified by MODEL_FILE to take in EMG signal from a Mindrove armband.
The normalized segmented EMG data is then taken in by the CNN, and it predicts an angular velocity.
The angular velocity is then broken down into its axis and theta components to construct a quaternion change.
This change is then added to the current quaternion to obtain the new quaternion.
"""

import sys
import json
import numpy as np
import tensorflow as tf
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from orbita_controller import Orbita
from pyquaternion import Quaternion

BoardShim.enable_dev_board_logger()
PARAMS = MindRoveInputParams()
BOARD_ID = BoardIds.MINDROVE_WIFI_BOARD
mindrove = BoardShim(BOARD_ID, PARAMS)
SAMPLING_RATE = BoardShim.get_sampling_rate(BOARD_ID)
EMG_CHANNELS = BoardShim.get_emg_channels(BOARD_ID)
WINDOW_DURATION = 0.25
REQ_SAMPLES = int(WINDOW_DURATION * SAMPLING_RATE) # 125
try:
    MODEL_FILE = sys.argv[1]
    with open(sys.argv[2], "r") as f:
        SCALING_FACTOR = json.load(f)["scale"]
except IndexError:
    raise SystemExit("Error: Must provide a model file (.keras) and a scaling factor file (.json).")
model = tf.keras.models.load_model(MODEL_FILE)


def integrate_3d_velocity(current_q_tuple, velocity_vector, dt):
    """
    current_q_tuple: (x, y, z, w) from Orbita
    velocity_vector: [wx, wy, wz] from CNN
    dt: time step (e.g., 0.25)
    """
    qx, qy, qz, qw = current_q_tuple
    q_old = Quaternion(w=qw, x=qx, y=qy, z=qz)

    # Only take large enough thetas/magnitudes of rotation
    omega_mag = np.linalg.norm(velocity_vector)
    if omega_mag < 0.05: 
        return current_q_tuple

    # Create q_diff from axis and theta
    axis = velocity_vector / omega_mag
    angle = omega_mag * dt
    q_delta = Quaternion(axis=axis, angle=angle)

    # Get new orientation from prev * diff
    q_new = q_old * q_delta
    return (q_new.x, q_new.y, q_new.z, q_new.w)

def main():
    with Orbita('./config.yaml') as orbita:
        orbita.wake_up()
        mindrove.prepare_session()
        mindrove.start_stream()
        print("3D Control Active. Move hand to control robot.")
        
        # Init Quaternion (x, y, z, w) should be straight up
        cur_o = orbita.get_orientation() 
        q_current = Quaternion(x=cur_o[0], y=cur_o[1], z=cur_o[2], w=cur_o[3])

        try:
            while True:
                if mindrove.get_board_data_count() >= REQ_SAMPLES:
                    data = mindrove.get_board_data(REQ_SAMPLES)
                    emg_window = data[EMG_CHANNELS].T # Shape (125, 8)
                    
                    # The CNN expects 3D input, so we add a batch dimension of 1
                    input_tensor = emg_window.reshape(1, 125, 8)
                    
                    # Normalize the same as training data
                    input_tensor = input_tensor / SCALING_FACTOR
                    
                    # Predicted velocity is [wx, wy, wz]
                    predicted_velocity = model.predict(input_tensor, verbose=0)[0]
                    
                    # Integrate with dt here (exactly WINDOW_DURATION of 0.25s)
                    q_new = integrate_3d_velocity(q_current, predicted_velocity, 0.25)

                    # Orbita's order of rpy: Z (outward) is roll, X (rightward) is pitch, Y (upward) is yaw
                    orbita.set_orientation((q_new.z, q_new.x, q_new.y, q_new.w))

                    q_current = q_new

        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            mindrove.stop_stream()
            mindrove.release_session()
            orbita.go_to_sleep()

if __name__ == "__main__":
    main()
  