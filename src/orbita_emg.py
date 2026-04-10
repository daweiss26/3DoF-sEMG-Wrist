"""
Uses a trained TCN model specified by MODEL_FILE to take in EMG signal from a Mindrove armband.
The normalized segmented EMG data is then taken in by the model, and it predicts an angular velocity.
The angular velocity is then broken down into its axis and theta components to construct a quaternion change.
This change is then added to the current quaternion to obtain the new quaternion.
The model takes in 250ms of data with a sliding window of 50ms.
The model also normalizes the data with an exponential moving average before predictions.
"""

import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
from transformer import Transformer
from adaptive_scaler import AdaptiveScaler
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from orbita_controller import Orbita
from pyquaternion import Quaternion
from quaternion_visualizer import QuaternionVisualizer

BoardShim.enable_dev_board_logger()
PARAMS = MindRoveInputParams()
BOARD_ID = BoardIds.MINDROVE_WIFI_BOARD
SAMPLING_RATE = BoardShim.get_sampling_rate(BOARD_ID)
EMG_CHANNELS = BoardShim.get_emg_channels(BOARD_ID)
WINDOW_DURATION = 0.25 # seconds
UPDATE_INTERVAL = 0.05 # seconds (20 Hz)
REQ_SAMPLES = int(WINDOW_DURATION * SAMPLING_RATE) # 125 samples from 0.25s window @ 500 Hz


def get_3d_position(q_current, prediction):
    """Returns the updated quaternion only if it exceeds the deadzone"""
    theta = np.linalg.norm(prediction)
    if theta > 0.05:
        R_delta, _ = cv2.Rodrigues(prediction)
        q_delta = Quaternion(matrix=R_delta)
        return q_current * q_delta
    else:
        return q_current

def integrate_3d_velocity(current_q_tuple, velocity_vector, dt):
    """Integrates angular velocity [wx, wy, wz] into quaternion orientation"""
    qx, qy, qz, qw = current_q_tuple
    q_current = Quaternion(w=qw, x=qx, y=qy, z=qz)

    # Deadzone for stability
    omega_mag = np.linalg.norm(velocity_vector)
    theta = omega_mag * dt
    if theta < 0.15:
        return current_q_tuple

    # Create delta quaternion
    axis = velocity_vector / omega_mag
    q_delta = Quaternion(axis=axis, radians=theta)

    # Apply rotation (Local Frame)
    return q_current * q_delta

def main():
    parser = argparse.ArgumentParser(description="Run Orbita3D using EMG")
    parser.add_argument("model_name", help="Path of the model file")
    parser.add_argument("--map_position", action="store_true", help="Map EMG to position")
    parser.add_argument("--simulate", action="store_true", help="Plot to a 3D visualizer instead of moving Orbita")
    args = parser.parse_args()

    MODEL_FILE = args.model_name
    model = tf.keras.models.load_model(MODEL_FILE)
    mindrove = BoardShim(BOARD_ID, PARAMS)

    # Calibration period
    print("Preparing...")
    time.sleep(3)
    mindrove.prepare_session()
    mindrove.start_stream()
    print("CALIBRATING: Please squeeze your hand as hard as possible for 5 seconds...")
    time.sleep(5)
    scaling_factor = np.percentile(np.abs(
        mindrove.get_current_board_data(2500)[EMG_CHANNELS]
    ), 99)
    mindrove.stop_stream()
    mindrove.release_session()
    print(f"Using {scaling_factor} as scaling factor")
    adaptive_scaler = AdaptiveScaler(initial_scale=scaling_factor)

    visualizer = None
    if args.simulate:
        print("SIMULATION MODE ACTIVE: Orbita will not move.")
        visualizer = QuaternionVisualizer()

    with Orbita('./config/default.yaml') as orbita:
        orbita.wake_up()
        mindrove.prepare_session()
        mindrove.start_stream()
        transformer = Transformer(0, 0)
        print("Control Active")
        
        # Initial Orientation
        cur_o = orbita.get_orientation() 
        # Orbita's order of rpy: Z (outward) is roll, X (rightward) is pitch, Y (upward) is yaw
        q_current = Quaternion(x=cur_o[1], y=cur_o[2], z=cur_o[0], w=cur_o[3])

        try:
            while True:
                loop_start = time.time()

                # We need exactly 125 samples for the TCN
                if mindrove.get_board_data_count() >= REQ_SAMPLES:
                    data = mindrove.get_current_board_data(REQ_SAMPLES) 
                    
                    # (1, 125, 8)
                    emg_window = data[EMG_CHANNELS].T
                    emg_window_norm = adaptive_scaler.update_and_normalize(emg_window)
                    input_tensor = emg_window_norm.reshape(1, REQ_SAMPLES, 8)
                    
                    # Predict
                    prediction = model.predict(input_tensor, verbose=0)[0]
                    
                    # Integrate
                    q_new = None
                    if args.map_position:
                        q_new = get_3d_position(q_current, prediction)
                    else:
                        q_new = integrate_3d_velocity(
                            (q_current.x, q_current.y, q_current.z, q_current.w), 
                            prediction, 
                            UPDATE_INTERVAL
                        )

                    q_clamped = transformer.get_q_clamped(q_new, q_current, orbita.TILT_LIMIT)
                    theta_step = transformer.get_theta_between_q(q_current, q_clamped) # Get detected orientation's angle from current
                    if theta_step > orbita.MIN_STEP and theta_step < orbita.MAX_STEP:
                        if args.simulate:
                            visualizer.update(q_clamped)
                        else:
                            # Orbita's order of rpy: Z (outward) is roll, X (rightward) is pitch, Y (upward) is yaw
                            orbita.set_orientation((q_clamped.z, q_clamped.x, q_clamped.y, q_clamped.w))
                        q_current = q_clamped

                elapsed = time.time() - loop_start
                sleep_time = max(0, UPDATE_INTERVAL - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            mindrove.stop_stream()
            mindrove.release_session()
            orbita.go_to_sleep()

if __name__ == "__main__":
    main()
