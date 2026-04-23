"""
This script runs wrist and hand tracking on the Orbita3D and Ability Hand using Mediapipe landmarking libraries.
The landmarks provide a rotation matrix for elbow-to-wrist to wrist-to-hand.
This rotation matrix is then converted to a quaternion command and sent to the Orbita3D.
Further, the landmarks provide angles for the digits of the Ability Hand to move to.
"""

import logging
import argparse
import time
import threading
from orbita_controller import Orbita
from landmarker import Landmarker
from transformer import Transformer
from abh_controller import AbilityHand
from cv2 import CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
state_lock = threading.Lock()
stop_event = threading.Event()
landmarking_result = (None, None)


def hand_task(abh: AbilityHand):
    logger.info('Hand task started')
    global landmarking_result
    last_timestamp = None
    while not stop_event.is_set():
        with state_lock:
            landmarks, timestamp = landmarking_result

        if landmarks is None or timestamp == last_timestamp:
            time.sleep(0.002) # Prevent busy-looping
            continue

        last_timestamp = timestamp

        try:
            abh.update(landmarks)
        except Exception as e:
            logger.error(f"[HAND THREAD ERROR] {e}")
            stop_event.set()
            break

def wrist_task(orbita: Orbita, transformer: Transformer):
    logger.info('Wrist task started')
    global landmarking_result
    last_timestamp = None
    last_quaternion = (0.0, 0.0, 0.0, 1.0)
    while not stop_event.is_set():
        with state_lock:
            landmarks, timestamp = landmarking_result

        if landmarks is None or timestamp == last_timestamp:
            time.sleep(0.002) # Prevent busy-looping
            continue

        last_timestamp = timestamp

        try:
            rotation_matrix = transformer.get_R_from_landmarks(landmarks, mirror_hand=True) # Get rotation matrix from landmarks for wrist
            clamped_quaternion = transformer.get_q_clamped_from_R(rotation_matrix, last_quaternion, orbita.TILT_LIMIT)

            theta_step = transformer.get_theta_between_q(last_quaternion, clamped_quaternion) # Get detected orientation's angle from current
            if theta_step > orbita.MIN_STEP and theta_step < orbita.MAX_STEP:
                orbita.set_orientation(clamped_quaternion) # Send command to wrist if angles are within limits
                last_quaternion = clamped_quaternion
        except Exception as e:
            logger.error(f"[WRIST THREAD ERROR] {e}")
            stop_event.set()
            break

def main():
    parser = argparse.ArgumentParser(description="Run Orbita3D and Ability Hand using Mediapipe")
    parser.add_argument("--disable_wrist", action="store_true", help="Prevents commands from being sent to Orbita")
    parser.add_argument("--disable_hand", action="store_true", help="Prevents command from being sent to Ability Hand")
    parser.add_argument("--use_elbow", action="store_true", help="Will try to detect user's elbow")
    parser.add_argument("--orbita_config", default="./config/default.yaml",help="Path to the Orbita3D YAML config file")
    parser.add_argument("--hand_port", default='cu.usbserial-BG01X7S0', help="Serial port for the Ability Hand (for example /dev/ttyUSB0 on Linux)")
    parser.add_argument("--camera_index", type=int, default=0, help="Camera index to pass to OpenCV")
    args = parser.parse_args()

    global landmarking_result
    stop_event.clear()

    with Orbita(args.orbita_config) as orbita:
        orbita.wake_up()

        with AbilityHand(args.hand_port) as abh:

            with Landmarker(args.camera_index, './task/hand_landmarker.task', './task/pose_landmarker_lite.task', args.use_elbow) as landmarker:
                frame_width = int(landmarker.camera.get(CAP_PROP_FRAME_WIDTH))
                frame_height = int(landmarker.camera.get(CAP_PROP_FRAME_HEIGHT))
                transformer = Transformer(frame_width, frame_height)

                wrist_thread = None
                if not args.disable_wrist:
                    wrist_thread = threading.Thread(
                        target=wrist_task,
                        args=(orbita, transformer),
                        daemon=True
                    )

                hand_thread = None
                if not args.disable_hand:
                    hand_thread = threading.Thread(
                        target=hand_task,
                        args=(abh,),
                        daemon=True
                    )
                
                if wrist_thread: wrist_thread.start()
                if hand_thread: hand_thread.start()

                try:
                    while landmarker.camera.isOpened() and not stop_event.is_set():
                        landmarking_results, timestamp, _ = landmarker.run_detection()

                        if landmarking_results is None: 
                            stop_event.set()
                            break

                        if not landmarking_results:
                            continue

                        with state_lock:
                            landmarking_result = (landmarking_results[0], timestamp) # Just grab the first, since this demo uses 1 setup

                finally:
                    stop_event.set()
                    if wrist_thread: wrist_thread.join(timeout=1.0)
                    if hand_thread: hand_thread.join(timeout=1.0)
                    logger.info('Tasks stopped')

if __name__ == "__main__":
    main()
