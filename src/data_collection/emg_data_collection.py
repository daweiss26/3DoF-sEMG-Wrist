"""
This script collects surface-EMG data and maps it to a corresponding angular rotation of the hand.
The recorded EMG data is segmented into 250ms pieces, and they are provided an angular velocity vector label.
Angular velocity is obtained by taking images of the hand at the beginning and ending of the segment,
and then dividing the the rotation by the time taken to execute it (250ms window).
The data and labels are matched using an empirically determined EMD from emd.py.
Can optionally add data to a "master" dataset.
"""

import os
import argparse
import numpy as np
import bisect
from landmarker import Landmarker
from transformer import Transformer
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from cv2 import CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, Rodrigues

BoardShim.enable_dev_board_logger()
PARAMS = MindRoveInputParams()
BOARD_ID = BoardIds.MINDROVE_WIFI_BOARD
SAMPLING_RATE = BoardShim.get_sampling_rate(BOARD_ID)
TIMESTEP_CHANNELS = BoardShim.get_timestamp_channel(BOARD_ID)
EMG_CHANNELS = BoardShim.get_emg_channels(BOARD_ID)
WINDOW_DURATION = 0.25 # seconds
WINDOW_NUM_SAMPLES = int(WINDOW_DURATION * SAMPLING_RATE) # 125 samples
EMD = 0.083 # seconds (Electromechanical Delay) experimentally determined


def get_closest_rotation(target_time, camera_data):
	"""Finds the rotation matrix closest to the target timestamp."""
	timestamps = [r[0] for r in camera_data]
	idx = bisect.bisect_left(timestamps, target_time)

	if idx == 0: return camera_data[0][1]
	if idx >= len(camera_data): return camera_data[-1][1]
		
	if camera_data[idx][0] - target_time < target_time - camera_data[idx - 1][0]: return camera_data[idx][1]
	else: return camera_data[idx - 1][1]

def process_and_save(all_mindrove_data, all_mediapipe_data, output_file, append_master, map_position):
	"""Process and save normalized labeled data to .npz file."""
	print("Processing data...")
	if not all_mindrove_data:
		print("No EMG data collected.")
		return

	all_mindrove_data = np.concatenate(all_mindrove_data, axis=1)
	emg_timestamps = all_mindrove_data[TIMESTEP_CHANNELS]
	emg_data = all_mindrove_data[EMG_CHANNELS]

	all_emg_windows = []
	all_rotation_labels = []

	curr_time = emg_timestamps[0]
	final_time = emg_timestamps[-1]

	while curr_time + WINDOW_DURATION < final_time:

		# EMG window idxs
		idx_window_start = bisect.bisect_left(emg_timestamps, curr_time)
		idx_window_end = idx_window_start + WINDOW_NUM_SAMPLES

		if idx_window_end > len(emg_timestamps) or idx_window_end <= idx_window_start:
			break

		actual_window_duration = emg_timestamps[idx_window_end-1] - emg_timestamps[idx_window_start]
		emg_window = emg_data[:, idx_window_start:idx_window_end].T # (Samples, 8)
		
		# Calculate R_diff: The rotation needed to go from R_start -> R_end
		# R_end = R_start * R_diff becomes R_diff = R_start.T * R_end
		R_start = get_closest_rotation(curr_time + EMD, all_mediapipe_data)
		R_end = get_closest_rotation(curr_time + EMD + WINDOW_DURATION, all_mediapipe_data)
		R_diff = R_start.T @ R_end
		
		# Convert R_diff to Rotation Vector (Axis * Angle) using Rodrigues
		# rot_vec magnitude is the angle (radians), direction is the axis
		rot_vec, _ = Rodrigues(R_diff)
		rot_vec = rot_vec.flatten() # Rodrigues returns arrays wrapped in an array

		# If enabled, just map EMG to rotation instead of velocity
		rotation_label = rot_vec / actual_window_duration # rad/s vector
		if map_position:
			rotation_label = rot_vec

		# Move onto next window
		all_emg_windows.append(emg_window)
		all_rotation_labels.append(rotation_label)
		curr_time += WINDOW_DURATION

	if all_emg_windows and all_rotation_labels:
		# By normalizing here, we lose raw data values, but standardizes
		scaling_factor = np.percentile(np.abs(all_emg_windows), 99)
		all_emg_windows_norm = all_emg_windows / scaling_factor
		np.savez_compressed(
			output_file, 
			emg=np.array(all_emg_windows_norm), 
			velocity=np.array(all_rotation_labels)
		)
		if append_master:
			merge_sessions(all_emg_windows_norm, all_rotation_labels)

		print(f"Successfully saved {len(all_emg_windows)} samples to {output_file}")
		print(f"Data Stats:")
		print(f"EMG Range: {np.min(all_emg_windows):.4f} to {np.max(all_emg_windows):.4f}")
		print(f"Scaling Factor: {scaling_factor}")
		print(f"Max Velocity: {np.max(np.abs(all_rotation_labels)):.4f} rad/s")
	else:
		print("No valid windows found.")

def merge_sessions(new_emg, new_vel, master_filename="master_dataset.npz"):
	"""Appends new normalized data to the master dataset"""
	if os.path.exists(master_filename):
		print(f"Found existing {master_filename}. Appending...")
		master_data = np.load(master_filename)
		old_emg = master_data['emg']
		old_vel = master_data['velocity']
		
		combined_emg = np.concatenate((old_emg, new_emg), axis=0)
		combined_vel = np.concatenate((old_vel, new_vel), axis=0)
	else:
		print(f"Creating new {master_filename}...")
		combined_emg = new_emg
		combined_vel = new_vel

	np.savez_compressed(master_filename, emg=combined_emg, velocity=combined_vel)
	print(f"Master file updated! Total Samples: {combined_emg.shape[0]}")

def main():
	parser = argparse.ArgumentParser(description="Train or Fine-tune TCN for Orbita3D")
	parser.add_argument("output_file", help="Name for the output model file")
	parser.add_argument("--append_master", action="store_true", help="Add this session to master dataset")
	parser.add_argument("--map_position", action="store_true", help="Map EMG to position")
	args = parser.parse_args()

	save = None
	all_mindrove_data = []
	all_mediapipe_data = []
	mindrove = BoardShim(BOARD_ID, PARAMS)

	with Landmarker(1, './task/hand_landmarker.task', './task/pose_landmarker_lite.task') as landmarker:
		frame_width = int(landmarker.camera.get(CAP_PROP_FRAME_WIDTH))
		frame_height = int(landmarker.camera.get(CAP_PROP_FRAME_HEIGHT))
		transformer = Transformer(frame_width, frame_height)

		mindrove.prepare_session()
		mindrove.start_stream()
		
		print("Recording Data. Press q to stop...")
		while landmarker.camera.isOpened():
			landmarking_results, img_ts_ms, save = landmarker.run_detection()
			if landmarking_results == None: break


			if landmarking_results:
				R = transformer.get_R_from_landmarks(landmarking_results[0])
				all_mediapipe_data.append((img_ts_ms / 1000.0, R))
			
			emg_data = mindrove.get_board_data()
			if emg_data.size > 0:
				all_mindrove_data.append(emg_data)
				
		mindrove.stop_stream()
		mindrove.release_session()

	if save:
		process_and_save(all_mindrove_data,
				   		 all_mediapipe_data,
				   		 args.output_file, 
						 args.append_master, 
						 args.map_position)
		
if __name__ == "__main__":
    main()
