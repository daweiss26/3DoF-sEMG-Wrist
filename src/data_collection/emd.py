"""
This experiment is designed to determine an estimate for the 
electromechanical delay (EMD)between forearm surface-EMG signal and movement.
A 10 second recording of Mindrove (sEMG) and Mediapipe (position) data is taken.
The hand/wrist starts in a resting position, and then an impulse movement is made.
Both time-series datasets are then searched for the index with the largest change in magnitude.
This indexes corresponds with timestamps that can then be compared.
"""

from landmarker import Landmarker
from transformer import Transformer
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from cv2 import CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT
import time
import datetime

BoardShim.enable_dev_board_logger()
PARAMS = MindRoveInputParams()
BOARD_ID = BoardIds.MINDROVE_WIFI_BOARD
mindrove = BoardShim(BOARD_ID, PARAMS)
TIMESTEP_CHANNEL = BoardShim.get_timestamp_channel(BOARD_ID)
EMG_CHANNELS = BoardShim.get_emg_channels(BOARD_ID)


def main():
	def get_datetime(time):
		return datetime.datetime.fromtimestamp(time)

	def get_emg_start(emg_data, timestamp_data):
		diffs = transformer.get_distance(emg_data, l2=False, transpose=True)
		max_diff, max_i = transformer.get_max(diffs)
		print(f"EMG impulse started between: {get_datetime(timestamp_data[max_i])} and {get_datetime(timestamp_data[max_i+1])}")
		print(f"Change was: {max_diff} (i: {max_i})")
		return (timestamp_data[max_i] + timestamp_data[max_i+1])/2

	def get_movement_start(hand_vectors):
		max_theta = 0
		max_i = None
		for i in range(len(hand_vectors)-1):
			theta = transformer.get_theta_from_unit_vector(hand_vectors[i][0], hand_vectors[i+1][0])
			if theta > max_theta:
				max_theta = theta
				max_i = i
		print(f"Movement started between: {get_datetime(hand_vectors[max_i][1]/1000)} and {get_datetime(hand_vectors[max_i+1][1]/1000)}")
		print(f"Theta was: {max_theta} (i: {max_i})")
		return (hand_vectors[max_i][1] + hand_vectors[max_i+1][1])/2000

	# Might need to adjust camera index depending on setup
	with Landmarker(1, './task/hand_landmarker.task', './task/pose_landmarker_lite.task') as landmarker:
		frame_width = int(landmarker.camera.get(CAP_PROP_FRAME_WIDTH))
		frame_height = int(landmarker.camera.get(CAP_PROP_FRAME_HEIGHT))
		transformer = Transformer(frame_width, frame_height)

		hand_vectors = []
		mindrove.prepare_session()
		mindrove.start_stream()
		start_time = time.time()

		# Collect position data
		while landmarker.camera.isOpened() and time.time() - start_time < 5: # Record for 10 sec
			landmarking_results, timestamp_img = landmarker.run_hand_detection()
			if landmarking_results:
				wrist_to_hand_unit = transformer.get_hand_vector(landmarking_results[0])
				hand_vectors.append((wrist_to_hand_unit, timestamp_img))
		
		# Collect EMG data
		data = mindrove.get_current_board_data(2500) # 500 Hz for 5 sec
		emg_data = data[EMG_CHANNELS]
		timestamp_data = data[TIMESTEP_CHANNEL]
		mindrove.stop_stream()
		mindrove.release_session()

		# Determine timestamp of greatest change
		if len(emg_data):
			emg_start = get_emg_start(emg_data, timestamp_data)
		if hand_vectors:
			movement_start = get_movement_start(hand_vectors)
		print(f"Time difference between movement and muscle signal was: {movement_start - emg_start} (sec)")

if __name__ == "__main__":
    main()
