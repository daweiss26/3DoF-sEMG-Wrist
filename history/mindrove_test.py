from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
import datetime
import time

BoardShim.enable_dev_board_logger()
params = MindRoveInputParams()
board_id = BoardIds.MINDROVE_WIFI_BOARD
board_shim = BoardShim(board_id, params)

timestamp_channel = BoardShim.get_timestamp_channel(board_id)
emg_channels = BoardShim.get_emg_channels(board_id)
sampling_rate = BoardShim.get_sampling_rate(board_id) # 500 Hz
window_size = 0.25 # seconds
num_points = int(window_size * sampling_rate)

board_shim.prepare_session()
board_shim.start_stream()

while True:
  if board_shim.get_board_data_count() >= num_points:
    data = board_shim.get_board_data(num_points)
    timestamp = data[timestamp_channel][-1]
    # print(f"Packet received at: {datetime.datetime.fromtimestamp(timestamp)}")
    emg_data = data[emg_channels]
    timestamp_data = data[timestamp_channel]
    # print(emg_data)
    print(timestamp_data)


#     scaler_filename = f"{args.model_name}.json"
#     with open(scaler_filename, "w") as f:
#         json.dump({"scale": scale_factor}, f)
#     print(f"Training Complete. Model saved to {args.model_name}.keras")
#     print(f"Scaler param saved to {scaler_filename}")


    # SCALER_FILE = MODEL_FILE.replace(".keras", ".json")
    # print(f"Loading TCN Model: {MODEL_FILE}.keras")
    # with open(SCALER_FILE, "r") as f:
    #     SCALING_FACTOR = json.load(f)["scaler"]
    #     print(f"Loaded Normalization Factor: {SCALING_FACTOR} uV")


    # def get_scaler(X, resume_path=None):
    # """Determines normalization factor"""
    # if resume_path:
    #     # If resuming, we use the scaler from the original model to stay consistent.
    #     scaler_path = resume_path.replace(".keras", ".json")
    #     try:
    #         with open(scaler_path, "r") as f:
    #             scaler = json.load(f)['scaler']
    #             print(f"Loaded existing scaler: {scaler} uV")
    #             return scaler
    #     except FileNotFoundError:
    #         print("Warning: Scalar file for model not found. Recalculating (Risk of Distribution Shift).")
    
    # # Robust 99th percentile scaling to handle outliers
    # scale = np.percentile(np.abs(X), 99)
    # print(f"Calculated new scaler: {scale} uV")
    # return scale