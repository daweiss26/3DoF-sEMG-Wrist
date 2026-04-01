import math, time, struct
from dynamixel_sdk import *

PORT = '/dev/cu.usbserial-B003LPN0'
BAUD = 1000000
PROTOCOL = 1.0
DXL_ID = 70

ADDR_TORQUE_ENABLE   = 58
ADDR_TOP_TARGET      = 59
ADDR_MID_TARGET      = 63
ADDR_BOT_TARGET      = 67
ADDR_TOP_CURRENT     = 71
ADDR_MID_CURRENT     = 75
ADDR_BOT_CURRENT     = 79

port = PortHandler(PORT)
packetHandler = PacketHandler(PROTOCOL)

if not port.openPort(): raise SystemExit("Failed to open port")
if not port.setBaudRate(BAUD): raise SystemExit("Failed to set baud")

dxl_model_number, dxl_comm_result, dxl_error = packetHandler.ping(port, DXL_ID)
print(dxl_model_number, dxl_comm_result, dxl_error)

if dxl_comm_result == COMM_SUCCESS:
    print(f"Found device at ID {DXL_ID}")
    print(f"Model number: {dxl_model_number}")
    print(f"Success Response: {packetHandler.getTxRxResult(dxl_comm_result)}")
else:
    print(f"Error: {packetHandler.getRxPacketError(dxl_error)}")
    print(f"Error Response: {packetHandler.getTxRxResult(dxl_comm_result)}")

def write_u8(addr, val):
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(port, DXL_ID, addr, val)
    if dxl_comm_result != COMM_SUCCESS or dxl_error:
        raise RuntimeError(packetHandler.getTxRxResult(dxl_comm_result) + f" err={dxl_error}")

def write_float(addr, f):
    data = struct.pack('<f', f)
    dxl_comm_result, dxl_error = packetHandler.writeTxRx(port, DXL_ID, addr, len(data), data)
    if dxl_comm_result != COMM_SUCCESS or dxl_error:
        raise RuntimeError(packetHandler.getTxRxResult(dxl_comm_result) + f" err={dxl_error}")

def read_float(addr):
    dxl_data, dxl_comm_result, dxl_error = packetHandler.readTxRx(port, DXL_ID, addr, 4)
    if dxl_comm_result != COMM_SUCCESS or dxl_error:
        raise RuntimeError(packetHandler.getTxRxResult(dxl_comm_result) + f" err={dxl_error}")
    return struct.unpack('<f', bytes(dxl_data))[0]

current_top_pos = read_float(ADDR_TOP_CURRENT)
current_mid_pos = read_float(ADDR_MID_CURRENT)
current_bot_pos = read_float(ADDR_BOT_CURRENT)
print(type(current_top_pos))
print("Top pos (deg):", math.degrees(current_top_pos))
print("Mid pos (deg):", math.degrees(current_mid_pos))
print("Bot pos (deg):", math.degrees(current_bot_pos))
write_float(ADDR_TOP_TARGET, current_top_pos)
write_float(ADDR_MID_TARGET, current_mid_pos)
write_float(ADDR_BOT_TARGET, current_bot_pos)
time.sleep(1)

target_top_pos = current_top_pos + math.radians(400)
target_mid_pos = current_mid_pos + math.radians(400)
target_bot_pos = current_bot_pos + math.radians(400)
print("Moving Top to pos (deg):", math.degrees(target_top_pos))
print("Moving Mid to pos (deg):", math.degrees(target_mid_pos))
print("Moving Bot to pos (deg):", math.degrees(target_bot_pos))
time.sleep(1)
write_u8(ADDR_TORQUE_ENABLE, 1)
time.sleep(1)
write_float(ADDR_TOP_TARGET, target_top_pos)
write_float(ADDR_MID_TARGET, target_mid_pos)
write_float(ADDR_BOT_TARGET, target_bot_pos)

for i in range(5):
    pos = read_float(ADDR_TOP_CURRENT)
    tar = read_float(ADDR_TOP_TARGET)
    print(i+1, "Top pos (deg):", math.degrees(pos))
    print(i+1, "Top tar (deg):", math.degrees(tar))
    time.sleep(1)

write_u8(ADDR_TORQUE_ENABLE, 0)

port.closePort()
