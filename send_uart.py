import serial
import struct
import time

# ================== CONFIG ==================
UART_PORT = "COM3"          # Linux: "/dev/ttyUSB0"
UART_BAUD = 115200


# ================== CHECKSUM ==================
def calc_checksum(action: int) -> int:
    return action ^ 0xFF

# ================== PACKET ==================
def pack_command(action: int) -> bytes:
    checksum = calc_checksum(action)
    return struct.pack("BB", action, checksum)

# ================== UART SEND ==================
class ESP32Commander:
    def __init__(self, port, baudrate):
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate
          
        )
        print("[OK] UART connected to ESP32")
        time.sleep(0.2)  # chờ ESP32 ổn định

    def send(self, action: int):
        packet = pack_command(action)
        self.ser.write(packet)
        self.ser.flush()

    def close(self):
        self.ser.close()

# # ================== MAIN ==================
# if __name__ == "__main__":
#     esp = ESP32Commander(UART_PORT, UART_BAUD)

#     try:
#         while True:
#             esp.send(0x01)     # ví dụ: lệnh START
#             time.sleep(0.5)

#             esp.send(0x02)     # ví dụ: lệnh STOP
#             time.sleep(0.5)

#     except KeyboardInterrupt:
#         esp.close()
#         print("Stopped")
