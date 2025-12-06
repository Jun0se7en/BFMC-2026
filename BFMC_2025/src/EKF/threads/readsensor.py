import serial
import struct

SERIAL_PORT = "/dev/ttyUSB0"
BAUDRATE = 115200
bias = [0.0, 0.0, 0.0]
filt = [0.0, 0.0, 0.0]
calib_idx = 0
CALIB_SAMPLES = 200
ALPHA = 0.135
DEADBAND_G = 0.02

current = {}
DISTANCE_THRESHOLD = 8

ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)

def parse_data(data):
    if len(data) >= 11 and data[0] == 0x55:
        data_type = data[1]

        if data_type == 0x51:  # Acceleration
            ax, ay, az = struct.unpack('<hhh', data[2:8])
            ax, ay, az = [x / 32768.0 * 16 for x in (ax, ay, az)]
            return 'ACC', (ax, ay, az)

        elif data_type == 0x52:  # Gyroscope
            gx, gy, gz = struct.unpack('<hhh', data[2:8])
            gx, gy, gz = [x / 32768.0 * 2000 for x in (gx, gy, gz)]
            return 'GYRO', (gx, gy, gz)

        elif data_type == 0x53:  # Euler angles (Roll, Pitch, Yaw)
            roll, pitch, yaw = struct.unpack('<hhh', data[2:8])
            roll, pitch, yaw = [x / 32768.0 * 180 for x in (roll, pitch, yaw)]
            return 'ANGLE', (roll, pitch, yaw)

        elif data_type == 0x57:  # GPS Longitude and Latitude
            lon, lat = struct.unpack('<ii', data[2:10])
            lon, lat = lon / 1e7, lat / 1e7
            return 'GPS', (lat, lon)

        elif data_type == 0x58:  # GPS Height and Speed
            height, speed = struct.unpack('<hh', data[2:6])
            height = height / 10.0
            speed = speed / 1000.0 * 3.6
            return 'GPS_SPEED', (height, speed)

    return None, None

if __name__ == "__main__":

    try:
        buffer = bytearray()
        while True:
            byte = ser.read()
            if byte:
                buffer += byte

                if buffer[0] != 0x55:
                    buffer.pop(0)
                    continue

                if len(buffer) >= 11:
                    sensor_type, values = parse_data(buffer[:11])
                    if sensor_type:
                        if sensor_type == 'ACC':
                            if calib_idx < CALIB_SAMPLES:
                                for i in range(3):
                                    bias[i] += values[i]
                                calib_idx += 1
                                if calib_idx == CALIB_SAMPLES:
                                    bias[:] = [b / CALIB_SAMPLES for b in bias]
                                    print(f"Bias Calibrated: {bias}")
                                continue
                            
                            acc_corr = [values[i] - bias[i] for i in range(3)]

                            # ---- (2) IIR low-pass ----
                            for i in range(3):
                                filt[i] = ALPHA*acc_corr[i] + (1-ALPHA)*filt[i]

                            # ---- (3) Dead-band ----
                            acc_out = [0.0 if abs(f) < DEADBAND_G else f for f in filt]
                            
                            ###### REMOVED ########
                            # ---- (4) Scale ----
                            acc_out = [f * 9.81 for f in acc_out]
                            # ---- (5) Convert to pixel/s^2 ----
                            acc_out = [f * 135/2 for f in acc_out]
                            print(f"Acceleration [g]: {acc_out}")
                        elif sensor_type == 'GYRO':
                            print(f"Gyroscope [°/s]: {values}")
                        elif sensor_type == 'ANGLE':
                            print(f"Euler angles [°]: {values}")
                        elif sensor_type == 'GPS':
                            print(f"GPS Latitude, Longitude [°]: {values}")
                        elif sensor_type == 'GPS_SPEED':
                            print(f"GPS Altitude [m], Speed [km/h]: {values}")
                    buffer = buffer[11:]

    except KeyboardInterrupt:
        print("Dừng chương trình")
    finally:
        ser.close()
