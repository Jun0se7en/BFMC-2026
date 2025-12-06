#!/usr/bin/env python3
# wtgahrs2_reader_filtered.py

import serial, struct, json, time
from collections import deque

# ========= Thông số người dùng chỉnh =========
PORT      = '/dev/ttyUSB0'
BAUD      = 230400
TIMEOUT   = 0.1

CALIB_SAMPLES = 200       # số mẫu đầu để lấy bias
ALPHA         = 0.135     # thông thấp IIR (≈5 Hz khi t ≈10 ms)
DEADBAND_G    = 0.02      # nếu |a| < ngưỡng → đặt = 0

# =============================================
MAXBUF = 2048

# ---------- Giải mã --------------------------
def parse_acc(data):
    ax, ay, az, T = struct.unpack('<hhhH', data)
    return ax, ay, az, T / 100.0

def parse_gyro(data):
    wx, wy, wz, T = struct.unpack('<hhhH', data)
    scale = 2000 / 32768.0
    return {'wx_dps': wx*scale, 'wy_dps': wy*scale,
            'wz_dps': wz*scale, 'temp_c': T/100.0}

def parse_angle(data):
    roll, pitch, yaw, ver = struct.unpack('<hhhH', data)
    scale = 180 / 32768.0
    return {'roll_deg': roll*scale, 'pitch_deg': pitch*scale,
            'yaw_deg': yaw*scale, 'ver': ver}

PARSERS = {0x51: 'acc', 0x52: 'gyro', 0x53: 'angle'}

# --------- Generator đọc khung ---------------
def iter_frames(ser):
    buf = deque(maxlen=MAXBUF)
    while True:
        buf.extend(ser.read(64))
        while len(buf) >= 11:
            if buf[0] != 0x55:
                buf.popleft(); continue
            fid = buf[1]
            if fid not in PARSERS:
                buf.popleft(); continue
            if len(buf) < 11: break
            frame = [buf.popleft() for _ in range(11)]
            if (sum(frame[:10]) & 0xFF) != frame[10]:
                buf.popleft(); continue          # CRC fail
            yield fid, bytes(frame[2:10])

# -------------- Main -------------------------
def main():
    ser = serial.Serial(PORT, BAUD, timeout=TIMEOUT)
    print(f'Opened {PORT} @ {BAUD}')

    # State biến cho acc
    bias = [0.0, 0.0, 0.0]
    filt = [0.0, 0.0, 0.0]
    calib_idx = 0

    current = {}
    t_print = time.time()

    try:
        for fid, payload in iter_frames(ser):

            tag = PARSERS[fid]

            # ========== Gia tốc ==========
            if tag == 'acc':
                ax_raw, ay_raw, az_raw, temp = parse_acc(payload)

                # scale sang g
                scale = 16 / 32768.0
                acc_g = [ax_raw*scale, ay_raw*scale, az_raw*scale]

                # ---- (1) Hiệu chuẩn bias ----
                if calib_idx < CALIB_SAMPLES:
                    for i in range(3):
                        bias[i] += acc_g[i]
                    calib_idx += 1
                    if calib_idx == CALIB_SAMPLES:
                        bias[:] = [b / CALIB_SAMPLES for b in bias]
                        print(f'Bias calibrated: {bias}')
                    continue    # bỏ ghi log trong giai đoạn calib

                acc_corr = [acc_g[i] - bias[i] for i in range(3)]

                # ---- (2) IIR low-pass ----
                for i in range(3):
                    filt[i] = ALPHA*acc_corr[i] + (1-ALPHA)*filt[i]

                # ---- (3) Dead-band ----
                acc_out = [0.0 if abs(f) < DEADBAND_G else f for f in filt]

                current['acc'] = {'ax_g': acc_out[0],
                                  'ay_g': acc_out[1],
                                  'az_g': acc_out[2],
                                  'temp_c': temp}

            # ========== Gyro / Angle ==========
            elif tag == 'gyro':
                current['gyro'] = parse_gyro(payload)
            elif tag == 'angle':
                current['angle'] = parse_angle(payload)

            # -------- In log mỗi 0,1 s -------
            if time.time() - t_print > 0.1:
                t_print = time.time()
                print(json.dumps(current, ensure_ascii=False, indent=None))

    except KeyboardInterrupt:
        print('Ctrl-C – exit')
    finally:
        ser.close()

if __name__ == '__main__':
    main()
