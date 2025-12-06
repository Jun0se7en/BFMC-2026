#!/usr/bin/env python3
"""
WTGAHRS2 – Calibration & Reference helper
Author : chatGPT demo – 2025-06-22
"""

import serial, time, sys, argparse

# ==== Hằng số cấu hình WITMotion ====
UNLOCK_CMD  = bytes([0xFF, 0xAA, 0x69, 0x88, 0xB5])
SAVE_CMD    = bytes([0xFF, 0xAA, 0x00, 0x00, 0x00])
CALSW_REG   = 0x01          # thanh ghi hiệu chuẩn

# Giá trị DATA_L cho CALSW
CAL_EXIT        = 0x00
CAL_ACC_GYRO    = 0x01
CAL_ALT_ZERO    = 0x03      # NEW – reset height
CAL_YAW_RESET   = 0x04      # NEW – reset yaw (Z-axis angle)
CAL_MAG         = 0x02

def pkt(reg, val):      # đóng gói 5 byte
    return bytes([0xFF, 0xAA, reg, val & 0xFF, (val >> 8) & 0xFF])

def send(port, data, note=""):
    port.write(data); port.flush()
    print(f"⮞ {note}: {[hex(b) for b in data]}")

# ---------- Chế độ cũ ----------
def calib_acc_gyro(port):
    print("➡️  Hiệu chuẩn Acc+Gyro – giữ module NẰM NGANG & bất động…")
    send(port, pkt(CALSW_REG, CAL_ACC_GYRO), "CALSW=0x01")
    time.sleep(4)
    finish(port)

def calib_mag(port):
    print("➡️  Hiệu chuẩn Từ kế – xoay chậm 360° quanh mọi trục…")
    send(port, pkt(CALSW_REG, CAL_MAG), "CALSW=0x02")
    input(" Nhấn <Enter> khi hoàn tất xoay…")
    finish(port)

# ---------- Chế độ MỚI ----------
def reset_yaw(port):
    print("➡️  Đặt Yaw = 0 ° tại tư thế hiện tại (trục Z)…")
    send(port, pkt(CALSW_REG, CAL_YAW_RESET), "CALSW=0x04")
    finish(port, save=False)      # chỉ gốc tham chiếu tạm thời

def reset_altitude(port):
    print("➡️  Đặt độ cao gốc (Height = 0 m) – đặt cảm biến tại mốc tham chiếu…")
    send(port, pkt(CALSW_REG, CAL_ALT_ZERO), "CALSW=0x03")
    finish(port, save=False)      # giá trị reset không cần lưu Flash

def finish(port, save=True):
    send(port, pkt(CALSW_REG, CAL_EXIT), "CALSW=0x00 (exit)")
    if save:
        send(port, SAVE_CMD, "SAVE")
    print("✅  Hoàn tất")

def main():
    parser = argparse.ArgumentParser(description="WTGAHRS2 calibration tool")
    parser.add_argument("-p", "--port", default="/dev/ttyUSB0")
    parser.add_argument("-b", "--baud", type=int, default=115200)
    parser.add_argument("--mode", choices=["acc", "mag", "yaw", "alt", "calib"],
                        required=True, help="acc | mag | yaw | alt | calib")
    args = parser.parse_args()

    try:
        with serial.Serial(args.port, args.baud, timeout=0.5) as ser:
            send(ser, UNLOCK_CMD, "UNLOCK")
            time.sleep(0.1)

            if   args.mode == "acc":  calib_acc_gyro(ser)
            elif args.mode == "mag":  calib_mag(ser)
            elif args.mode == "yaw":  reset_yaw(ser)
            elif args.mode == "alt":  reset_altitude(ser)
            elif args.mode == "calib":
                calib_acc_gyro(ser)
                time.sleep(2)
                reset_yaw(ser)
                time.sleep(2)
                reset_altitude(ser)

    except serial.SerialException as e:
        sys.exit(f"Serial error: {e}")

if __name__ == "__main__":
    main()
