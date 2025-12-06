import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.spatial.transform import Rotation as R

class ExtendedKalmanFilter:
    def __init__(self):
        # Trạng thái: [x, y, heading, vx, vy]
        self.state = np.zeros(5)
        
        # Ma trận hiệp phương sai
        self.P = np.eye(5) * 1.0
        
        # Ma trận nhiễu quá trình
        self.Q = np.eye(5)
        self.Q[0, 0] = 0.1  # x position
        self.Q[1, 1] = 0.1  # y position
        self.Q[2, 2] = 0.001   # heading (rad)
        self.Q[3, 3] = 0.1   # vx
        self.Q[4, 4] = 0.1   # vy
        
        # Ma trận nhiễu đo lường cho IMU
        self.R_imu = np.eye(3)
        self.R_imu[0, 0] = 0.1  # ax
        self.R_imu[1, 1] = 0.1  # ay
        self.R_imu[2, 2] = 0.01 # heading rate
        
        # Ma trận nhiễu đo lường cho GPS
        self.R_gps = np.eye(2)
        self.R_gps[0, 0] = 1.0  # x position
        self.R_gps[1, 1] = 1.0  # y position
        
        # Time step
        self.dt_imu = 1/20.0  # IMU: 20Hz
        self.dt_gps = 1.0     # GPS: 1Hz
        
        # Lưu tọa độ gốc cho GPS
        self.origin_lon = None
        self.origin_lat = None
        
        self.g = 9.81  # Gia tốc trọng trường (m/s^2)
        
    def initialize_projection(self, lon, lat):
        self.origin_lon = lon
        self.origin_lat = lat
    
    def gps_to_xy(self, lon, lat):
        if self.origin_lon is None or self.origin_lat is None:
            raise ValueError("Projection not initialized")
        
        # Chuyển đổi đơn giản dựa trên khoảng cách trên mặt đất
        # 1 độ kinh tuyến ≈ 111,320 * cos(vĩ độ) mét
        # 1 độ vĩ tuyến ≈ 110,574 mét
        lat_factor = 110574
        lon_factor = 111320 * math.cos(math.radians(lat))
        
        x = (lon - self.origin_lon) * lon_factor
        y = (lat - self.origin_lat) * lat_factor
        
        return x, y
    
    def predict(self, ax, ay, angleZ, dt=None):
        # Sử dụng dt từ tham số nếu được cung cấp, nếu không sử dụng giá trị mặc định
        if dt is None:
            dt = self.dt_imu
            
        # Chuyển đổi góc từ độ sang radian
        angleZ_rad = np.radians(angleZ)
        
        # Chuyển đổi gia tốc từ g sang m/s^2
        ax_ms2 = ax * self.g
        ay_ms2 = ay * self.g
        
        # Chuyển đổi gia tốc từ IMU frame sang global frame
        cos_heading = np.cos(angleZ_rad)
        sin_heading = np.sin(angleZ_rad)
        
        ax_global = ax_ms2 * cos_heading - ay_ms2 * sin_heading
        ay_global = ax_ms2 * sin_heading + ay_ms2 * cos_heading
        
        # Jacobian của hàm trạng thái
        F = np.eye(5)
        F[0, 3] = dt  # dx/dvx
        F[1, 4] = dt  # dy/dvy
        
        # Cập nhật trạng thái
        # x = x + vx*dt + 0.5*ax*dt^2
        # y = y + vy*dt + 0.5*ay*dt^2
        # heading = heading + wz*dt
        # vx = vx + ax*dt
        # vy = vy + ay*dt
        
        self.state[0] += self.state[3] * dt + 0.5 * ax_global * dt**2
        self.state[1] += self.state[4] * dt + 0.5 * ay_global * dt**2
        self.state[2] = angleZ_rad  # Sử dụng trực tiếp từ IMU
        self.state[3] += ax_global * dt
        self.state[4] += ay_global * dt
        
        # Cập nhật ma trận hiệp phương sai
        self.P = F @ self.P @ F.T + self.Q
        
        return self.state.copy()
    
    def update_gps(self, x_gps, y_gps):
        # Ma trận H cho phép đo GPS
        H = np.zeros((2, 5))
        H[0, 0] = 1.0  # Đo x
        H[1, 1] = 1.0  # Đo y
        
        # Độ lệch giữa phép đo và dự đoán
        y = np.array([x_gps, y_gps]) - self.state[0:2]
        
        # Phần còn lại của thuật toán Kalman
        S = H @ self.P @ H.T + self.R_gps
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Cập nhật trạng thái và hiệp phương sai
        self.state = self.state + K @ y
        self.P = (np.eye(5) - K @ H) @ self.P
        
        return self.state.copy()

def convert_time_to_seconds(time_str):
    """
    Chuyển đổi thời gian từ định dạng 'HH:MM:SS' sang số giây kể từ 00:00:00
    """
    try:
        # Kiểm tra xem time_str đã là số hay chưa
        try:
            return float(time_str)
        except ValueError:
            # Nếu không phải số, thử xử lý như chuỗi thời gian
            time_parts = time_str.split(':')
            if len(time_parts) == 3:
                return float(time_parts[0]) * 3600 + float(time_parts[1]) * 60 + float(time_parts[2])
            else:
                return float(time_str)  # Nếu không phải định dạng HH:MM:SS, coi như đã là số giây
    except Exception as e:
        print(f"Lỗi khi chuyển đổi thời gian '{time_str}': {e}")
        return 0.0

def process_data(file_path):
    # Đọc dữ liệu từ file CSV
    data = pd.read_csv(file_path)
    
    # Kiểm tra tên cột GPS
    gps_columns = []
    if 'Lon(deg)' in data.columns and 'Lat(deg)' in data.columns:
        gps_columns = ['Lon(deg)', 'Lat(deg)']
    elif 'Lon' in data.columns and 'Lat' in data.columns:
        gps_columns = ['Lon', 'Lat']
    else:
        raise ValueError("Không tìm thấy cột GPS. Vui lòng kiểm tra tên cột.")
    
    # Khởi tạo EKF
    ekf = ExtendedKalmanFilter()
    
    # Tìm dòng đầu tiên có GPS
    first_valid_gps = data.dropna(subset=gps_columns).iloc[0]
    ekf.initialize_projection(first_valid_gps[gps_columns[0]], first_valid_gps[gps_columns[1]])
    
    # Khởi tạo mảng để lưu kết quả
    timestamps = []
    ekf_positions = []
    gps_positions = []
    
    last_gps_time = -999.0
    prev_time = None
    
    # Xác định tên cột IMU
    time_col = 'Time(s)' if 'Time(s)' in data.columns else 'Time'
    ax_col = 'ax(g)' if 'ax(g)' in data.columns else 'ax'
    ay_col = 'ay(g)' if 'ay(g)' in data.columns else 'ay'
    az_col = 'az(g)' if 'az(g)' in data.columns else 'az'
    wx_col = 'wx(deg/s)' if 'wx(deg/s)' in data.columns else 'wx'
    wy_col = 'wy(deg/s)' if 'wy(deg/s)' in data.columns else 'wy'
    wz_col = 'wz(deg/s)' if 'wz(deg/s)' in data.columns else 'wz'
    angleX_col = 'AngleX(deg)' if 'AngleX(deg)' in data.columns else 'AngleX'
    angleY_col = 'AngleY(deg)' if 'AngleY(deg)' in data.columns else 'AngleY'
    angleZ_col = 'AngleZ(deg)' if 'AngleZ(deg)' in data.columns else 'AngleZ'
    
    # Lặp qua từng dòng dữ liệu
    for index, row in data.iterrows():
        # Chuyển đổi thời gian sang số giây
        time_str = str(row[time_col])
        current_time = convert_time_to_seconds(time_str)
        
        # Tính dt thực tế giữa các mẫu
        dt = None
        if prev_time is not None:
            dt = current_time - prev_time
            # Nếu dt quá lớn hoặc âm, sử dụng giá trị mặc định
            if dt <= 0 or dt > 1.0:
                dt = ekf.dt_imu
        prev_time = current_time
        
        ax = row[ax_col]
        ay = row[ay_col]
        az = row[az_col]
        wx = row[wx_col]
        wy = row[wy_col]
        wz = row[wz_col]
        angleX = row[angleX_col]
        angleY = row[angleY_col]
        angleZ = row[angleZ_col]
        
        # Bước dự đoán với dữ liệu IMU và dt thực tế
        state = ekf.predict(wx, wy, wz, ax, ay, az, angleX, angleY, angleZ, dt)
        
        # Kiểm tra xem có dữ liệu GPS không và đã đủ 1 giây chưa
        if not pd.isna(row[gps_columns[0]]) and not pd.isna(row[gps_columns[1]]) and (current_time - last_gps_time >= ekf.dt_gps):
            last_gps_time = current_time
            x_gps, y_gps = ekf.gps_to_xy(row[gps_columns[0]], row[gps_columns[1]])
            state = ekf.update_gps(x_gps, y_gps)
            gps_positions.append((x_gps, y_gps))
        
        # Lưu kết quả
        timestamps.append(current_time)
        ekf_positions.append((state[0], state[1]))
    
    return timestamps, ekf_positions, gps_positions, ekf

def plot_results(timestamps, ekf_positions, gps_positions):
    plt.figure(figsize=(12, 8))
    
    # Vẽ quỹ đạo EKF
    ekf_x, ekf_y = zip(*ekf_positions)
    plt.plot(ekf_x, ekf_y, 'b-', label='EKF Trajectory')
    
    # Vẽ điểm GPS
    if gps_positions:
        gps_x, gps_y = zip(*gps_positions)
        plt.scatter(gps_x, gps_y, c='r', marker='o', label='GPS Measurements')
    
    plt.title('Vehicle Trajectory')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('vehicle_trajectory.png')
    plt.show()

def main():
    file_path = 'sensor_data.csv'  # Thay đổi tên file tùy theo dữ liệu thực tế
    
    try:
        timestamps, ekf_positions, gps_positions, ekf = process_data(file_path)
        plot_results(timestamps, ekf_positions, gps_positions)
        
        print(f"Kết quả đã được lưu vào 'vehicle_trajectory.png'")
        
    except Exception as e:
        print(f"Lỗi: {e}")

if __name__ == "__main__":
    main()