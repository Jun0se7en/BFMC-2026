import numpy as np
import matplotlib.pyplot as plt
from extended_kalman_filter import ExtendedKalmanFilter
from kinematic_bycicle_model import KinematicBicycleModel
import pandas as pd
from Transform import Transform
import math
transform = Transform()

df = pd.read_csv('./Retrieving_Data20.csv')
data = df.values

def calculate_gnss_heading(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    # Calculate the difference in longitude
    delta_lon = lon2 - lon1
    # Calculate the initial bearing
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    initial_bearing = math.atan2(x, y)
    # Convert bearing from radians to degrees
    initial_bearing = math.degrees(initial_bearing)
    # Normalize the bearing to 0-360 degrees
    heading = (initial_bearing + 360) % 360
    return heading

# Thiết lập mô phỏng
np.random.seed(0)
length_rear = 1.05  # chiều dài từ trục sau đến trọng tâm, tính bằng mét
delta_time = 0.01  # khoảng thời gian giữa các lần lấy mẫu
ekf = ExtendedKalmanFilter(length_rear, delta_time)
kbm = KinematicBicycleModel(length_rear, delta_time)
# Thông số mô phỏng
sigma_x, sigma_y, sigma_h = 0.001 , 0.001, 0.0001 # Nhiễu đo lường GNSS: 0.5m và 1 độ cho heading

# Ma trận hiệp phương sai ban đầu và nhiễu quá trình
# P_cov = np.eye(3)  # Khởi tạo ma trận hiệp phương sai
P_cov = np.array([
    [0.1, 0, 0],
    [0, 0.1, 0],
    [0, 0, 0.1],
])

Q_cov = np.array([
    [0.1**2,0,0],
    [0,0.1**2,0],
    [0,0, 0.1**2]
])  # Ma trận nhiễu quá trình, với heading tính bằng độ


#---------------------------------------------------------------------------
#GNSS
lat = data[0][1]
lon = data[0][2]
alt = 0.0
target_lla = [lat, lon, alt]
ned = transform.lla_to_ned(target_lla)
#---------------------------------------------------------------------------
#---------------------------init
x, y= ned[1], ned[0]
heading = 0
#--------------------------input
velocity = data[0][3]
angular_velocity = data[0][7]
#---------------------------plot data
measurements = []
predicted_states = []
updated_states=[]
states=[]
#---------------------------------------------------------------------------
prev_lat = lat
prev_lon = lon
#---------------------------------------------------------------------------
for i in range(1, data.shape[0]):
  
    if data[i][4] != 0:
        angular_velocity = data[i][7]
    else :
        angular_velocity = 0

    velocity = data[i][3]
    lat = data[i][1]
    lon = data[i][2]
    alt = 0.0
    predicted_state, predicted_err = ekf.prediction_state(P_cov, Q_cov,x, y, heading, velocity, angular_velocity, 0.01)
    predicted_states.append(predicted_state)

    if (int(lat) != 0 and int(lon) != 0 ) and (lat != prev_lat or lon != prev_lon): 

        # heading_meas = calculate_gnss_heading(prev_lat,prev_lon,lat,lon)
        heading_meas = data[i][5]
        print(heading_meas)
        alt = 0.0 
        target_lla = [lat, lon, alt]
        ned = transform.lla_to_ned(target_lla)
        x_meas = ned[1]
        y_meas = ned[0]
        measurements.append([x_meas,y_meas])
        updated_state, updated_err = ekf.update_state(
        x_meas, y_meas,heading_meas, sigma_x, sigma_y,sigma_h, predicted_state, predicted_err)
        x, y, heading = updated_state
        P_cov = updated_err
        prev_lat = lat
        prev_lon = lon
        updated_states.append(updated_state)
    else:
        x, y, heading = predicted_state
        P_cov = predicted_err
        
# Chuyển đổi dữ liệu để vẽ
predicted_states = np.array(predicted_states)
measurements = np.array(measurements)
updated_states = np.array(updated_states)
# Vẽ đồ thị kết quả
plt.figure(figsize=(10, 8))
plt.plot(predicted_states[:, 0], predicted_states[:, 1], label="Predicted Position", color="red")
plt.plot(updated_states[:, 0], updated_states[:, 1], label="updated_state", color="blue")
plt.plot(measurements[:, 0], measurements[:, 1], label="measurements", color="green")
plt.xlabel("X Position (m)")
plt.ylabel("Y Position (m)")
plt.legend()
plt.title("EKF Position Estimation with GNSS Measurements")
plt.grid()
plt.show()
