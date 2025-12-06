import cv2
import numpy as np

class LaneKeeping():
    def __init__(self):
        self.HEIGHT_HORIZON = 50
        self.OFFSET_RATIO = 0.65
        self.CHECKPOINT = 70
        self.ratio = 12
        self.calc_speed = 0
    def find_left_right_points(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = (gray*(255/np.max(gray))).astype(np.uint8)
        h, w = gray.shape
        black_image = np.copy(image)
        left_points = []
        right_points = []
        for row in range(int(h/self.ratio)):
            line_row = gray[row*self.ratio, :]
            for x, y in enumerate(line_row):
                if x == 0 and y > 210:
                    break
                if y > 210:
                    min_x = x
                    left_points.append((min_x, row*self.ratio))
                    cv2.circle(black_image, (np.int64(min_x), row*self.ratio), 5, (0,255,255), -1)
                    break
            for x, y in enumerate(reversed(line_row)):
                if x == 0 and y > 210:
                    break
                if y > 210:
                    max_x = w - x
                    right_points.append((max_x, row*self.ratio))
                    cv2.circle(black_image, (np.int64(max_x), row*self.ratio), 5, (255,0,255), -1)
                    break

        left_points = np.array(left_points)
        right_points = np.array(right_points)
        return left_points, right_points, black_image
    
    def find_middle_points(self, image):
        OFFSET = 160
        left_points, right_points, black_image = self.find_left_right_points(image)
        middle_points = list()
        num_points = 5 # number of points
        if(left_points.shape[0] < num_points and right_points.shape[0] < num_points):
            if self.calc_speed <= 20:
                self.calc_speed += 0.1
            else:
                self.calc_speed = 20
        elif(left_points.shape[0] >= num_points and right_points.shape[0] >= num_points):
            if self.calc_speed <= 20:
                self.calc_speed += 0.1
            else:
                self.calc_speed = 20
            # print('Two Lines')
            # Finding left function
            x_lefts = list()
            y_lefts = list()
            for i in left_points:
                x_lefts.append(i[0])
                y_lefts.append(i[1])
            left_func = np.polyfit(x_lefts,y_lefts,2)
            # Finding right function
            x_rights = list()
            y_rights = list()
            for i in right_points:
                x_rights.append(i[0])
                y_rights.append(i[1])
            right_func = np.polyfit(x_rights,y_rights,2)
            for i in range(239, self.HEIGHT_HORIZON, -1):
                ######################################################################
                x_left = np.roots([left_func[0], left_func[1], left_func[2]-i])
                # print(x_left)
                if left_func[0] > 0:
                    x1 = np.min(x_left)
                else:
                    x1 = np.max(x_left)
                    
                ######################################################################
                x_right = np.roots([right_func[0], right_func[1], right_func[2]-i])
                if right_func[0] > 0:
                    x2 = np.max(x_right)
                else:
                    x2 = np.min(x_right)
                if ((np.abs(np.imag(x1)) < 1e-10) > 0 and (np.abs(np.imag(x2)) < 1e-10) > 0):
                    cv2.circle(black_image, (np.int64(x1), i), 5, (255,0,0), -1)
                    cv2.circle(black_image, (np.int64(x2), i), 5, (0,0,255), -1)
                    middle_point = (x1+x2)/2
                    # print(i, x1, x2)
                    cv2.circle(black_image, (np.int64(middle_point), i), 5, (0,255,0), -1)
                    middle_points.append([np.int64(middle_point),i])
        ## Finding Only Right Lane
        elif left_points.shape[0]<num_points:
            if self.calc_speed >= 10:
                self.calc_speed -= 0.5
            else:
                self.calc_speed = 10
            # print('Only Right Lane')
            x_rights = list()
            y_rights = list()
            for i in right_points:
                x_rights.append(i[0])
                y_rights.append(i[1])
            right_func = np.polyfit(x_rights,y_rights,2)
            chosen_right_points = []
            for i in range(239, self.HEIGHT_HORIZON, -1):
                x_right = np.roots([right_func[0], right_func[1], right_func[2]-i])
                # print(x_right)
                if right_func[0] > 0:
                    x2 = np.max(x_right)
                else:
                    x2 = np.min(x_right)
                
                if (np.abs(np.imag(x2)) < 1e-10) > 0:
                    cv2.circle(black_image, (np.int64(x2), i), 5, (0,0,255), -1)
                    #################################################
                    middle_point = np.int64(x2-OFFSET-self.OFFSET_RATIO*(i-240+1))
                    cv2.circle(black_image, (np.int64(middle_point), i), 5, (0,255,0), -1)
                    middle_points.append([np.int64(middle_point),i])
                    if np.int64(x2) < 320:
                        chosen_right_points.append((np.int64(x2), i))
            chosen_right_point = chosen_right_points[int(len(chosen_right_points)/2)]
            cv2.circle(black_image, chosen_right_point, 5, (230,230,230), -1)
            coeff_deriv_right_func = right_func[0]*2*chosen_right_point[0] + right_func[1]
            right_thres_angle = np.arctan(coeff_deriv_right_func)*(180/np.pi)
            # print(right_thres_angle)
            if right_thres_angle <= 35:
                OFFSET += 70
                if right_thres_angle <= 19:
                    OFFSET += 150
                middle_points = list()
                for i in range(239, self.HEIGHT_HORIZON, -1):
                    x_right = np.roots([right_func[0], right_func[1], right_func[2]-i])
                    if right_func[0] > 0:
                        x2 = np.max(x_right)
                    else:
                        x2 = np.min(x_right)
                    if (np.abs(np.imag(x2)) < 1e-10) > 0:
                        #################################################
                        middle_point = np.int64(x2-OFFSET-self.OFFSET_RATIO*(i-240+1))
                        cv2.circle(black_image, (np.int64(middle_point), i), 5, (0,255,128), -1)
                        middle_points.append([np.int64(middle_point),i])
        ## Finding Only Left Lane
        elif right_points.shape[0]<num_points:
            if self.calc_speed >= 10:
                self.calc_speed -= 0.5
            else:
                self.calc_speed = 10
            # print('Only Left Lane')
            x_lefts = list()
            y_lefts = list()
            for i in left_points:
                x_lefts.append(i[0])
                y_lefts.append(i[1])
            left_func = np.polyfit(x_lefts,y_lefts,2)
            chosen_left_points = []
            for i in range(239, self.HEIGHT_HORIZON, -1):
                x_left = np.roots([left_func[0], left_func[1], left_func[2]-i])
                if left_func[0]>0:
                    x1 = np.min(x_left)
                else:
                    x1 = np.max(x_left)
                
                if (np.abs(np.imag(x1)) < 1e-10) > 0:
                    cv2.circle(black_image, (np.int64(x1), i), 5, (255,0,0), -1)
                    ########################################
                    middle_point = np.int64(x1+OFFSET+self.OFFSET_RATIO*(i-240+1))
                    cv2.circle(black_image, (np.int64(middle_point), i), 5, (0,255,0), -1)
                    middle_points.append([np.int64(middle_point),i])
                    if np.int64(x1) < 320:
                        chosen_left_points.append((np.int64(x1), i))
            chosen_left_point = chosen_left_points[int(len(chosen_left_points)/2)]
            cv2.circle(black_image, chosen_left_point, 5, (230,230,230), -1)
            coeff_deriv_left_func = left_func[0]*2*chosen_left_point[0] + left_func[1]
            left_thres_angle = -np.arctan(coeff_deriv_left_func)*(180/np.pi)
            # print(left_thres_angle)
            if left_thres_angle <= 35:
                OFFSET += 70
                if left_thres_angle <= 19:
                    OFFSET += 150
                middle_points = list()
                for i in range(239, self.HEIGHT_HORIZON, -1):
                    x_left = np.roots([left_func[0], left_func[1], left_func[2]-i])
                    if left_func[0] > 0:
                        x1 = np.min(x_left)
                    else:
                        x1 = np.max(x_left)
                    if (np.abs(np.imag(x1)) < 1e-10) > 0:
                        #################################################
                        middle_point = np.int64(x1+OFFSET+self.OFFSET_RATIO*(i-240+1))
                        cv2.circle(black_image, (np.int64(middle_point), i), 5, (0,255,128), -1)
                        middle_points.append([np.int64(middle_point),i])

        return middle_points, black_image

    def AngCal(self, image):
        try:
            middle_points, black_image = self.find_middle_points(image)
            middle_points = np.array(middle_points)
            chosen_middle = self.CHECKPOINT
            angle = 0
            # angle_thres = 3
        
            # # Tìm hàm số bậc 2 fit với tập điểm middle points
            # coeffs = np.polyfit(middle_points[:, 0], middle_points[:, 1], 2)
            # poly_func = np.poly1d(coeffs)
            
            # # Tạo các giá trị x để vẽ đường cong dựa trên hàm số vừa fit
            # x_values = np.linspace(0, width, width)
            # y_values = poly_func(x_values)
            # # x_values = np.linspace(0, width, 1000)
            # # y_values = poly_func(x_values)
            
            # # Vẽ đường cong lên ảnh gốc
            # for i in range(len(x_values) - 1):
            #     pt1 = (int(x_values[i]), int(y_values[i]))
            #     pt2 = (int(x_values[i+1]), int(y_values[i+1]))
            #     cv2.line(black_image, pt1, pt2, (0, 100, 0), 2)  # Màu xanh lá cây (0, 255, 0)    
            
            if len(middle_points)>0:
                ## Only < chosen middle
                if middle_points[0][1] < chosen_middle:
                    x = middle_points[0][0]
                    y = middle_points[0][1]
                    angle = np.int64(np.arctan((159-x)/(239-y))*(180/np.pi))
                    angle = np.int64(angle/2.3)
                    # print(x, y)
                    cv2.circle(black_image, (x, y), 5, (128, 128, 128), -1)
                ## Only > chosen middle
                elif middle_points[-1][1] > chosen_middle:
                    x = middle_points[-1][0]
                    y = middle_points[-1][1]
                    angle = np.int64(np.arctan((159-x)/(239-y))*(180/np.pi))
                    if(y > chosen_middle and y < chosen_middle + 23): # thres 1
                        # print('Thres 1')
                        angle = np.int64(angle/2)
                    elif(y >= chosen_middle + 23 and y < chosen_middle + 23*2): # thres 2
                        # print('Thres 2')
                        angle = np.int64(angle/2)
                    elif(y >= chosen_middle + 23*2 and y < chosen_middle + 23*2 + 22): # thres 3
                        # print('Thres 3')
                        angle = np.int64(angle/2)
                    elif(y >= chosen_middle + 23*2+22 and y < chosen_middle + 23*2 + 22*2): # thres 4
                        # print('Thres 4')
                        angle = np.int64(angle/2)
                    # print(x, y)
                    cv2.circle(black_image, (x, y), 5, (128, 128, 128), -1)
                else:
                    for i in sorted(middle_points, key=lambda x: x[1]):
                        if i[1] == chosen_middle:
                            x = i[0]
                            y = i[1]
                            angle = np.int64(np.arctan((159-x)/(239-y))*(180/np.pi)) 
                            angle = np.int64(angle/2.3)
                            cv2.circle(black_image, (x, y), 5, (128, 128, 128), -1)
                            # print(x, y)
                            break
                    
            # Send Image
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 3
            text_color = (0, 0, 255)  # Màu văn bản: Trắng
            text_position = (10, 20)  # Vị trí của văn bản: Góc trái phía trên
            cv2.putText(black_image, str(float(-angle)), text_position, font, font_scale, text_color, font_thickness)
            return self.calc_speed, float(-angle), black_image
        except:
            return 0, 0.0, image