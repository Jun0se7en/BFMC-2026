import numpy as np
import os 
import cv2

class ImagePreprocessing():
    def __init__(self, opt):
        self.opt = opt

    def get_sobel_image(self, sobel_image, threshold):
        abs_sobel_image = np.absolute(sobel_image)
        scaled_sobelx = np.uint8(255*abs_sobel_image/np.max(abs_sobel_image))
        binary = np.zeros_like(scaled_sobelx)
        binary[(scaled_sobelx >= threshold[0]) 
                & (scaled_sobelx <= threshold[1])] = 255
        
        return binary

    def process_image(self, frame):
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        bgr_image = np.copy(frame)
        
        bgr_image = self.region_of_interest(bgr_image)
        red_channel = bgr_image[:,:,2]
        hls = np.float64(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HLS))
        l_channel = hls[:,:,1]

        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize = 3)
        sxbinary = self.get_sobel_image(sobelx, self.opt["sobel_x_thres"])

        r_binary = np.zeros_like(red_channel)
        r_binary[(red_channel >= self.opt["red_thres"][0]) \
                & (red_channel <= self.opt["red_thres"][1])] = 255
        combined_binary = np.zeros_like(sxbinary)
        grayImg = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        grayImg[grayImg >= 17] = 255
        adaptive = cv2.adaptiveThreshold(grayImg, 255,\
                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, self.opt["adaptive_block_size"], self.opt["adaptive_offset"])
        combined_binary[((adaptive == 255) & (sxbinary == 255)) 
                        | ((sxbinary == 255) & (r_binary == 255))] = 255
        
        new_sxbinary = self.region_of_interest(np.where((combined_binary == 255) & (adaptive == 255), 255, combined_binary))

        
        new_sxbinary = cv2.dilate(new_sxbinary, \
                                np.ones((self.opt["dilate_kernel"], self.opt["dilate_kernel"]), np.uint8)) 

        new_sxbinary = cv2.erode(new_sxbinary, \
                                np.ones((self.opt["dilate_kernel"], self.opt["dilate_kernel"]), np.uint8)) 
        

        return new_sxbinary, grayImg
        
    def region_of_interest(self, frame):
        height = frame.shape[0]
        width = frame.shape[1]
        mask = np.zeros_like(frame)
        region_of_interest_vertices = np.array([[   (0, height),
                                                    (width, height),
                                                    (width, height - 20),
                                                    (width - 70, 0.4 * height),
                                                    (70, 0.4 * height),
                                                    (0, height - 20)]], np.int32)
        cv2.fillPoly(mask, region_of_interest_vertices, 255)
        masked_image = cv2.bitwise_and(frame, mask)
        return masked_image
    
    def process_image2(self, frame):
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        bgr_image = np.copy(frame)
        hls = np.float64(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HLS))
        l_channel = hls[:,:,1]
        
        sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize = 3)
        sybinary = self.get_sobel_image(sobely, [110, 250])
        return sybinary