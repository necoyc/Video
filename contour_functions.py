import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
from skimage.segmentation import quickshift

FOIL_SIZE = 0.3
MM_FROM_SOURCE = 1
H_DEFAULT = 0.0

def segment_frame(frame):
    '''
    - Алгоритм сегментации MeanShift
    - frame: Исходное изображение в формате Mat-like.
    - spatial_radius: Радиус пространственного окна. Определяет, насколько далеко должны быть пиксели, чтобы считаться частью одного и того же кластера в пространстве расположения.
    - color_radius: Радиус цветового окна. Определяет, насколько далеко в цветовом пространстве должны быть пиксели, чтобы считаться частью одного и того же кластера.
    - min_density: Минимальная плотность кластера. Определяет минимальное количество пикселей в кластере. Кластеры с меньшим количеством пикселей будут объединены с ближайшими кластерами.
    '''
    spatial_radius = 100  
    color_radius = 20    
    min_density = 150    

    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    shifted_frame = cv2.pyrMeanShiftFiltering(lab_frame, spatial_radius, color_radius, min_density)
    
    return shifted_frame
def ksegment_frame(image):
    pixels = image.reshape(-1, 3)

    num_clusters = 2
    kmeans = KMeans(n_clusters=num_clusters, random_state=4)
    kmeans.fit(pixels)
    labels = kmeans.labels_

    desired_colors = np.array([[0, 0, 0], [255, 255, 255]])
    for i in range(num_clusters):
        pixels[labels == i] = desired_colors[i]

    modified_image = pixels.reshape(image.shape)

    return modified_image

def calculate_roi(frame, left_percent, right_percent, top_percent, bottom_percent):
    height, width = frame.shape[:2]

    left_offset = int(width * left_percent / 100)
    right_offset = int(width * right_percent / 100)
    top_offset = int(height * top_percent / 100)
    bottom_offset = int(height * bottom_percent / 100)

    roi_x1 = left_offset
    roi_x2 = width - right_offset
    roi_y1 = top_offset
    roi_y2 = height - bottom_offset

    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]

    return roi
def draw_contours(image, contours, i):
    image_with_contours = image.copy()
    
    cv2.drawContours(image_with_contours, contours, i, (0, 255, 0), 2)
    
    return image_with_contours


import cv2
import numpy as np
def measure_max_depth_test(frame, current_time, scale, default_depth, default_area, offsets):
    top_percent, right_percent, bottom_percent, left_percent = offsets
    
    # Calculate Region of Interest (ROI)
    roi = calculate_roi(frame, left_percent, right_percent, top_percent, bottom_percent)
    
    
    # Apply quickshift segmentation
    final_roi = quickshift(roi, ratio=1.2, kernel_size=5, max_dist=50)
    final_roi_gray = final_roi.astype(np.uint8) 
    
    # Find contours from the final_roi
    contours, _ = cv2.findContours(final_roi_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imwrite(fr"pic\exp1\{MM_FROM_SOURCE} mm\img_{current_time}s_gray_contours.jpg", draw_contours(final_roi_gray, contours, -1))  
    depth = 0.0
    area = 0.0 

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        S = cv2.contourArea(contour)
        
        # Checking if contour is at the top-left corner (assuming depth increases as y increases)
        if y == 0 and x == 0:
            depth = h
            area = S
            break
           
    # If default values are not provided, set them to the first contour's values
    if default_depth is None:
        default_depth = depth
        default_area = area
        
    # Calculate depth and area differences scaled by 'scale'
    depth_diff = (depth - default_depth) * scale
    area_diff = (area - default_area) * (scale ** 2)
    
    return depth_diff, area_diff, default_depth, default_area


def measure_max_depth(frame, current_time, scale, default_depth, default_area, offsets):
    top_percent = offsets[0]
    right_percent = offsets[1]
    bottom_percent = offsets[2]
    left_percent = offsets[3]
    
    #frame = segment_frame(frame)
    roi = calculate_roi(frame, left_percent, right_percent, top_percent, bottom_percent)
    
    k_frame = ksegment_frame(roi)

    gray_roi = cv2.cvtColor(k_frame, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(gray_roi, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    
    #cv2.imwrite(fr"pic\article\img_{current_time}s_roi.jpg", roi)
    #cv2.imwrite(fr"pic\article\img_{current_time}s_gtay_roi.jpg", gray_roi)   
    #cv2.imwrite(fr"pic\exp1\{MM_FROM_SOURCE} mm\img_{current_time}s_gray_contours.jpg", draw_contours(k_frame, contours, -1))   
    #cv2.imwrite(fr"pic\article\img_{current_time}s_countours.jpg", draw_contours(roi, contours, -1))
    
    depth = 0.0
    area = 0.0 

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        S = cv2.contourArea(contour)
        
        if y == 0 and x == 0:
            depth = h
            area = S
            break
           
    if default_depth == None:
        default_depth = depth
        default_area = area
        
    return (depth - default_depth)*scale, (area - default_area)*scale**2, default_depth, default_area

def find_scale(frame, real_size, offsets):
    global H_DEFAULT
    frame_height, _, _ = frame.shape    

    top_percent = offsets[0]
    right_percent = offsets[1]
    bottom_percent = offsets[2]
    left_percent = offsets[3]
    
    roi = calculate_roi(frame, left_percent, right_percent, top_percent, bottom_percent)
    frame_origin = roi.copy()
    frame = segment_frame(roi)
    #frame = ksegment_frame(roi)

    gray_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_roi = cv2.GaussianBlur(gray_roi, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blurred_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    
    
    os.makedirs(r"pic\test1", exist_ok=True)
    #cv2.imwrite(fr"pic\test1\img_scale_origin.jpg", frame_origin)
    #cv2.imwrite(fr"pic\test1\img_scale_meanshift.jpg", frame)
    cv2.imwrite(fr"pic\test1\img_scale_meanshift_contours.jpg", draw_contours(frame, contours, -1))
    #cv2.imwrite(fr"pic\test1\img_scale_countours.jpg", draw_contours(frame_origin, contours, -1))



    x_max = 0
    h_max = 0.0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if x > x_max and h > FOIL_SIZE*frame_height:
            x_max = x
            h_max = h
        
    if h_max == 0:
        edges = cv2.Canny(thresh, 100, 200)
        contours1, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        for contour in contours1:
            x1, y1, w1, h1 = cv2.boundingRect(contour)
            if x1 > x_max and h1 > FOIL_SIZE*frame_height/2:
                x_max = x1
                h_max = h1
                
    if H_DEFAULT == 0.0 and h_max != 0:
        H_DEFAULT = h_max
                
    if h_max == 0:
        print("Error occured, used default value...")
        h_max = H_DEFAULT
    return real_size/float(h_max)