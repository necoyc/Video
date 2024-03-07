import cv2
import numpy as np
import os
from tqdm import tqdm

from data_functions import process_data
from contour_functions import find_scale, measure_max_depth

def process_video(video_path, t_begin, t_end, f0, B, real_size, i, offsets_for_scale, offsets_for_contours):
    cap = cv2.VideoCapture(video_path)
    
    default_depth, default_area = None, None
    
    time_values = []
    depth_values = []
    area_values = []

    ret, frame = cap.read()
    scale = find_scale(frame, real_size, offsets_for_scale) 
    
    print(f"scale: {scale}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for _ in tqdm(range(i)):
        
        depth_values.append([])
        area_values.append([])
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_skip = int(fps / f0)

            if current_time < t_begin:
                continue
            elif current_time > t_end:
                break

            if _ == 0:
                time_values.append(current_time)
            depth, area, default_depth, default_area = measure_max_depth(frame, current_time, scale, default_depth, default_area, offsets_for_contours)
            depth_values[_].append(depth)
            area_values[_].append(area)
            
            frame_skip = int(f0 + np.exp( B * (current_time - t_begin)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_skip)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cap.release()
    
    return time_values, depth_values, area_values

    
if __name__ == "__main__":
    video_paths = ["mov\MVI_8531.MOV"]
    save_folder = r"data\article";
    os.makedirs(save_folder, exist_ok=True)

    offsets_for_scale = [0, 0, 3, 10]#top, right, bottom, left
    offsets_for_contours = [5, 25, 50, 25]#top, right, bottom, left

    t_begin = 55.0  #начальный момент времени
    t_end = 55.5    #конечный момент времени
    f0 = 25.0        #f0 - начальная частота
    B = 3*0        #B - коэффициент в экспоненте
    real_size = 15.0  #в миллиметрах 
    i = 1  #количество итераций для одного видео

    for video_path in video_paths:
        filename = video_path[-12:-4]
        print(f"Filename: {filename}")
        times, depths, areas = process_video(video_path, t_begin, t_end, f0, B, real_size, i, offsets_for_scale, offsets_for_contours)
        process_data(times, depths, os.path.join(save_folder, f"{filename}"), f"{filename}_Depth", "Time, s", "h, mm")
        process_data(times, areas, os.path.join(save_folder, f"{filename}"), f"{filename}_Area", "Time, s", "a, mm^2")