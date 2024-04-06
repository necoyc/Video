import cv2
import numpy as np
import os
from tqdm import tqdm

from data_functions import process_data
from contour_functions import find_scale, measure_max_depth, measure_max_depth_test

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
            depth, area, default_depth, default_area = measure_max_depth_test(frame, current_time, scale, default_depth, default_area, offsets_for_contours)
            depth_values[_].append(depth)
            area_values[_].append(area)
            
            frame_skip = int(f0 + np.exp( B * (current_time - t_begin)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_skip)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cap.release()
    
    return time_values, depth_values, area_values

    
if __name__ == "__main__":
    directory_path = rf"mov\exp1\1 mm"
    os.makedirs(directory_path, exist_ok=True)
    video_paths = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.MOV'):
                video_paths.append(os.path.join(root, file))
    save_folder = rf"data\exp1\1 mm";
    os.makedirs(save_folder, exist_ok=True)

    offsets_for_scale = [0, 15, 30, 40]#top, right, bottom, left
    offsets_for_contours = [0, 40, 75, 40]#top, right, bottom, left

    t_begin = 0.5  #начальный момент времени
    t_end = 15.0    #конечный момент времени
    f0 = 10.0        #f0 - начальная частота
    B = 3*0        #B - коэффициент в экспоненте
    real_size = 15.0  #в миллиметрах 
    i = 1  #количество итераций для одного видео

    #TODO: граница излучателя определить и исключить 
    for video_path in video_paths:
        filename = video_path[-12:-4]
        print(f"Filename: {filename}")
        times, depths, areas = process_video(video_path, t_begin, t_end, f0, B, real_size, i, offsets_for_scale, offsets_for_contours)
        process_data(times, depths, os.path.join(save_folder, f"{filename}"), f"{filename}_Depth", r'$\mathit{Время}$ $\mathit{T_{э}}$, $\mathdefault{с}$', r'$\mathit{Глубина}$ $\mathit{эрозии}$ $\mathit{L_{э}}$, $\mathdefault{мм}$')
        process_data(times, areas, os.path.join(save_folder, f"{filename}"), f"{filename}_Area", r'$\mathit{Время}$ $\mathit{T_{э}}$, $\mathdefault{с}$',  r'$\mathit{Площадь}$ $\mathit{эрозии}$ $\mathit{S_{э}}$, $\mathdefault{мм^2}$')