import os
import torch.backends
import yaml
from tqdm import tqdm
import cv2
import torch
from ultralytics import YOLO
import warnings
import time

warnings.filterwarnings(action='ignore')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

proj_dir = os.path.dirname(os.path.abspath(__file__))
with open(f'{proj_dir}/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# ------------- Config parameters start ------------- #
data_dir = config['data_dir']
preproc_data_dir = config['preproc_data_dir']
model_dir = config['model_dir']
yolo_conf = config['yolo_conf']
categories = config['categories']
clip_len = config['clip_len']
# ------------- Config parameters end ------------- #

def create_video_from_frames(clip_frames, out_fp, fps=15):

    # Read the first frame to get the size (width, height)
    first_frame = cv2.imread(clip_frames[0])
    
    height, width, _ = first_frame.shape
    frame_size = (width, height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_fp, fourcc, fps, frame_size)

    for frame_path in clip_frames:
        frame = cv2.imread(frame_path)
        out.write(frame)

    # Release the video writer
    out.release()

def preprocess_video(yolo_model, video_name, video_fp, out_clip, out_frames, clip_len):

    cap = cv2.VideoCapture(video_fp)
    frame_idx = 0
    saved_frames = []
    target_resolution = (224, 224)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame, verbose=False)
        detections = results[0].boxes.data
        
        if len(detections) > 0:
            for detection in detections:
                if int(detection[5]) == 0:
                    x1, y1, x2, y2 = map(int, detection[:4])
                    cropped_frame = frame[y1:y2, x1:x2]
                    if cropped_frame.size == 0:
                        continue
                    
                    os.makedirs(os.path.join(out_frames, video_name), exist_ok=True)
                    frame_filename = os.path.join(out_frames, video_name, f'frame_{frame_idx:04d}.jpg')

                    cropped_frame = cv2.resize(cropped_frame, target_resolution)
                    cv2.imwrite(frame_filename, cropped_frame)
                    saved_frames.append(frame_filename)
                    frame_idx += 1
                    break

    cap.release()

    if len(saved_frames) >= clip_len:
        clip_idx = 0
        for i in range(0, len(saved_frames)-clip_len+1, clip_len):
            clip_frames = saved_frames[i: i+clip_len]

            os.makedirs(os.path.join(out_clip, video_name), exist_ok=True)
            clip_filename = os.path.join(out_clip, video_name, f'{video_name}_clip_{clip_idx:04d}.avi')
            create_video_from_frames(clip_frames, clip_filename)
            
    return


def main():
    
    yolo_model = YOLO(f'{model_dir}/{yolo_conf}').to(device).eval()
    torch.backends.cudnn.benchmark = True

    for category in categories:
        video_path = os.path.join(data_dir, 'abnormal', category) if category != 'normal' else os.path.join(data_dir, 'normal')
        out_dir = os.path.join(preproc_data_dir, 'abnormal', category) if category != 'normal' else os.path.join(preproc_data_dir, 'normal')

        out_clip, out_frames = os.path.join(out_dir, 'clip'), os.path.join(out_dir, 'frames')

        os.makedirs(out_clip, exist_ok=True)
        os.makedirs(out_frames, exist_ok=True)

        video_files = os.listdir(video_path)
        
        for video_fp in tqdm(video_files, desc='Processing Videos'):
            start_time = time.time()
            video_name = video_fp.split('.')[0]
            preprocess_video(yolo_model, video_name, f'{video_path}/{video_fp}', out_clip, out_frames, clip_len)
            print(f'Time taken for {video_name}: {(time.time() - start_time)/60:.2f} mins')

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps > 0:
        duration = frame_count / fps
        print(fps, duration, frame_count)
    else:
        print(fps, frame_count)

if __name__ == '__main__':

    main()