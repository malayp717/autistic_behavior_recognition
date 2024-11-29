import os
import cv2
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO
import warnings
import time

# Ignore warnings
warnings.filterwarnings(action='ignore')

# Set device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load configuration
proj_dir = os.path.dirname(os.path.abspath(__file__))
with open(f'{proj_dir}/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Configuration parameters
data_dir = config['data_dir']
preproc_data_dir = "/hdd/malay/autistic_behavior_recognition/preprocessed_2/"
model_dir = config['model_dir']
yolo_conf = config['yolo_conf']
categories = config['categories']
clip_len = config['clip_len']

def preprocess_video(yolo_model, video_name, video_fp, out_clip, out_frames, clip_len=30):
    cap = cv2.VideoCapture(video_fp)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'{video_name}: {total_frames} frames at {fps:.2f} FPS')

    frame_idx = 0
    saved_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection on the frame
        results = yolo_model(frame, verbose=False)
        detections = results[0].boxes.data

        if len(detections) > 0:
            for detection in detections:
                # Focus on person class (class_id = 0 in COCO dataset)
                if int(detection[5]) == 0:
                    x1, y1, x2, y2 = map(int, detection[:4])

                    # Validate and crop the frame
                    if x2 > x1 and y2 > y1:
                        cropped_frame = frame[y1:y2, x1:x2]

                        if cropped_frame.size != 0:
                            os.makedirs(os.path.join(out_frames, video_name), exist_ok=True)
                            frame_filename = os.path.join(out_frames, video_name, f'frame_{frame_idx:04d}.jpg')
                            cv2.imwrite(frame_filename, cropped_frame)
                            saved_frames.append(frame_filename)
                            frame_idx += 1
                            break

    cap.release()
    print(f'Saved {len(saved_frames)} frames for {video_name}')

    # Segment into 30-frame clips
    if len(saved_frames) >= clip_len:
        clip_idx = 0
        for i in range(0, len(saved_frames), clip_len):
            clip_frames = saved_frames[i:i + clip_len]
            if len(clip_frames) < clip_len:
                break  # Skip if fewer than 30 frames

            os.makedirs(os.path.join(out_clip, video_name), exist_ok=True)
            clip_filename = os.path.join(out_clip, video_name, f'{video_name}_clip_{clip_idx:04d}.avi')
            clip_idx += 1

            # Create video writer
            frame_shape = cv2.imread(clip_frames[0]).shape[:2][::-1]
            out = cv2.VideoWriter(clip_filename, cv2.VideoWriter_fourcc(*'XVID'), fps, frame_shape)

            for frame_path in clip_frames:
                frame = cv2.imread(frame_path)
                out.write(frame)
            out.release()

            print(f'Saved video clip: {clip_filename} with {len(clip_frames)} frames.')

        print(f'Generated {clip_idx} clips for {video_name}')
    else:
        print(f'Not enough frames to create clips for {video_name}')

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

if __name__ == '__main__':
    main()
