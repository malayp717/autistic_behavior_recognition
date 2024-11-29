import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x / 255.0),
                ])

class VideoDataset(Dataset):
    def __init__(self, video_files, labels, desc, clip_len, min_clip_len, transform=None):
        self.video_files = video_files
        self.labels = labels
        self.clip_len = clip_len
        self.min_clip_len = min_clip_len
        self.transform = transform
        self.desc = desc

    def __len__(self):
        return len(self.video_files)
    
    def getbadclips(self):

        cnt = 0

        for idx in range(len(self.video_files)):
            video_path = self.video_files[idx]
            label = self.labels[idx]

            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
            cap.release()

            if len(frames) < self.min_clip_len:
                print(video_path)
                cnt += 1
            
            return cnt

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]

        if self.desc is not None:
            desc = self.desc[idx]
        else:
            desc = None

        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()

        # if len(frames) < self.min_clip_len:
        #     return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)

        if len(frames) < self.clip_len:
            padding = [torch.zeros_like(frames[0]) for _ in range(self.clip_len - len(frames))]
            frames.extend(padding)
        else:
            frames = frames[:self.clip_len]

        video_tensor = torch.stack(frames)
        return video_tensor, label, desc

# def collate_fn(batch):
#     batch = [item for item in batch if item is not None]
#     if not batch:
#         return torch.empty(0, 0, 3, 224, 224), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
#     videos, labels, desc = zip(*batch)
#     return torch.stack(videos), torch.tensor(labels), list(desc)