import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import VideoDataset
from VST import VST, VSTWithCLIP
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import yaml
import warnings
import time
from loss import VSTLoss, VSTWithClipLoss
from utils import get_video_files_and_labels, save_chkpt, load_chkpt
warnings.filterwarnings(action='ignore')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()

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
batch_size = config['batch_size']
lr = config['lr']
num_epochs = config['num_epochs']
model_conf = config['model']
setting = config['setting']
# ------------- Config parameters end ------------- #

os.environ['TORCH_HOME'] = model_dir
os.environ['TRANSFORMERS_CACHE'] = model_dir

transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: x / 255.0),
                ])

def train(model, optimizer, scheduler, loader, criterion, device, model_conf):

    train_loss, train_correct, train_total = 0, 0, 0
    train_preds, train_true = [], []

    model.train()
    for _, data in enumerate(loader):
        if model_conf == 'VST':
            videos, labels = data[0].to(device), data[1].to(device)
        else:
            videos, labels, desc = data[0].to(device), data[1].to(device), data[2]

        optimizer.zero_grad()
        if model_conf == 'VST':
            logits = model(videos)
        else:
            logits, video_features, text_features = model(videos, desc)

        t_loss = criterion(logits, labels) if model_conf == 'VST' else criterion(logits, labels, video_features, text_features)
        t_loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1).cpu().numpy()
        train_preds.extend(preds)
        train_true.extend(labels.cpu().numpy())
        
        train_loss += t_loss.item() * labels.size(0)
        train_correct += (logits.argmax(dim=1) == labels).sum().item()
        train_total += labels.size(0)

        # print(f'Loss: {t_loss.item():.4f} Acc: {train_correct/train_total:.4f}')

    scheduler.step()

    train_f1 = f1_score(train_true, train_preds, average='weighted')
    train_loss /= train_total
    train_acc = train_correct / train_total

    return model, optimizer, train_loss, train_acc, train_f1


def validate(model, loader, criterion, model_conf):

    val_preds, val_true = [], []
    val_loss, val_correct, val_total = 0, 0, 0
    
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(loader):
            if model_conf == 'VST':
                videos, labels = data[0].to(device), data[1].to(device)
            else:
                videos, labels, desc = data[0].to(device), data[1].to(device), data[2]

            if model_conf == 'VST':
                logits = model(videos)
            else:
                logits, video_features, text_features = model(videos, desc)
            
            val_loss = criterion(logits, labels) if model_conf == 'VST' else criterion(logits, labels, video_features, text_features)

            preds = logits.argmax(dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_true.extend(labels.cpu().numpy())

            val_loss += val_loss.item() * labels.size(0)
            val_correct += (logits.argmax(dim=1) == labels).sum().item()
            val_total += labels.size(0)

    f1 = f1_score(val_true, val_preds, average='weighted')

    val_loss /= val_total
    val_acc = val_correct / val_total

    return f1, val_loss, val_acc


def cross_validate(model, criterion, video_files, labels, desc, clip_len, min_clip_len, batch_size, num_epochs, device):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(video_files, labels)):
        print(f"\nFold {fold+1}|5")
        train_videos, train_labels = [video_files[i] for i in train_idx], [labels[i] for i in train_idx]
        val_videos, val_labels = [video_files[i] for i in val_idx], [labels[i] for i in val_idx]

        if desc is not None:
            train_desc, val_desc = [desc[i] for i in train_idx], [desc[i] for i in val_idx]
        else:
            train_desc, val_desc = None, None

        train_dataset = VideoDataset(train_videos, train_labels, train_desc, clip_len, min_clip_len, transform)
        val_dataset = VideoDataset(val_videos, val_labels, val_desc, clip_len, min_clip_len, transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        # Load the model from previous checkpoint
        TRAIN_LOSS, TRAIN_ACC, VAL_LOSS, VAL_ACC = [], [], [], []
        chkpt_fp = f'{model_dir}/{model_conf}_{setting}_{fold}.pth.tar'

        if os.path.exists(chkpt_fp):
            chkpt = load_chkpt(chkpt_fp)
            model.load_state_dict(chkpt['model_state_dict'])
            optimizer.load_state_dict(chkpt['optimizer_state_dict'])
            TRAIN_LOSS, VAL_LOSS = chkpt['train_loss'], chkpt['val_loss']
            TRAIN_ACC, VAL_ACC = chkpt['train_acc'], chkpt['val_acc']
        
        start_time = time.time()
        for epoch in tqdm(range(num_epochs), desc="Training"):
        # for epoch in range(num_epochs):

            model, optimizer, train_loss, train_acc, train_f1 = train(model, optimizer, scheduler, train_loader, criterion, device, model_conf)
            val_f1, val_loss, val_acc = validate(model, val_loader, criterion, model_conf)

            TRAIN_LOSS.append(train_loss)
            TRAIN_ACC.append(train_acc)

            VAL_LOSS.append(val_loss)
            VAL_ACC.append(val_acc)

            save_chkpt(model, optimizer, TRAIN_LOSS, TRAIN_ACC, VAL_LOSS, VAL_ACC, chkpt_fp)

            # if (epoch+1)%2 == 0:
            print(f'{epoch+1}|{num_epochs} \t train_loss: {train_loss:.4f} \t train_acc: {train_acc:.4f} \t train_f1: {train_f1:.4f}\
                    val_loss: {val_loss:.4f} \t val_acc: {val_acc:.4f} \t val_f1: {val_f1:.4f} \t time_taken: {(time.time()-start_time)/60:.4f} mins')
            start_time = time.time()

        all_f1_scores.append(val_f1)

    print(f"\nAverage F1 Score: {np.mean(all_f1_scores):.4f}")
    
def main(model_conf, setting):

    num_classes = 2 if setting == 'binary' else len(categories)
    
    video_files, labels, desc, weights = get_video_files_and_labels(preproc_data_dir, categories, setting)
    if model_conf == 'VST':
        desc = None 

    weights = weights.to(device)
    model = VST(num_classes).to(device) if model_conf == 'VST' else VSTWithCLIP(num_classes).to(device)
    criterion = VSTLoss(weight=weights) if model_conf == 'VST' else VSTWithClipLoss(weight=weights)

    cross_validate(model, criterion, video_files, labels, desc, clip_len, min_clip_len=5, batch_size=batch_size,
                   num_epochs=num_epochs, device=device)


if __name__ == "__main__":
    
    main(model_conf, setting)