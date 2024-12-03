import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from dataset import VideoDataset
from models import VST, VSTWithCLIP
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
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
torch.manual_seed(0)

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
desc_req = config['desc_req']
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

def train(model, optimizer, scheduler, loader, criterion, device, model_conf, num_classes):

    train_loss, train_correct, train_total = 0, 0, 0
    train_preds, train_true, train_losses = [], [], []

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
        train_losses.append(t_loss.item())

        # print(f'Batch Loss: {t_loss.item():.4f} Acc: {train_correct/train_total:.4f}')

    scheduler.step()

    train_f1 = f1_score(train_true, train_preds, average='weighted')
    train_loss /= train_total
    train_acc = train_correct / train_total
    auc_roc = roc_auc_score(label_binarize(train_true, classes=[i for i in range(num_classes)]), train_preds, average='macro', multi_class='ovr')

    preds_count = [train_preds.count(c) for c in range(num_classes)]
    true_count = [train_true.count(c) for c in range(num_classes)]

    print(f'Train\n Preds: {preds_count} \t True: {true_count}')

    return model, optimizer, train_loss, train_acc, train_f1, train_losses, auc_roc


def validate(model, loader, criterion, model_conf, num_classes):

    val_preds, val_true, val_losses = [], [], []
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
            
            v_loss = criterion(logits, labels) if model_conf == 'VST' else criterion(logits, labels, video_features, text_features)

            preds = logits.argmax(dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_true.extend(labels.cpu().numpy())

            val_loss += v_loss.item() * labels.size(0)
            val_correct += (logits.argmax(dim=1) == labels).sum().item()
            val_total += labels.size(0)

            val_losses.append(v_loss.item())

    f1 = f1_score(val_true, val_preds, average='weighted')

    val_loss /= val_total
    val_acc = val_correct / val_total
    auc_roc = roc_auc_score(label_binarize(val_true, classes=[i for i in range(num_classes)]), val_preds, average='macro', multi_class='ovr')

    preds_count = [val_preds.count(c) for c in range(num_classes)]
    true_count = [val_true.count(c) for c in range(num_classes)]

    print(f'Validation\n Preds: {preds_count} \t True: {true_count}')
    return f1, val_loss, val_acc, val_losses, auc_roc


def k_fold(model, criterion, video_files, labels, desc, clip_len, min_clip_len, batch_size, num_epochs, num_classes, desc_req, device):

    videoDataset = VideoDataset(video_files, labels, desc, clip_len, min_clip_len, transform)
    train_size = int(0.8 * len(videoDataset))
    val_size = len(videoDataset) - train_size

    train_dataset, val_dataset = random_split(videoDataset, [train_size, val_size])

    train_labels = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
    class_counts = torch.tensor([(train_labels == t).sum() for t in torch.unique(train_labels)], dtype=torch.float)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels.long()]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=8, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, prefetch_factor=4)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Load the model from previous checkpoint
    TRAIN_LOSS, TRAIN_ACC, VAL_LOSS, VAL_ACC = [], [], [], []
    chkpt_fp = f'{model_dir}/{model_conf}_{setting}.pth.tar' if desc_req == False else f'{model_dir}/{model_conf}_exp_{setting}.pth.tar'

    if os.path.exists(chkpt_fp):
        chkpt = load_chkpt(chkpt_fp)
        model.load_state_dict(chkpt['model_state_dict'])
        optimizer.load_state_dict(chkpt['optimizer_state_dict'])
        TRAIN_LOSS, VAL_LOSS = chkpt['train_loss'], chkpt['val_loss']
        TRAIN_ACC, VAL_ACC = chkpt['train_acc'], chkpt['val_acc']
    
    start_time = time.time()
    for epoch in range(num_epochs):

        model, optimizer, train_loss, train_acc, train_f1, train_losses, train_auc_roc = train(model, optimizer, scheduler, train_loader, criterion, device,
                                                                                model_conf, num_classes)
        val_f1, val_loss, val_acc, val_losses, val_auc_roc = validate(model, val_loader, criterion, model_conf, num_classes)

        TRAIN_LOSS.extend(train_losses)
        TRAIN_ACC.append(train_acc)

        VAL_LOSS.extend(val_losses)
        VAL_ACC.append(val_acc)

        save_chkpt(model, optimizer, TRAIN_LOSS, TRAIN_ACC, VAL_LOSS, VAL_ACC, chkpt_fp)

        print(f'{epoch+1}|{num_epochs} \t train_loss: {train_loss:.4f} \t train_acc: {train_acc:.4f} \t train_f1: {train_f1:.4f}\
                train_auc_roc: {train_auc_roc:.4f} \t val_loss: {val_loss:.4f} \t val_acc: {val_acc:.4f} \t val_f1: {val_f1:.4f}\
                        val_auc_roc: {val_auc_roc:.4f} \t time_taken: {(time.time()-start_time)/60:.4f} mins')
        start_time = time.time()

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
    
def main(model_conf, setting, desc_req):

    num_classes = 2 if setting == 'binary' else len(categories)
    
    video_files, labels, desc, weights = get_video_files_and_labels(preproc_data_dir, categories, setting, desc_req)
    if model_conf == 'VST':
        desc = None 

    weights = weights.to(device)
    description_mode = 'word' if desc_req == False else 'descriptive'
    model = VST(num_classes).to(device) if model_conf == 'VST' else VSTWithCLIP(num_classes).to(device)
    criterion = VSTLoss(weight=weights) if model_conf == 'VST' else VSTWithClipLoss(weight=weights)

    total_params, trainable_params = count_parameters(model)
    print(f'total_params: {total_params} \t trainable_params: {trainable_params}')

    k_fold(model, criterion, video_files, labels, desc, clip_len, min_clip_len=5, batch_size=batch_size,
                   num_epochs=num_epochs, num_classes=num_classes, desc_req=desc_req, device=device)


if __name__ == "__main__":
    print(f"--------- Training Configuration \t\t Model: {model_conf} \t\t Setting: {setting} \t\t desc_req: {desc_req} classification ---------")
    main(model_conf, setting, desc_req)