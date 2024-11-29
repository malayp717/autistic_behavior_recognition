import glob
import torch
import torch.nn as nn
from transformers import CLIPTokenizer

category_desc = {
    
}

class Tokenizer:
    def __init__(self, max_length, device):
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.device = device
        self.max_length = max_length
    
    def forward(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
        return tokens['input_ids'].to(self.device)

def get_label_weights(weights):

    weights = torch.tensor(weights, dtype=torch.float32)
    weights /= torch.sum(weights)
    weights = 1/weights

    return weights

def get_video_files_and_labels_binary(data_dir):

    abnormal_video_files = sorted(glob.glob(f"{data_dir}/abnormal/*/clip/*/*.avi"))
    normal_video_files = sorted(glob.glob(f"{data_dir}/normal/clip/*/*.avi"))
    
    video_files = abnormal_video_files + normal_video_files
    labels = [0] * len(abnormal_video_files) + [1] * len(normal_video_files)

    weights = [len(video_files)/len(abnormal_video_files), len(video_files)/len(normal_video_files)]
    weights = get_label_weights(weights)

    return video_files, labels, weights

def get_video_files_and_labels_multi(data_dir, categories, device):

    video_files, labels, desc, weights = [], [], [], []
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    tokenizer = Tokenizer(max_length=5, device=device)

    for category in categories:
        if category == 'normal':
            cat_dir = sorted(glob.glob(f'{data_dir}/{category}/clip/*/*.avi'))
        else:
            cat_dir = sorted(glob.glob(f'{data_dir}/abnormal/{category}/clip/*/*.avi'))
        
        weights.append(len(cat_dir))
        desc.extend([category]*len(cat_dir) )

        #tokenized_output = tokenizer.forward(category)
        #token_tensor = tokenized_output.repeat((len(cat_dir), 1))

        video_files.extend(cat_dir)
        #desc.append(token_tensor)
        labels.extend([cat_to_idx[category]] * len(cat_dir))
    
    weights = get_label_weights(weights)
    #desc = torch.cat(desc, dim=0).long()

    return video_files, labels, desc, weights

def save_chkpt(model, optimizer, train_loss, train_acc, val_loss, val_acc, fp):
    chkpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
    }

    torch.save(chkpt, fp)

def load_chkpt(fp):
    chkpt = torch.load(fp)
    return chkpt