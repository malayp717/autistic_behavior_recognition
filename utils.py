import numpy as np
import glob
from sklearn.utils.class_weight import compute_class_weight
import torch
from category_description import category_desc

def get_label_weights(labels):

    num_classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=num_classes, y=labels)
    weights = torch.tensor(weights, dtype=torch.float32)

    return weights

def get_video_files_and_labels(data_dir, categories, setting, desc_req):

    video_files, labels, desc = [], [], []
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}

    for category in categories:
        if category == 'normal':
            cat_dir = sorted(glob.glob(f'{data_dir}/{category}/clip/*/*.avi'))
        else:
            cat_dir = sorted(glob.glob(f'{data_dir}/abnormal/{category}/clip/*/*.avi'))

        if not desc_req:
            desc.extend([category] * len(cat_dir))
        else:
            desc.extend([category_desc[category]]  * len(cat_dir))

        video_files.extend(cat_dir)

        if setting == 'binary':
            labels.extend([0 if category == 'normal' else 1] * len(cat_dir))
        else:
            labels.extend([cat_to_idx[category]] * len(cat_dir))

    weights = get_label_weights(labels)
    perm = np.random.permutation(len(video_files))
    video_files, labels, desc = [video_files[i] for i in perm], [labels[i] for i in perm], [desc[i] for i in perm]
    
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