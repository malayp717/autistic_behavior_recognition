import os
import yaml
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from models import VST, VSTWithCLIP
from utils import load_chkpt, get_video_files_and_labels
from dataset import VideoDataset
from tqdm import tqdm

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
# batch_size = 1
lr = config['lr']
num_epochs = config['num_epochs']
model_conf = config['model']
setting = config['setting']
desc_req = config['desc_req']

num_classes = 2 if setting == 'binary' else len(categories)
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

def zero_shot(model, loader, model_conf):

    all_true_labels, all_predicted_labels = [], []
    model.to(device)

    model.eval()
    for i, data in tqdm(enumerate(loader)):
        if model_conf == 'VST':
            videos, labels = data[0].to(device), data[1].to(device)
        else:
            videos, labels, desc = data[0].to(device), data[1].to(device), data[2]

        with torch.no_grad():
            # Extract video embeddings
            batch_size, num_frames, C, H, W, = videos.size()
            videos = videos.view(batch_size, C, num_frames, H, W)
            video_embeddings = model.video_model(videos)
            video_embeddings = model.video_embed(video_embeddings)
            video_embeddings = F.normalize(video_embeddings, dim=1)

            # Extract text embeddings
            text_tokens = model.tokenizer(desc, return_tensors="pt", padding=True, truncation=True).to(device)
            text_embeddings = model.text_model(**text_tokens).last_hidden_state[:, 0, :]
            text_embeddings = F.normalize(text_embeddings, dim=1)

            similarity_scores = torch.matmul(video_embeddings, text_embeddings.T)

            # Predict class for each video
            predicted_class_indices = similarity_scores.argmax(dim=1)

        # Store true and predicted labels
        all_true_labels.extend(labels.cpu().numpy())
        all_predicted_labels.extend(predicted_class_indices.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted')
    conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

def main():

    video_files, labels, desc, _ = get_video_files_and_labels(preproc_data_dir, categories, setting, desc_req)
    videoDataset = VideoDataset(video_files, labels, desc, clip_len, min_clip_len=5, transform=transform)
    train_size = int(0.8 * len(videoDataset))
    val_size = len(videoDataset) - train_size

    description_mode = 'word' if desc_req == False else 'descriptive'
    model = VSTWithCLIP(num_classes, description_mode=description_mode)
    chkpt_fp = f'{model_dir}/{model_conf}_{setting}.pth.tar' if desc_req == False else f'{model_dir}/{model_conf}_exp_{setting}.pth.tar'
    chkpt = load_chkpt(chkpt_fp)
    model.load_state_dict(chkpt['model_state_dict'])

    _, val_dataset = random_split(videoDataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, prefetch_factor=8)
    zero_shot(model, val_loader, model_conf)

if __name__ == '__main__':
    main()