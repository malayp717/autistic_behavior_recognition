import os
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import VideoDataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, CLIPTokenizer, CLIPTextModel
from models import VSTWithCLIP
from utils import load_chkpt, get_video_files_and_labels

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
# batch_size = config['batch_size']
batch_size = 1
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

def summarize(model, loader, model_conf):

    summarizer = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    summarizer_tokenizer = T5Tokenizer.from_pretrained("t5-base")

    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    model.to(device)
    model.eval()
    for _, data in enumerate(loader):
        if model_conf == 'VST':
            videos, _ = data[0].to(device), data[1].to(device)
        else:
            videos, _, desc = data[0].to(device), data[1].to(device), data[2]

        with torch.no_grad():
            batch_size, num_frames, C, H, W, = videos.size()
            videos = videos.view(batch_size, C, num_frames, H, W)
            video_features = model.video_model(videos)
            video_features = F.normalize(video_features, dim=1)

        text_tokens = clip_tokenizer(desc, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            text_embeddings = model.text_model(**text_tokens).last_hidden_state[:, 0, :]  # Shape: (B, text_feature_dim)
            text_embeddings = F.normalize(text_embeddings, dim=1)

        # Combine video and text features into a multimodal embedding
        # multimodal_features = torch.cat((video_features, text_embeddings), dim=1)  # Shape: (B, visual_feature_dim + text_feature_dim)

        input_text = ["Summarize the child behavior in the video: " + desc[0]]

        # Tokenize input prompt
        input_ids = summarizer_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)

        # Step 3: Generate a summary using the T5 model (or another large language model)
        with torch.no_grad():
            summary_ids = summarizer.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)

        # Decode the generated summary
        summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Output the summary
        print("Generated Summary: ", summary)

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
    summarize(model, val_loader, model_conf)

if __name__ == '__main__':
    main()