import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
from torchvision.models.video import swin3d_b
import os
import yaml
import warnings
warnings.filterwarnings('ignore')

proj_dir = os.path.dirname(os.path.abspath(__file__))
with open(f'{proj_dir}/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

model_dir = config['model_dir']
os.environ['TORCH_HOME'] = model_dir
os.environ['TRANSFORMERS_CACHE'] = model_dir

class VST(nn.Module):
    def __init__(self, num_classes):
        super(VST, self).__init__()

        self.backbone = swin3d_b(weights='KINETICS400_V1')

        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.features[6].parameters():
            param.requires_grad = True
            
        for param in self.backbone.norm.parameters():
            param.requires_grad = True

        for param in self.backbone.avgpool.parameters():
            param.requires_grad = True

        for param in self.backbone.head.parameters():
            param.requires_grad = True    
        
        self.backbone.head = nn.Linear(self.backbone.head.in_features, num_classes)

    def forward(self, x):
        batch_size, num_frames, C, H, W, = x.size()
        x = x.view(batch_size, C, num_frames, H, W)

        logits = self.backbone(x)
        return logits
    
class VSTWithCLIP(nn.Module):
    def __init__(self, num_classes, description_mode):
        super(VSTWithCLIP, self).__init__()
        self.hid_dim = 512

        self.video_model = VST(num_classes).backbone
        self.in_features = self.video_model.head.in_features
        self.video_model.head = nn.Identity()
        self.video_embed = nn.Linear(self.in_features, self.hid_dim)

        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.text_model.parameters():
            param.requires_grad = False
        
        self.classifier = nn.Linear(self.hid_dim*2, num_classes)
        self.max_len = 50 if description_mode == 'word' else 500

    def forward(self, video, texts):
        batch_size, num_frames, C, H, W, = video.size()
        video = video.view(batch_size, C, num_frames, H, W)

        video_features = self.video_model(video)
        video_features = self.video_embed(video_features)

        text_tokens = self.tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_len).to(video.device)
        text_features = self.text_model(**text_tokens).last_hidden_state[:, 0, :]

        combined_features = torch.cat((video_features, text_features), dim=1)
        logits = self.classifier(combined_features)

        return logits, video_features, text_features