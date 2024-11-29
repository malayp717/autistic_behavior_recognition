import torch
from torch import nn
from torchvision.models.video import swin3d_b
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Video-Text Model
class VideoSwinCLIPClassifier(nn.Module):
    def __init__(self, num_classes, swin_weights="KINETICS400_V1"):
        super(VideoSwinCLIPClassifier, self).__init__()

        # Video Swin Transformer (Pretrained)
        self.video_model = swin3d_b(weights=swin_weights)
        in_features = self.video_model.head.in_features
        self.video_model.head = nn.Identity()  # Remove classification head
        
        # Reduce video features to match text features (512 for CLIP base model)
        self.video_embed = nn.Linear(in_features, 512)

        # CLIP Text Model
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        # Classification Layer
        self.classifier = nn.Linear(512 * 2, num_classes)  # Concatenated features

    def forward(self, videos, texts):
        # Video features
        video_features = self.video_model(videos)  # Shape: (batch_size, in_features)
        video_features = self.video_embed(video_features)  # Shape: (batch_size, 512)

        # Text features
        text_tokens = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=50
        ).to(videos.device)
        text_features = self.text_model(**text_tokens).last_hidden_state[:, 0, :]  # CLS token

        # Concatenate video and text features
        combined_features = torch.cat((video_features, text_features), dim=1)
        logits = self.classifier(combined_features)

        return logits, video_features, text_features


# Combined Loss
class CombinedLoss(nn.Module):
    def __init__(self, weight=None, lambda_contrastive=0.1):
        super(CombinedLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight)
        self.lambda_contrastive = lambda_contrastive

    def forward(self, logits, labels, video_features, text_features):
        # Cross-Entropy Loss
        ce_loss = self.cross_entropy(logits, labels)
        # Contrastive Loss
        normalized_video_features = F.normalize(video_features, dim=1)
        normalized_text_features = F.normalize(text_features, dim=1)
        contrastive_loss = torch.mean(1 - torch.sum(normalized_video_features * normalized_text_features, dim=1))
        # Combined Loss
        loss = ce_loss + self.lambda_contrastive * contrastive_loss
        return loss


# Dummy Dataset
class DummyVideoTextDataset(Dataset):
    def __init__(self, num_samples=100, num_classes=5, video_dim=(3, 16, 224, 224)):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.video_dim = video_dim
        self.texts = ["action class description " + str(i) for i in range(num_classes)]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        video = torch.rand(self.video_dim)  # Dummy video tensor
        label = idx % self.num_classes
        text = self.texts[label]
        return video, text, label


# Training Script
def train_model():
    # Hyperparameters
    num_classes = 5
    batch_size = 1
    num_epochs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataset and Dataloader
    dataset = DummyVideoTextDataset(num_samples=100, num_classes=num_classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, Loss, and Optimizer
    model = VideoSwinCLIPClassifier(num_classes=num_classes).to(device)
    criterion = CombinedLoss(lambda_contrastive=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for videos, texts, labels in dataloader:
            videos, labels = videos.to(device), labels.to(device)
            logits, video_features, text_features = model(videos, texts)

            # Compute loss
            loss = criterion(logits, labels, video_features, text_features)
            epoch_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader):.4f}")


# Run training
if __name__ == "__main__":
    train_model()