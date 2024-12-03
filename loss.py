import torch
import torch.nn as nn

class VSTLoss(nn.Module):
    def __init__(self, weight):
        super(VSTLoss, self).__init__()
        # self.loss = nn.CrossEntropyLoss(weight=weight)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, logits, labels):
        loss = self.loss(logits, labels)
        return loss
    
class VSTWithClipLoss(nn.Module):
    def __init__(self, weight, lambda_contrastive=0.1):
        super(VSTWithClipLoss, self).__init__()
        # self.loss = nn.CrossEntropyLoss(weight=weight)
        self.loss = nn.CrossEntropyLoss()
        self.lambda_contrastive = lambda_contrastive
    
    def forward(self, logits, labels, video_features, text_features):
        ce_loss = self.loss(logits, labels)
        
        normalized_video_features = nn.functional.normalize(video_features, dim=1)
        normalized_text_features = nn.functional.normalize(text_features, dim=1)

        contrastive_loss = torch.mean(1 - torch.sum(normalized_video_features * normalized_text_features, dim=1))
        
        loss = ce_loss + self.lambda_contrastive * contrastive_loss
        return loss