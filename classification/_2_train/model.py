# model.py
import torch
import torch.nn as nn

from classification._2_train.layers_mmf_3 import MMFConv2dv5, MMFConv2dv6, MMFLinear, MMFConv2d, MMFConv2dRes, MMFLinearv1, MMFConv2dv1, MMFLinearv5 
from classification._2_train.layers_mmf_3 import MMFLinearv3, MMFConv2dv3, MMFLinearv4, MMFConv2dv4, MMFLinearv6, MMFConv2dv6

###################### YOLOv1 Classification ######################
class YOLOv1Classifier(nn.Module):
    """
    YOLOv1 architecture adapted for CIFAR-10 classification.
    Follows the exact conv layers from the original YOLOv1 paper
    (Redmon et al., 2016, arXiv:1506.02640), but replaces grid regression
    with a simple classification head (10 classes).
    
    Input:  [B, 3, 32, 32]  (CIFAR-10 RGB images)
    Output: [B, 10]         (class logits)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Backbone classification: 20 convolutional layers + maxpools (exact from YOLOv1 paper)
        self.backbone_classification = nn.Sequential(

            # Conv1: 7x7x64-s-2 + MaxPool: 2x2-s-2
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), #1
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv2: 3x3x192 + MaxPool: 2x2-s-2
            nn.Conv2d(64, 192, kernel_size=3, padding=1), #2
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv3–6: 1x1x128 + 3x3x256 + 1x1x256 + 3x3x512 + MaxPool: 2x2-s-2
            nn.Conv2d(192, 128, kernel_size=1), #3
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), #4
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=1), #5
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), #6
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv7–16: (1x1x256 + 3x3x512)x4 + 1x1x512 + 3x3x1024 + Maxpool:2x2-s-2
            nn.Conv2d(512, 256, kernel_size=1), #7
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), #8
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1), #9
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), #10
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1), #11
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), #12
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1), #13
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), #14
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, kernel_size=1), #15
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), #16
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv17–20: (1x1x512 + 3x3x1024)x2
            nn.Conv2d(1024, 512, kernel_size=1), #17
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), #18
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1), #19
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), #20
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Classification head: exactly as described in YOLOv1 paper (first 20 convs + avg pool + FC)
        self.head_classification = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),          # average-pooling layer to 1×1
            nn.Flatten(),                          # flatten [B, 1024, 1, 1] → [B, 1024]
            nn.Linear(1024, num_classes)           # single fully connected layer
        )

    def forward(self, x):
        # x: [B, 3, 32, 32]
        features = self.backbone_classification(x)        # → [B, 1024, 1, 1]
        logits = self.head_classification(features)       # → [B, 10]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        return logits
    
    def get_probs(self, x, temperature=1.0):
        logits = self(x)
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        return probs

###################### YOLOv1 Classification MMF ######################
class YOLOv1ClassifierMMF(nn.Module):
    """
    YOLOv1 architecture adapted for CIFAR-10 classification.
    Follows the exact conv layers from the original YOLOv1 paper
    (Redmon et al., 2016, arXiv:1506.02640), but replaces grid regression
    with a simple classification head (10 classes).
    
    Input:  [B, 3, 32, 32]  (CIFAR-10 RGB images)
    Output: [B, 10]         (class logits)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Backbone classification: 20 convolutional layers + maxpools (exact from YOLOv1 paper)
        self.backbone_classification = nn.Sequential(

            # Conv1: 7x7x64-s-2 + MaxPool: 2x2-s-2
            MMFConv2d(3, 64, kernel_size=7, stride=2, padding=3), #1
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv2: 3x3x192 + MaxPool: 2x2-s-2
            MMFConv2d(64, 192, kernel_size=3, padding=1), #2
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv3–6: 1x1x128 + 3x3x256 + 1x1x256 + 3x3x512 + MaxPool: 2x2-s-2
            MMFConv2d(192, 128, kernel_size=1), #3
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(128, 256, kernel_size=3, padding=1), #4
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 256, kernel_size=1), #5
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 512, kernel_size=3, padding=1), #6
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv7–16: (1x1x256 + 3x3x512)x4 + 1x1x512 + 3x3x1024 + Maxpool:2x2-s-2
            MMFConv2d(512, 256, kernel_size=1), #7
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 512, kernel_size=3, padding=1), #8
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 256, kernel_size=1), #9
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 512, kernel_size=3, padding=1), #10
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 256, kernel_size=1), #11
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 512, kernel_size=3, padding=1), #12
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 256, kernel_size=1), #13
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 512, kernel_size=3, padding=1), #14
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 512, kernel_size=1), #15
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 1024, kernel_size=3, padding=1), #16
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv17–20: (1x1x512 + 3x3x1024)x2
            MMFConv2d(1024, 512, kernel_size=1), #17
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 1024, kernel_size=3, padding=1), #18
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(1024, 512, kernel_size=1), #19
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 1024, kernel_size=3, padding=1), #20
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Classification head: exactly as described in YOLOv1 paper (first 20 convs + avg pool + FC)
        self.head_classification = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),          # average-pooling layer to 1×1
            nn.Flatten(),                           # flatten [B, 1024, 1, 1] → [B, 1024]
            MMFLinear(1024, num_classes),            # single fully connected layer
        )

    def forward(self, x):
        # x: [B, 3, 32, 32]
        features = self.backbone_classification(x)        # → [B, 1024, 1, 1]
        logits = self.head_classification(features)       # → [B, 10]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        return logits
    
    def get_probs(self, x, temperature=1.0):
        logits = self(x)
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        return probs

###################### YOLOv1 Classification MMFv1 ######################
class YOLOv1ClassifierMMFv1(nn.Module):
    """
    YOLOv1 architecture adapted for CIFAR-10 classification.
    Follows the exact conv layers from the original YOLOv1 paper
    (Redmon et al., 2016, arXiv:1506.02640), but replaces grid regression
    with a simple classification head (10 classes).

    Modified from YOLOv1ClassifierMMF: removed backward quantization to all MMFConv2d and MMFLinear layers
    
    Input:  [B, 3, 32, 32]  (CIFAR-10 RGB images)
    Output: [B, 10]         (class logits)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Backbone classification: 20 convolutional layers + maxpools (exact from YOLOv1 paper)
        self.backbone_classification = nn.Sequential(

            # Conv1: 7x7x64-s-2 + MaxPool: 2x2-s-2
            MMFConv2dv1(3, 64, kernel_size=7, stride=2, padding=3), #1
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv2: 3x3x192 + MaxPool: 2x2-s-2
            MMFConv2dv1(64, 192, kernel_size=3, padding=1), #2
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv3–6: 1x1x128 + 3x3x256 + 1x1x256 + 3x3x512 + MaxPool: 2x2-s-2
            MMFConv2dv1(192, 128, kernel_size=1), #3
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv1(128, 256, kernel_size=3, padding=1), #4
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv1(256, 256, kernel_size=1), #5
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv1(256, 512, kernel_size=3, padding=1), #6
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv7–16: (1x1x256 + 3x3x512)x4 + 1x1x512 + 3x3x1024 + Maxpool:2x2-s-2
            MMFConv2dv1(512, 256, kernel_size=1), #7
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv1(256, 512, kernel_size=3, padding=1), #8
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv1(512, 256, kernel_size=1), #9
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv1(256, 512, kernel_size=3, padding=1), #10
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv1(512, 256, kernel_size=1), #11
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv1(256, 512, kernel_size=3, padding=1), #12
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv1(512, 256, kernel_size=1), #13
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv1(256, 512, kernel_size=3, padding=1), #14
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv1(512, 512, kernel_size=1), #15
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv1(512, 1024, kernel_size=3, padding=1), #16
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv17–20: (1x1x512 + 3x3x1024)x2
            MMFConv2dv1(1024, 512, kernel_size=1), #17
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv1(512, 1024, kernel_size=3, padding=1), #18
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv1(1024, 512, kernel_size=1), #19
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv1(512, 1024, kernel_size=3, padding=1), #20
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Classification head: exactly as described in YOLOv1 paper (first 20 convs + avg pool + FC)
        self.head_classification = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),          # average-pooling layer to 1×1
            nn.Flatten(),                           # flatten [B, 1024, 1, 1] → [B, 1024]
            MMFLinearv1(1024, num_classes),            # single fully connected layer
        )

    def forward(self, x):
        # x: [B, 3, 32, 32]
        features = self.backbone_classification(x)        # → [B, 1024, 1, 1]
        logits = self.head_classification(features)       # → [B, 10]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        return logits
    
    def get_probs(self, x, temperature=1.0):
        logits = self(x)
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        return probs

###################### YOLOv1 Classification MMFv2 ######################
class YOLOv1ClassifierMMFv2(nn.Module):
    """
    YOLOv1 architecture adapted for CIFAR-10 classification.
    Follows the exact conv layers from the original YOLOv1 paper
    (Redmon et al., 2016, arXiv:1506.02640), but replaces grid regression
    with a simple classification head (10 classes).

    Same as YOLOv1ClassifierMMF, but with wider channels (double) in Conv2d layers
    
    Input:  [B, 3, 32, 32]  (CIFAR-10 RGB images)
    Output: [B, 10]         (class logits)
    """
    def __init__(self, num_classes=10, channel_factor=2):
        super().__init__()
        
         # Backbone classification: 20 convolutional layers + maxpools (exact from YOLOv1 paper)
        self.backbone_classification = nn.Sequential(

            # Conv1: 7x7x64-s-2 + MaxPool: 2x2-s-2
            MMFConv2d(3, 64 * channel_factor, kernel_size=7, stride=2, padding=3), #1
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv2: 3x3x192 + MaxPool: 2x2-s-2
            MMFConv2d(64 * channel_factor, 192 * channel_factor, kernel_size=3, padding=1), #2
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv3–6: 1x1x128 + 3x3x256 + 1x1x256 + 3x3x512 + MaxPool: 2x2-s-2
            MMFConv2d(192 * channel_factor, 128 * channel_factor, kernel_size=1), #3
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(128 * channel_factor, 256 * channel_factor, kernel_size=3, padding=1), #4
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256 * channel_factor, 256 * channel_factor, kernel_size=1), #5
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256 * channel_factor, 512 * channel_factor, kernel_size=3, padding=1), #6
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv7–16: (1x1x256 + 3x3x512)x4 + 1x1x512 + 3x3x1024 + Maxpool:2x2-s-2
            MMFConv2d(512 * channel_factor, 256 * channel_factor, kernel_size=1), #7
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256 * channel_factor, 512 * channel_factor, kernel_size=3, padding=1), #8
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512 * channel_factor, 256 * channel_factor, kernel_size=1), #9
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256 * channel_factor, 512 * channel_factor, kernel_size=3, padding=1), #10
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512 * channel_factor, 256 * channel_factor, kernel_size=1), #11
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256 * channel_factor, 512 * channel_factor, kernel_size=3, padding=1), #12
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512 * channel_factor, 256 * channel_factor, kernel_size=1), #13
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256 * channel_factor, 512 * channel_factor, kernel_size=3, padding=1), #14
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512 * channel_factor, 512 * channel_factor, kernel_size=1), #15
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512 * channel_factor, 1024 * channel_factor, kernel_size=3, padding=1), #16
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv17–20: (1x1x512 + 3x3x1024)x2
            MMFConv2d(1024 * channel_factor, 512 * channel_factor, kernel_size=1), #17
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512 * channel_factor, 1024 * channel_factor, kernel_size=3, padding=1), #18
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(1024 * channel_factor, 512 * channel_factor, kernel_size=1), #19
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512 * channel_factor, 1024 * channel_factor, kernel_size=3, padding=1), #20
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Classification head: exactly as described in YOLOv1 paper (first 20 convs + avg pool + FC)
        self.head_classification = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),          # average-pooling layer to 1×1
            nn.Flatten(),                           # flatten [B, 1024, 1, 1] → [B, 1024]
            MMFLinear(1024 * channel_factor, num_classes), # single fully connected layer
        )

    def forward(self, x):
        # x: [B, 3, 32, 32]
        features = self.backbone_classification(x)        # → [B, 1024, 1, 1]
        logits = self.head_classification(features)       # → [B, 10]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        return logits
    
    def get_probs(self, x, temperature=1.0):
        logits = self(x)
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        return probs

###################### YOLOv1 Classification MMFv3 ######################
class YOLOv1ClassifierMMFv3(nn.Module):
    """
    YOLOv1 architecture adapted for CIFAR-10 classification.
    Follows the exact conv layers from the original YOLOv1 paper
    (Redmon et al., 2016, arXiv:1506.02640), but replaces grid regression
    with a simple classification head (10 classes).

    Modified from YOLOv1ClassifierMMF: replaced act_quant to int8 with f8
    
    Input:  [B, 3, 32, 32]  (CIFAR-10 RGB images)
    Output: [B, 10]         (class logits)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Backbone classification: 20 convolutional layers + maxpools (exact from YOLOv1 paper)
        self.backbone_classification = nn.Sequential(

            # Conv1: 7x7x64-s-2 + MaxPool: 2x2-s-2
            MMFConv2dv3(3, 64, kernel_size=7, stride=2, padding=3), #1
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv2: 3x3x192 + MaxPool: 2x2-s-2
            MMFConv2dv3(64, 192, kernel_size=3, padding=1), #2
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv3–6: 1x1x128 + 3x3x256 + 1x1x256 + 3x3x512 + MaxPool: 2x2-s-2
            MMFConv2dv3(192, 128, kernel_size=1), #3
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv3(128, 256, kernel_size=3, padding=1), #4
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv3(256, 256, kernel_size=1), #5
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv3(256, 512, kernel_size=3, padding=1), #6
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv7–16: (1x1x256 + 3x3x512)x4 + 1x1x512 + 3x3x1024 + Maxpool:2x2-s-2
            MMFConv2dv3(512, 256, kernel_size=1), #7
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv3(256, 512, kernel_size=3, padding=1), #8
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv3(512, 256, kernel_size=1), #9
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv3(256, 512, kernel_size=3, padding=1), #10
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv3(512, 256, kernel_size=1), #11
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv3(256, 512, kernel_size=3, padding=1), #12
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv3(512, 256, kernel_size=1), #13
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv3(256, 512, kernel_size=3, padding=1), #14
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv3(512, 512, kernel_size=1), #15
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv3(512, 1024, kernel_size=3, padding=1), #16
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv17–20: (1x1x512 + 3x3x1024)x2
            MMFConv2dv3(1024, 512, kernel_size=1), #17
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv3(512, 1024, kernel_size=3, padding=1), #18
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv3(1024, 512, kernel_size=1), #19
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv3(512, 1024, kernel_size=3, padding=1), #20
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Classification head: exactly as described in YOLOv1 paper (first 20 convs + avg pool + FC)
        self.head_classification = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),          # average-pooling layer to 1×1
            nn.Flatten(),                           # flatten [B, 1024, 1, 1] → [B, 1024]
            MMFLinearv3(1024, num_classes),            # single fully connected layer
        )

    def forward(self, x):
        # x: [B, 3, 32, 32]
        features = self.backbone_classification(x)        # → [B, 1024, 1, 1]
        logits = self.head_classification(features)       # → [B, 10]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        return logits
    
    def get_probs(self, x, temperature=1.0):
        logits = self(x)
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        return probs

###################### YOLOv1 Classification MMFv4 ######################
class YOLOv1ClassifierMMFv4(nn.Module):
    """
    YOLOv1 architecture adapted for CIFAR-10 classification.
    Follows the exact conv layers from the original YOLOv1 paper
    (Redmon et al., 2016, arXiv:1506.02640), but replaces grid regression
    with a simple classification head (10 classes).

    Modified from YOLOv1ClassifierMMF: removed activation quantization to all MMFConv2d and MMFLinear layers
    
    Input:  [B, 3, 32, 32]  (CIFAR-10 RGB images)
    Output: [B, 10]         (class logits)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Backbone classification: 20 convolutional layers + maxpools (exact from YOLOv1 paper)
        self.backbone_classification = nn.Sequential(

            # Conv1: 7x7x64-s-2 + MaxPool: 2x2-s-2
            MMFConv2dv4(3, 64, kernel_size=7, stride=2, padding=3), #1
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv2: 3x3x192 + MaxPool: 2x2-s-2
            MMFConv2dv4(64, 192, kernel_size=3, padding=1), #2
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv3–6: 1x1x128 + 3x3x256 + 1x1x256 + 3x3x512 + MaxPool: 2x2-s-2
            MMFConv2dv4(192, 128, kernel_size=1), #3
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv4(128, 256, kernel_size=3, padding=1), #4
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv4(256, 256, kernel_size=1), #5
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv4(256, 512, kernel_size=3, padding=1), #6
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv7–16: (1x1x256 + 3x3x512)x4 + 1x1x512 + 3x3x1024 + Maxpool:2x2-s-2
            MMFConv2dv4(512, 256, kernel_size=1), #7
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv4(256, 512, kernel_size=3, padding=1), #8
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv4(512, 256, kernel_size=1), #9
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv4(256, 512, kernel_size=3, padding=1), #10
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv4(512, 256, kernel_size=1), #11
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv4(256, 512, kernel_size=3, padding=1), #12
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv4(512, 256, kernel_size=1), #13
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv4(256, 512, kernel_size=3, padding=1), #14
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv4(512, 512, kernel_size=1), #15
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv4(512, 1024, kernel_size=3, padding=1), #16
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv17–20: (1x1x512 + 3x3x1024)x2
            MMFConv2dv4(1024, 512, kernel_size=1), #17
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv4(512, 1024, kernel_size=3, padding=1), #18
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv4(1024, 512, kernel_size=1), #19
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv4(512, 1024, kernel_size=3, padding=1), #20
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Classification head: exactly as described in YOLOv1 paper (first 20 convs + avg pool + FC)
        self.head_classification = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),          # average-pooling layer to 1×1
            nn.Flatten(),                           # flatten [B, 1024, 1, 1] → [B, 1024]
            MMFLinearv4(1024, num_classes),            # single fully connected layer
        )

    def forward(self, x):
        # x: [B, 3, 32, 32]
        features = self.backbone_classification(x)        # → [B, 1024, 1, 1]
        logits = self.head_classification(features)       # → [B, 10]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        return logits
    
    def get_probs(self, x, temperature=1.0):
        logits = self(x)
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        return probs

###################### YOLOv1 Classification MMFv5 ######################
class YOLOv1ClassifierMMFv5(nn.Module):
    """
    YOLOv1 architecture adapted for CIFAR-10 classification.
    Follows the exact conv layers from the original YOLOv1 paper
    (Redmon et al., 2016, arXiv:1506.02640), but replaces grid regression
    with a simple classification head (10 classes).

    Modified from YOLOv1ClassifierMMF: weighn initialization further away from 0
    
    Input:  [B, 3, 32, 32]  (CIFAR-10 RGB images)
    Output: [B, 10]         (class logits)
    """
    def __init__(self, num_classes=10, weight_init_scale=1.0):
        super().__init__()
        
        # Backbone classification: 20 convolutional layers + maxpools (exact from YOLOv1 paper)
        self.backbone_classification = nn.Sequential(

            # Conv1: 7x7x64-s-2 + MaxPool: 2x2-s-2
            MMFConv2dv5(3, 64, kernel_size=7, stride=2, padding=3, weight_init_scale=weight_init_scale), #1
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv2: 3x3x192 + MaxPool: 2x2-s-2
            MMFConv2dv5(64, 192, kernel_size=3, padding=1, weight_init_scale=weight_init_scale), #2
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv3–6: 1x1x128 + 3x3x256 + 1x1x256 + 3x3x512 + MaxPool: 2x2-s-2
            MMFConv2dv5(192, 128, kernel_size=1, weight_init_scale=weight_init_scale), #3
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv5(128, 256, kernel_size=3, padding=1, weight_init_scale=weight_init_scale), #4
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv5(256, 256, kernel_size=1, weight_init_scale=weight_init_scale), #5
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv5(256, 512, kernel_size=3, padding=1, weight_init_scale=weight_init_scale), #6
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv7–16: (1x1x256 + 3x3x512)x4 + 1x1x512 + 3x3x1024 + Maxpool:2x2-s-2
            MMFConv2dv5(512, 256, kernel_size=1, weight_init_scale=weight_init_scale), #7
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv5(256, 512, kernel_size=3, padding=1, weight_init_scale=weight_init_scale), #8
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv5(512, 256, kernel_size=1, weight_init_scale=weight_init_scale), #9
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv5(256, 512, kernel_size=3, padding=1, weight_init_scale=weight_init_scale), #10
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv5(512, 256, kernel_size=1, weight_init_scale=weight_init_scale), #11
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv5(256, 512, kernel_size=3, padding=1, weight_init_scale=weight_init_scale), #12
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv5(512, 256, kernel_size=1, weight_init_scale=weight_init_scale), #13
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv5(256, 512, kernel_size=3, padding=1, weight_init_scale=weight_init_scale), #14
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv5(512, 512, kernel_size=1, weight_init_scale=weight_init_scale), #15
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv5(512, 1024, kernel_size=3, padding=1, weight_init_scale=weight_init_scale), #16
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv17–20: (1x1x512 + 3x3x1024)x2
            MMFConv2dv5(1024, 512, kernel_size=1, weight_init_scale=weight_init_scale), #17
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv5(512, 1024, kernel_size=3, padding=1, weight_init_scale=weight_init_scale), #18
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv5(1024, 512, kernel_size=1, weight_init_scale=weight_init_scale), #19
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv5(512, 1024, kernel_size=3, padding=1, weight_init_scale=weight_init_scale), #20
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Classification head: exactly as described in YOLOv1 paper (first 20 convs + avg pool + FC)
        self.head_classification = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),          # average-pooling layer to 1×1
            nn.Flatten(),                           # flatten [B, 1024, 1, 1] → [B, 1024]
            MMFLinearv5(1024, num_classes, weight_init_scale=weight_init_scale),  # single fully connected layer
        )

    def forward(self, x):
        # x: [B, 3, 32, 32]
        features = self.backbone_classification(x)        # → [B, 1024, 1, 1]
        logits = self.head_classification(features)       # → [B, 10]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        return logits
    
    def get_probs(self, x, temperature=1.0):
        logits = self(x)
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        return probs

###################### YOLOv1 Classification MMFv6 ######################
class YOLOv1ClassifierMMFv6(nn.Module):
    """
    YOLOv1 architecture adapted for CIFAR-10 classification.
    Follows the exact conv layers from the original YOLOv1 paper
    (Redmon et al., 2016, arXiv:1506.02640), but replaces grid regression
    with a simple classification head (10 classes).

    Modified from YOLOv1ClassifierMMF: weight scale learnable
    
    Input:  [B, 3, 32, 32]  (CIFAR-10 RGB images)
    Output: [B, 10]         (class logits)
    """
    def __init__(self, num_classes=10, weight_init_scale=1.0):
        super().__init__()
        
        # Backbone classification: 20 convolutional layers + maxpools (exact from YOLOv1 paper)
        self.backbone_classification = nn.Sequential(

            # Conv1: 7x7x64-s-2 + MaxPool: 2x2-s-2
            MMFConv2dv6(3, 64, kernel_size=7, stride=2, padding=3, weight_init_scale=weight_init_scale), #1
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv2: 3x3x192 + MaxPool: 2x2-s-2
            MMFConv2dv6(64, 192, kernel_size=3, padding=1, weight_init_scale=weight_init_scale), #2
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv3–6: 1x1x128 + 3x3x256 + 1x1x256 + 3x3x512 + MaxPool: 2x2-s-2
            MMFConv2dv6(192, 128, kernel_size=1, weight_init_scale=weight_init_scale), #3
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv6(128, 256, kernel_size=3, padding=1, weight_init_scale=weight_init_scale), #4
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv6(256, 256, kernel_size=1, weight_init_scale=weight_init_scale), #5
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv6(256, 512, kernel_size=3, padding=1, weight_init_scale=weight_init_scale), #6
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv7–16: (1x1x256 + 3x3x512)x4 + 1x1x512 + 3x3x1024 + Maxpool:2x2-s-2
            MMFConv2dv6(512, 256, kernel_size=1, weight_init_scale=weight_init_scale), #7
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv6(256, 512, kernel_size=3, padding=1, weight_init_scale=weight_init_scale), #8
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv6(512, 256, kernel_size=1, weight_init_scale=weight_init_scale), #9
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv6(256, 512, kernel_size=3, padding=1, weight_init_scale=weight_init_scale), #10
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv6(512, 256, kernel_size=1, weight_init_scale=weight_init_scale), #11
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv6(256, 512, kernel_size=3, padding=1, weight_init_scale=weight_init_scale), #12
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv6(512, 256, kernel_size=1, weight_init_scale=weight_init_scale), #13
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv6(256, 512, kernel_size=3, padding=1, weight_init_scale=weight_init_scale), #14
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv6(512, 512, kernel_size=1, weight_init_scale=weight_init_scale), #15
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv6(512, 1024, kernel_size=3, padding=1, weight_init_scale=weight_init_scale), #16
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv17–20: (1x1x512 + 3x3x1024)x2
            MMFConv2dv6(1024, 512, kernel_size=1, weight_init_scale=weight_init_scale), #17
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv6(512, 1024, kernel_size=3, padding=1, weight_init_scale=weight_init_scale), #18
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv6(1024, 512, kernel_size=1, weight_init_scale=weight_init_scale), #19
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2dv6(512, 1024, kernel_size=3, padding=1, weight_init_scale=weight_init_scale), #20
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # Classification head: exactly as described in YOLOv1 paper (first 20 convs + avg pool + FC)
        self.head_classification = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),          # average-pooling layer to 1×1
            nn.Flatten(),                           # flatten [B, 1024, 1, 1] → [B, 1024]
            MMFLinearv6(1024, num_classes, weight_init_scale=weight_init_scale),  # single fully connected layer
        )

    def forward(self, x):
        # x: [B, 3, 32, 32]
        features = self.backbone_classification(x)        # → [B, 1024, 1, 1]
        logits = self.head_classification(features)       # → [B, 10]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        return logits
    
    def get_probs(self, x, temperature=1.0):
        logits = self(x)
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        return probs






###################### YOLOv1 Bbox ######################
class YOLOv1Bbox(nn.Module):
    """
    YOLOv1 architecture adapted for CIFAR-10 classification.
    Follows the exact conv layers from the original YOLOv1 paper
    (Redmon et al., 2016, arXiv:1506.02640), but replaces grid regression
    with a simple classification head (10 classes).
    
    Input:  [B, 3, 32, 32]  (CIFAR-10 RGB images)
    Output: [B, 10]         (class logits)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Backbone classification: 20 convolutional layers + maxpools (exact from YOLOv1 paper)
        self.backbone_classification = nn.Sequential(

            # Conv1: 7x7x64-s-2 + MaxPool: 2x2-s-2
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), #1
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv2: 3x3x192 + MaxPool: 2x2-s-2
            nn.Conv2d(64, 192, kernel_size=3, padding=1), #2
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv3–6: 1x1x128 + 3x3x256 + 1x1x256 + 3x3x512 + MaxPool: 2x2-s-2
            nn.Conv2d(192, 128, kernel_size=1), #3
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), #4
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 256, kernel_size=1), #5
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), #6
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv7–16: (1x1x256 + 3x3x512)x4 + 1x1x512 + 3x3x1024 + Maxpool:2x2-s-2
            nn.Conv2d(512, 256, kernel_size=1), #7
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), #8
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1), #9
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), #10
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1), #11
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), #12
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 256, kernel_size=1), #13
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1), #14
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 512, kernel_size=1), #15
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), #16
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv17–20: (1x1x512 + 3x3x1024)x2
            nn.Conv2d(1024, 512, kernel_size=1), #17
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), #18
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 512, kernel_size=1), #19
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), #20
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Backbone yolo: 4 additional convolutional layers + maxpools (exact from YOLOv1 paper)
        self.backbone_yolo = nn.Sequential(

            # Conv21–22: 3x3x1024 + 3x3x1024-s-2
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), #21
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1, stride=2), #22
            nn.LeakyReLU(0.1, inplace=True),

            # Conv23-24: 3x3x1024 + 3x3x1024
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), #23
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), #24
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        # YOLO head (adapted from YOLOv1's FC layers)
        self.head_yolo = nn.Sequential(
            # will do this later
            nn.Flatten(),
            nn.Linear(1024 * 1 * 1, 4096),  # after all pools → 1×1 feature map
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        # x: [B, 3, 32, 32]
        features = self.backbone_classification(x)        # → [B, 1024, 1, 1]
        features = self.backbone_yolo(features)
        logits = self.head_yolo(features)       # → [B, 10]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        return logits
    
    def get_probs(self, x, temperature=1.0):
        logits = self(x)
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        return probs
  
###################### YOLOv1 Bbox MMF ######################
class YOLOv1BboxMMF(nn.Module):
    """
    YOLOv1 architecture adapted for CIFAR-10 classification.
    Follows the exact conv layers from the original YOLOv1 paper
    (Redmon et al., 2016, arXiv:1506.02640), but replaces grid regression
    with a simple classification head (10 classes).
    
    Input:  [B, 3, 32, 32]  (CIFAR-10 RGB images)
    Output: [B, 10]         (class logits)
    """
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Backbone classification: 20 convolutional layers + maxpools (exact from YOLOv1 paper)
        self.backbone_classification = nn.Sequential(

            # Conv1: 7x7x64-s-2 + MaxPool: 2x2-s-2
            MMFConv2d(3, 64, kernel_size=7, stride=2, padding=3), #1
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv2: 3x3x192 + MaxPool: 2x2-s-2
            MMFConv2d(64, 192, kernel_size=3, padding=1), #2
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv3–6: 1x1x128 + 3x3x256 + 1x1x256 + 3x3x512 + MaxPool: 2x2-s-2
            MMFConv2d(192, 128, kernel_size=1), #3
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(128, 256, kernel_size=3, padding=1), #4
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 256, kernel_size=1), #5
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 512, kernel_size=3, padding=1), #6
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv7–16: (1x1x256 + 3x3x512)x4 + 1x1x512 + 3x3x1024 + Maxpool:2x2-s-2
            MMFConv2d(512, 256, kernel_size=1), #7
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 512, kernel_size=3, padding=1), #8
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 256, kernel_size=1), #9
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 512, kernel_size=3, padding=1), #10
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 256, kernel_size=1), #11
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 512, kernel_size=3, padding=1), #12
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 256, kernel_size=1), #13
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(256, 512, kernel_size=3, padding=1), #14
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 512, kernel_size=1), #15
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 1024, kernel_size=3, padding=1), #16
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Conv17–20: (1x1x512 + 3x3x1024)x2
            MMFConv2d(1024, 512, kernel_size=1), #17
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 1024, kernel_size=3, padding=1), #18
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(1024, 512, kernel_size=1), #19
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(512, 1024, kernel_size=3, padding=1), #20
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Backbone yolo: 4 additional convolutional layers + maxpools (exact from YOLOv1 paper)
        self.backbone_yolo = nn.Sequential(

            # Conv21–22: 3x3x1024 + 3x3x1024-s-2
            MMFConv2d(1024, 1024, kernel_size=3, padding=1), #21
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(1024, 1024, kernel_size=3, padding=1, stride=2), #22
            nn.LeakyReLU(0.1, inplace=True),

            # Conv23-24: 3x3x1024 + 3x3x1024
            MMFConv2d(1024, 1024, kernel_size=3, padding=1), #23
            nn.LeakyReLU(0.1, inplace=True),
            MMFConv2d(1024, 1024, kernel_size=3, padding=1), #24
            nn.LeakyReLU(0.1, inplace=True),
        )

        # YOLO head (adapted from YOLOv1's FC layers)
        self.head_yolo = nn.Sequential(
            # will do this later
            nn.Flatten(),
            MMFLinear(1024 * 1 * 1, 4096),  # after all pools → 1×1 feature map
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            MMFLinear(4096, num_classes)
        )

    def forward(self, x):
        # x: [B, 3, 32, 32]
        features = self.backbone_classification(x)        # → [B, 1024, 1, 1]
        features = self.backbone_yolo(features)
        logits = self.head_yolo(features)       # → [B, 10]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        return logits
    
    def get_probs(self, x, temperature=1.0):
        logits = self(x)
        scaled_logits = logits / temperature
        probs = torch.softmax(scaled_logits, dim=1)
        return probs
  




   
# Quick test / usage example
if __name__ == "__main__":
    model = YOLOv1Classifier()
    image = torch.randn(512, 3, 32, 32)
    out = model(image)
    print("Output shape:", out.shape) 
    print("Sample output min/max:", out.min().item(), out.max().item())