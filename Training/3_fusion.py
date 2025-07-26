import torch.nn as nn
from torchvision import models
from timm import create_model

wave_ckpt_path   = '/content/drive/MyDrive/dataset/wave_model.pth'
spiral_ckpt_path = '/content/drive/MyDrive/dataset/spiral_model.pth'

class HybridFusionModel(nn.Module):
    def __init__(self, wave_ckpt, spiral_ckpt, proj_dim=256):
        super().__init__()
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Identity()
        state_w = torch.load(wave_ckpt, map_location='cpu')
        self.resnet.load_state_dict(state_w, strict=False)

        self.vit = create_model('vit_base_patch16_224', pretrained=False)
        self.vit.reset_classifier(0)
        state_s = torch.load(spiral_ckpt, map_location='cpu')
        self.vit.load_state_dict(state_s, strict=False)

        self.proj_res = nn.Linear(512, proj_dim)
        self.proj_vit = nn.Linear(768, proj_dim)

        self.classifier = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(proj_dim, 1)
        )

    def forward(self, wave_img, spiral_img):
        feat_w = self.resnet(wave_img)     # [B,512]
        feat_s = self.vit(spiral_img)      # [B,768]
        p_w = self.proj_res(feat_w)        # [B,proj_dim]
        p_s = self.proj_vit(feat_s)        # [B,proj_dim]
        fused = torch.cat([p_w, p_s], dim=1)  # [B,2*proj_dim]
        return self.classifier(fused)        # [B,1]

model = HybridFusionModel(wave_ckpt_path, spiral_ckpt_path, proj_dim=256).to(DEVICE)
print("Hybrid model initialized with pretrained wave & spiral branches.")