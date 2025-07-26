import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PairedDataset(Dataset):
    def __init__(self, wave_root, spiral_root, split, transform):
        self.samples = []
        for lbl, cls in enumerate(['healthy','parkinson']):
            w_dir = os.path.join(wave_root, split, cls)
            s_dir = os.path.join(spiral_root, split, cls)
            w_files = sorted(f for f in os.listdir(w_dir) if f.lower().endswith(('.png','.jpg')))
            s_files = sorted(f for f in os.listdir(s_dir) if f.lower().endswith(('.png','.jpg')))
            n = min(len(w_files), len(s_files))
            for i in range(n):
                self.samples.append((
                    os.path.join(w_dir,   w_files[i]),
                    os.path.join(s_dir,   s_files[i]),
                    lbl
                ))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        w_path, s_path, lbl = self.samples[idx]
        w = self.transform(Image.open(w_path).convert('L'))
        s = self.transform(Image.open(s_path).convert('L'))
        return (w, s), lbl

paired_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

wave_root   = '/content/drive/MyDrive/dataset/wave_edge_enhanced'
spiral_root = '/content/drive/MyDrive/dataset/spiral_edge_enhanced'
test_ds  = PairedDataset(wave_root, spiral_root, 'testing', paired_transform)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)
print(f"Test pairs: {len(test_ds)}")

from torchvision import models
from timm import create_model
class HybridFusionModel(torch.nn.Module):
    def __init__(self, wave_ckpt, spiral_ckpt, proj_dim=256):
        super().__init__()
        # ResNet branch
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = torch.nn.Identity()
        self.resnet.load_state_dict(torch.load(wave_ckpt, map_location='cpu'), strict=False)
        # ViT branch
        self.vit = create_model('vit_base_patch16_224', pretrained=False)
        self.vit.reset_classifier(0)
        self.vit.load_state_dict(torch.load(spiral_ckpt, map_location='cpu'), strict=False)
        # Projections & classifier
        self.proj_res = torch.nn.Linear(512, proj_dim)
        self.proj_vit = torch.nn.Linear(768, proj_dim)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(proj_dim*2, proj_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(proj_dim, 1)
        )
    def forward(self, w, s):
        pr = self.proj_res(self.resnet(w))
        pv = self.proj_vit(self.vit(s))
        return self.classifier(torch.cat([pr,pv], dim=1))

wave_ckpt   = '/content/drive/MyDrive/dataset/wave_model.pth'
spiral_ckpt = '/content/drive/MyDrive/dataset/spiral_model.pth'
hybrid_ckpt = '/content/drive/MyDrive/dataset/hybrid_model.pth'

model = HybridFusionModel(wave_ckpt, spiral_ckpt).to(DEVICE)
model.load_state_dict(torch.load(hybrid_ckpt, map_location=DEVICE))
model.eval()

all_labels, all_preds, all_probs = [], [], []
with torch.no_grad():
    for (w_batch, s_batch), labels in test_loader:
        w_batch, s_batch = w_batch.to(DEVICE), s_batch.to(DEVICE)
        logits = model(w_batch, s_batch)
        probs  = torch.sigmoid(logits).cpu().numpy().flatten()
        preds  = (probs > 0.5).astype(int)
        all_labels.extend(labels.numpy().tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

y_true = np.array(all_labels)
y_pred = np.array(all_preds)
y_prob = np.array(all_probs)

print("Accuracy: ", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall:   ", recall_score(y_true, y_pred))
print("F1 Score: ", f1_score(y_true, y_pred))
print("ROC AUC:  ", roc_auc_score(y_true, y_prob))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['healthy','parkinson']))

for cls, name in [(0,'healthy'), (1,'parkinson')]:
    mask = (y_true == cls)
    acc_cls = (y_pred[mask] == cls).sum() / mask.sum()
    print(f"Accuracy ({name}): {acc_cls:.4f} [{int(mask.sum())} samples]")