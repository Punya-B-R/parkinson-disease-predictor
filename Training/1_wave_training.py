import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image

class WaveFullDataset(Dataset):
    def __init__(self, root_dir, splits, transform):
        self.samples = []
        for label_idx, cls in enumerate(['healthy','parkinson']):
            for split in splits:
                cls_dir = os.path.join(root_dir, split, cls)
                for fname in sorted(os.listdir(cls_dir)):
                    if fname.lower().endswith(('.png','.jpg','.jpeg')):
                        path = os.path.join(cls_dir, fname)
                        self.samples.append((path, label_idx))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('L')
        img = single_transform(img)
        return img, label

wave_full_ds = WaveFullDataset(
    root_dir=wave_root,
    splits=['training','validation'],
    transform=single_transform
)
wave_full_loader = DataLoader(
    wave_full_ds, batch_size=16, shuffle=True, num_workers=0
)

print("Combined wave samples:", len(wave_full_ds))

wave_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
wave_model.fc = nn.Linear(wave_model.fc.in_features, 1)
wave_model = wave_model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(wave_model.parameters(), lr=1e-4, weight_decay=1e-3)

EPOCHS = 5
for epoch in range(1, EPOCHS+1):
    wave_model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(wave_full_loader, desc=f"Wave Full Train E{epoch}"):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)
        optimizer.zero_grad()
        out = wave_model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch}: Avg Loss = {running_loss/len(wave_full_loader):.4f}")

save_path = '/content/drive/MyDrive/dataset/wave_model.pth'
torch.save(wave_model.state_dict(), save_path)
print("Saved wave_model to", save_path)