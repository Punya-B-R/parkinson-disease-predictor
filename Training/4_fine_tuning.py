import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

paired_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

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
                self.samples.append((os.path.join(w_dir, w_files[i]),
                                     os.path.join(s_dir, s_files[i]),
                                     lbl))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        w_path, s_path, lbl = self.samples[idx]
        w = self.transform(Image.open(w_path).convert('L'))
        s = self.transform(Image.open(s_path).convert('L'))
        return (w, s), lbl

batch_size = 4
train_ds = PairedDataset('/content/drive/MyDrive/dataset/wave_edge_enhanced', '/content/drive/MyDrive/dataset/spiral_edge_enhanced', 'training', paired_transform)
val_ds   = PairedDataset('/content/drive/MyDrive/dataset/wave_edge_enhanced', '/content/drive/MyDrive/dataset/spiral_edge_enhanced', 'validation', paired_transform)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

print("Paired train/val samples:", len(train_ds), "/", len(val_ds))

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW([
    {'params': model.resnet.parameters(),    'lr': 1e-5},
    {'params': model.vit.parameters(),       'lr': 1e-5},
    {'params': model.proj_res.parameters(),  'lr': 1e-4},
    {'params': model.proj_vit.parameters(),  'lr': 1e-4},
    {'params': model.classifier.parameters(),'lr': 1e-4},
], weight_decay=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)

best_val_loss = float('inf')
for epoch in range(1, 6):
    model.train()
    train_loss = 0
    for (w, s), lbl in tqdm(train_loader, desc=f"E{epoch} train"):
        w, s, lbl = w.to(DEVICE), s.to(DEVICE), lbl.to(DEVICE).float().unsqueeze(1)
        optimizer.zero_grad()
        loss = criterion(model(w, s), lbl)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    print(f"Epoch {epoch} Train Loss: {train_loss/len(train_loader):.4f}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for (w, s), lbl in tqdm(val_loader, desc=f"E{epoch} val"):
            w, s, lbl = w.to(DEVICE), s.to(DEVICE), lbl.to(DEVICE).float().unsqueeze(1)
            val_loss += criterion(model(w, s), lbl).item()
    val_loss /= len(val_loader)
    print(f"Epoch {epoch} Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'hybrid_model.pth')
        print("Saved best hybrid_model.pth")

    scheduler.step(val_loss)

print("Fine-tuning complete; best val loss:", best_val_loss)