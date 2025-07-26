from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from timm import create_model
import os
from tqdm import tqdm

class SpiralFullDataset(Dataset):
    def __init__(self, root_dir, splits, transform):
        self.samples = []
        for lbl, cls in enumerate(['healthy','parkinson']):
            for split in splits:
                for fname in sorted(os.listdir(os.path.join(root_dir, split, cls))):
                    if fname.lower().endswith(('.png','.jpg','.jpeg')):
                        path = os.path.join(root_dir, split, cls, fname)
                        self.samples.append((path, lbl))
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, lbl = self.samples[idx]
        img = Image.open(path).convert('L')
        return self.transform(img), lbl

spiral_full_root = '/content/drive/MyDrive/dataset/spiral_edge_enhanced'
spiral_ds = SpiralFullDataset(spiral_full_root, ['training','validation'], single_transform)
spiral_loader = DataLoader(spiral_ds, batch_size=16, shuffle=True, num_workers=0)
print("Spiral full samples:", len(spiral_ds))

spiral_model = create_model('vit_base_patch16_224', pretrained=True)
spiral_model.reset_classifier(1)
spiral_model = spiral_model.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(spiral_model.parameters(), lr=1e-4, weight_decay=1e-3)

for epoch in range(1, 6):
    spiral_model.train()
    total_loss = 0
    for imgs, lbls in tqdm(spiral_loader, desc=f"Spiral Full Train E{epoch}"):
        imgs = imgs.to(DEVICE)
        lbls = lbls.to(DEVICE).float().unsqueeze(1)
        optimizer.zero_grad()
        out = spiral_model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch} avg loss: {total_loss/len(spiral_loader):.4f}")

ckpt_path = '/content/drive/MyDrive/dataset/spiral_model.pth'
torch.save(spiral_model.state_dict(), ckpt_path)
print("Saved spiral_model to", ckpt_path)
