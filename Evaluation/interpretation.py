import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class WaveWrapper(nn.Module):
    def __init__(self, hybrid_model, spiral_img):
        super().__init__()
        self.hybrid = hybrid_model
        self.spiral = spiral_img
    def forward(self, wave_img):
        return self.hybrid(wave_img, self.spiral)

class SpiralWrapper(nn.Module):
    def __init__(self, hybrid_model, wave_img):
        super().__init__()
        self.hybrid = hybrid_model
        self.wave   = wave_img
    def forward(self, spiral_img):
        return self.hybrid(self.wave, spiral_img)

wave_target   = model.resnet.layer4[-1]
spiral_target = model.vit.patch_embed.proj

count = 0
for (waves, spirals), labels in test_loader:
    waves, spirals, labels = waves.to(DEVICE), spirals.to(DEVICE), labels.to(DEVICE)
    with torch.no_grad():
        logits = model(waves, spirals)
        probs  = torch.sigmoid(logits).flatten()
        preds  = (probs > 0.5).int()
    for i in range(len(labels)):
        if labels[i] == 1 and preds[i] == 1:
            w_img = waves[i].unsqueeze(0)
            s_img = spirals[i].unsqueeze(0)

            w_np = w_img.cpu().numpy()[0].transpose(1,2,0)
            w_np = (w_np - w_np.min()) / (w_np.max() - w_np.min())
            s_np = s_img.cpu().numpy()[0].transpose(1,2,0)
            s_np = (s_np - s_np.min()) / (s_np.max() - s_np.min())

            wave_wrapper = WaveWrapper(model, s_img)
            cam_wave = GradCAMPlusPlus(model=wave_wrapper, target_layers=[wave_target])
            mask_w   = cam_wave(input_tensor=w_img, targets=[ClassifierOutputTarget(0)])[0]
            cam_w_img= show_cam_on_image(w_np, mask_w, use_rgb=True)

            spiral_wrapper = SpiralWrapper(model, w_img)
            cam_spiral = GradCAMPlusPlus(model=spiral_wrapper, target_layers=[spiral_target])
            mask_s   = cam_spiral(input_tensor=s_img, targets=[ClassifierOutputTarget(0)])[0]
            cam_s_img= show_cam_on_image(s_np, mask_s, use_rgb=True)

            fig, axs = plt.subplots(1,4, figsize=(16,4), dpi=300)
            fig.suptitle(f"Parkinson’s TP #{count+1}", fontsize=14)

            axs[0].imshow(w_np)
            axs[0].set_title("Wave Original")
            axs[1].imshow(cam_w_img)
            axs[1].set_title("Wave GradCAM++")
            axs[2].imshow(s_np)
            axs[2].set_title("Spiral Original")
            axs[3].imshow(cam_s_img)
            axs[3].set_title("Spiral GradCAM++")

            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.show()

            count += 1
            if count >= 5:  
                break
    if count >= 5:
        break

if count == 0:
    print("No Parkinson’s true positives found!")
