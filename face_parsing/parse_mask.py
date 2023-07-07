import cv2
import torch
import torchvision
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as transforms

from . model import BiSeNet

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

def init_parsing_model(model_path, device="cpu"):
    net = BiSeNet(19)
    net.to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    return net

def transform_images(imgs):
    tensor_images = torch.stack([transform(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))) for img in imgs], dim=0)
    return tensor_images

def get_parsed_mask(net, imgs, classes=[1, 2, 3, 4, 5, 10, 11, 12, 13], device="cpu", batch_size=8):
    masks = []
    for i in tqdm(range(0, len(imgs), batch_size), total=len(imgs) // batch_size, desc="Face-parsing"):
        batch_imgs = imgs[i:i + batch_size]

        tensor_images = transform_images(batch_imgs).to(device)
        with torch.no_grad():
            out = net(tensor_images)[0]
        parsing = out.argmax(dim=1).cpu().numpy()
        batch_masks = np.isin(parsing, classes)

        masks.append(batch_masks)

    masks = np.concatenate(masks, axis=0)
    # masks = np.repeat(np.expand_dims(masks, axis=1), 3, axis=1)

    for i, mask in enumerate(masks):
        cv2.imwrite(f"mask/{i}.jpg", (mask * 255).astype("uint8"))

    return masks

