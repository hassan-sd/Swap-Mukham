from torchvision.transforms import Normalize
import torchvision.transforms as T
import torch.nn as nn
from PIL import Image
import numpy as np
import torch
import timm
from tqdm import tqdm

# https://github.com/Whiax/NSFW-Classifier/raw/main/nsfwmodel_281.pth
normalize_t = Normalize((0.4814, 0.4578, 0.4082), (0.2686, 0.2613, 0.2757))

#nsfw classifier
class NSFWClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        nsfw_model=self
        nsfw_model.root_model = timm.create_model('convnext_base_in22ft1k', pretrained=True)
        nsfw_model.linear_probe = nn.Linear(1024, 1, bias=False)

    def forward(self, x):
        nsfw_model = self
        x = normalize_t(x)
        x = nsfw_model.root_model.stem(x)
        x = nsfw_model.root_model.stages(x)
        x = nsfw_model.root_model.head.global_pool(x)
        x = nsfw_model.root_model.head.norm(x)
        x = nsfw_model.root_model.head.flatten(x)
        x = nsfw_model.linear_probe(x)
        return x

    def is_nsfw(self, img_paths, threshold = 0.98):
        skip_step = 1
        total_len = len(img_paths)
        if total_len < 100: skip_step = 1
        if total_len > 100 and total_len < 500: skip_step = 10
        if total_len > 500 and total_len < 1000: skip_step = 20
        if total_len > 1000 and total_len < 10000: skip_step = 50
        if total_len > 10000: skip_step = 100

        for idx in tqdm(range(0, total_len, skip_step), total=int(total_len // skip_step), desc="Checking for NSFW contents"):
            _img = Image.open(img_paths[idx]).convert('RGB')
            img = _img.resize((224, 224))
            img = np.array(img)/255
            img = T.ToTensor()(img).unsqueeze(0).float()
            if next(self.parameters()).is_cuda:
                img = img.cuda()
            with torch.no_grad():
                score = self.forward(img).sigmoid()[0].item()
            if score > threshold:
                print(f"Detected nsfw score:{score}")
                _img.save("nsfw.jpg")
                return True
        return False

def get_nsfw_detector(model_path='nsfwmodel_281.pth', device="cpu"):
    #load base model
    nsfw_model = NSFWClassifier()
    nsfw_model = nsfw_model.eval()
    #load linear weights
    linear_pth = model_path
    linear_state_dict = torch.load(linear_pth, map_location='cpu')
    nsfw_model.linear_probe.load_state_dict(linear_state_dict)
    nsfw_model = nsfw_model.to(device)
    return nsfw_model
