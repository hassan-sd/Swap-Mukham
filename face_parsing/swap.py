import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np

from .model import BiSeNet

mask_regions = {
    "Background":0,
    "Skin":1,
    "L-Eyebrow":2,
    "R-Eyebrow":3,
    "L-Eye":4,
    "R-Eye":5,
    "Eye-G":6,
    "L-Ear":7,
    "R-Ear":8,
    "Ear-R":9,
    "Nose":10,
    "Mouth":11,
    "U-Lip":12,
    "L-Lip":13,
    "Neck":14,
    "Neck-L":15,
    "Cloth":16,
    "Hair":17,
    "Hat":18
}

# Borrowed from simswap
# https://github.com/neuralchen/SimSwap/blob/26c84d2901bd56eda4d5e3c5ca6da16e65dc82a6/util/reverse2original.py#L30
class SoftErosion(nn.Module):
    def __init__(self, kernel_size=15, threshold=0.6, iterations=1):
        super(SoftErosion, self).__init__()
        r = kernel_size // 2
        self.padding = r
        self.iterations = iterations
        self.threshold = threshold

        # Create kernel
        y_indices, x_indices = torch.meshgrid(torch.arange(0., kernel_size), torch.arange(0., kernel_size))
        dist = torch.sqrt((x_indices - r) ** 2 + (y_indices - r) ** 2)
        kernel = dist.max() - dist
        kernel /= kernel.sum()
        kernel = kernel.view(1, 1, *kernel.shape)
        self.register_buffer('weight', kernel)

    def forward(self, x):
        x = x.float()
        for i in range(self.iterations - 1):
            x = torch.min(x, F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding))
        x = F.conv2d(x, weight=self.weight, groups=x.shape[1], padding=self.padding)

        mask = x >= self.threshold
        x[mask] = 1.0
        x[~mask] /= x[~mask].max()

        return x, mask

device = "cpu"

def init_parser(pth_path, mode="cpu"):
    global device
    device = mode
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    if device == "cuda":
        net.cuda()
        net.load_state_dict(torch.load(pth_path))
    else:
        net.load_state_dict(torch.load(pth_path, map_location=torch.device('cpu')))
    net.eval()
    return net


def image_to_parsing(img, net):
    img = cv2.resize(img, (512, 512))
    img = img[:,:,::-1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    img = transform(img.copy())
    img = torch.unsqueeze(img, 0)

    with torch.no_grad():
        img = img.to(device)
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        return parsing


def get_mask(parsing, classes):
    res = parsing == classes[0]
    for val in classes[1:]:
        res += parsing == val
    return res

def swap_regions(source, target, net, smooth_mask, includes=[1,2,3,4,5,10,11,12,13], blur=10):
    parsing = image_to_parsing(source, net)

    if len(includes) == 0:
        return source, np.zeros_like(source)

    include_mask = get_mask(parsing, includes)
    mask = np.repeat(include_mask[:, :, np.newaxis], 3, axis=2).astype("float32")

    if smooth_mask is not None:
        mask_tensor = torch.from_numpy(mask.copy().transpose((2, 0, 1))).float().to(device)
        face_mask_tensor = mask_tensor[0] + mask_tensor[1]
        soft_face_mask_tensor, _ = smooth_mask(face_mask_tensor.unsqueeze_(0).unsqueeze_(0))
        soft_face_mask_tensor.squeeze_()
        mask = np.repeat(soft_face_mask_tensor.cpu().numpy()[:, :, np.newaxis], 3, axis=2)

    if blur > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), blur)

    resized_source = cv2.resize((source/255).astype("float32"), (512, 512))
    resized_target = cv2.resize((target/255).astype("float32"), (512, 512))

    result = mask * resized_source + (1 - mask) * resized_target
    normalized_result = (result - np.min(result)) / (np.max(result) - np.min(result))
    result = cv2.resize((result*255).astype("uint8"), (source.shape[1], source.shape[0]))

    return result

def mask_regions_to_list(values):
    out_ids = []
    for value in values:
        if value in mask_regions.keys():
            out_ids.append(mask_regions.get(value))
    return out_ids
