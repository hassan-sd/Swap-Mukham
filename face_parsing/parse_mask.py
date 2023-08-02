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
        batch_size = x.size(0)  # Get the batch size
        output = []

        for i in tqdm(range(batch_size), desc="Soft-Erosion", leave=False):
            input_tensor = x[i:i+1]  # Take one input tensor from the batch
            input_tensor = input_tensor.float()  # Convert input to float tensor
            input_tensor = input_tensor.unsqueeze(1)  # Add a channel dimension

            for _ in range(self.iterations - 1):
                input_tensor = torch.min(input_tensor, F.conv2d(input_tensor, weight=self.weight,
                                                                groups=input_tensor.shape[1],
                                                                padding=self.padding))
            input_tensor = F.conv2d(input_tensor, weight=self.weight, groups=input_tensor.shape[1],
                                    padding=self.padding)

            mask = input_tensor >= self.threshold
            input_tensor[mask] = 1.0
            input_tensor[~mask] /= input_tensor[~mask].max()

            input_tensor = input_tensor.squeeze(1)  # Remove the extra channel dimension
            output.append(input_tensor.detach().cpu().numpy())

        return np.array(output)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])



def init_parsing_model(model_path, device="cpu"):
    net = BiSeNet(19)
    net.to(device)
    if device == 'cpu':
        model = torch.load(model_path, map_location='cpu')
    else:
        model = torch.load(model_path)
    net.load_state_dict(model)
    net.eval()
    return net

def transform_images(imgs):
    tensor_images = torch.stack([transform(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))) for img in imgs], dim=0)
    return tensor_images

def get_parsed_mask(net, imgs, classes=[1, 2, 3, 4, 5, 10, 11, 12, 13], device="cpu", batch_size=8, softness=20):
    if softness > 0:
        smooth_mask = SoftErosion(kernel_size=17, threshold=0.9, iterations=softness).to(device)

    masks = []
    for i in tqdm(range(0, len(imgs), batch_size), total=len(imgs) // batch_size, desc="Face-parsing"):
        batch_imgs = imgs[i:i + batch_size]

        tensor_images = transform_images(batch_imgs).to(device)
        with torch.no_grad():
            out = net(tensor_images)[0]
        # parsing = out.argmax(dim=1)
        # arget_classes = torch.tensor(classes).to(device)
        # batch_masks = torch.isin(parsing, target_classes).to(device)
        ## torch.isin was slightly slower in my test, so using np.isin
        parsing = out.argmax(dim=1).detach().cpu().numpy()
        batch_masks = np.isin(parsing, classes).astype('float32')

        if softness > 0:
            # batch_masks = smooth_mask(batch_masks).transpose(1,0,2,3)[0]
            mask_tensor = torch.from_numpy(batch_masks.copy()).float().to(device)
            batch_masks = smooth_mask(mask_tensor).transpose(1,0,2,3)[0]

        yield batch_masks

        #masks.append(batch_masks)

    #if len(masks) >= 1:
    #    masks = np.concatenate(masks, axis=0)
    # masks = np.repeat(np.expand_dims(masks, axis=1), 3, axis=1)

    # for i, mask in enumerate(masks):
    #    cv2.imwrite(f"mask/{i}.jpg", (mask * 255).astype("uint8"))

    #return masks
