import torch
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

run_with_cuda = False

def init_parser(pth_path, use_cuda=False):
    global run_with_cuda
    run_with_cuda = use_cuda

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    if run_with_cuda:
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
        if run_with_cuda:
            img = img.cuda()
        out = net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        return parsing


def get_mask(parsing, classes):
    res = parsing == classes[0]
    for val in classes[1:]:
        res += parsing == val
    return res

def swap_regions(source, target, net, includes=[1,2,3,4,5,10,11,12,13], excludes=[7,8], blur_size=25):
    parsing = image_to_parsing(source, net)
    if len(includes) == 0:
        return source, np.zeros_like(source)
    include_mask = get_mask(parsing, includes)
    include_mask = np.repeat(np.expand_dims(include_mask.astype('float32'), axis=2), 3, 2)
    if len(excludes) > 0:
        exclude_mask = get_mask(parsing, excludes)
        exclude_mask = np.repeat(np.expand_dims(exclude_mask.astype('float32'), axis=2), 3, 2)
        include_mask -= exclude_mask
    mask = 1 - cv2.GaussianBlur(include_mask.clip(0,1), (0, 0), blur_size)
    result = (1 - mask) * cv2.resize(source, (512, 512)) + mask * cv2.resize(target, (512, 512))
    result = cv2.resize(result.astype("float32"), (source.shape[1], source.shape[0]))
    return result, mask.astype('float32')

def mask_regions_to_list(values):
    out_ids = []
    for value in values:
        if value in mask_regions.keys():
            out_ids.append(mask_regions.get(value))
    return out_ids
