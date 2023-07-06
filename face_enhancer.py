import os
import torch
import gfpgan
from PIL import Image
from upscaler.RealESRGAN import RealESRGAN

face_enhancer_list = ['NONE', 'GFPGAN', 'REAL-ESRGAN 2x', 'REAL-ESRGAN 4x', 'REAL-ESRGAN 8x']

def load_face_enhancer_model(name='GFPGAN', device="cpu"):
    if name == 'GFPGAN':
        model_path = "./assets/pretrained_models/GFPGANv1.4.pth"
        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_path)
        model = gfpgan.GFPGANer(model_path=model_path, upscale=1)
    elif name == 'REAL-ESRGAN 2x':
        model_path = "./assets/pretrained_models/RealESRGAN_x2.pth"
        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_path)
        model = RealESRGAN(device, scale=2)
        model.load_weights(model_path, download=False)
    elif name == 'REAL-ESRGAN 4x':
        model_path = "./assets/pretrained_models/RealESRGAN_x4.pth"
        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_path)
        model = RealESRGAN(device, scale=4)
        model.load_weights(model_path, download=False)
    elif name == 'REAL-ESRGAN 8x':
        model_path = "./assets/pretrained_models/RealESRGAN_x8.pth"
        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model_path)
        model = RealESRGAN(device, scale=8)
        model.load_weights(model_path, download=False)
    else:
        model = None
    return model

def gfpgan_enhance(img, model, has_aligned=True):
    _, imgs, _ = model.enhance(img, paste_back=True, has_aligned=has_aligned)
    return imgs[0]

def realesrgan_enhance(img, model):
    img = model.predict(img)
    return img