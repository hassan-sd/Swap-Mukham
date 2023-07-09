import time
import torch
import onnx
import cv2
import onnxruntime
import numpy as np
from tqdm import tqdm
from onnx import numpy_helper
from skimage import transform as trans
import torchvision.transforms.functional as F
from utils import make_white_image, laplacian_blending

arcface_dst = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32)

def estimate_norm(lmk, image_size=112, mode='arcface'):
    assert lmk.shape == (5, 2)
    assert image_size % 112 == 0 or image_size % 128 == 0
    if image_size % 112 == 0:
        ratio = float(image_size) / 112.0
        diff_x = 0
    else:
        ratio = float(image_size) / 128.0
        diff_x = 8.0 * ratio
    dst = arcface_dst * ratio
    dst[:, 0] += diff_x
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, dst)
    M = tform.params[0:2, :]
    return M


def norm_crop2(img, landmark, image_size=112, mode='arcface'):
    M = estimate_norm(landmark, image_size, mode)
    warped = cv2.warpAffine(img, M, (image_size, image_size), borderValue=0.0)
    return warped, M


class Inswapper():
    def __init__(self, model_file=None, batch_size=32, providers=['CPUExecutionProvider']):
        self.model_file = model_file
        self.batch_size = batch_size

        model = onnx.load(self.model_file)
        graph = model.graph
        self.emap = numpy_helper.to_array(graph.initializer[-1])
        self.input_mean = 0.0
        self.input_std = 255.0

        self.session_options = onnxruntime.SessionOptions()
        self.session = onnxruntime.InferenceSession(self.model_file, sess_options=self.session_options, providers=providers)

        inputs = self.session.get_inputs()
        self.input_names = [inp.name for inp in inputs]
        outputs = self.session.get_outputs()
        self.output_names = [out.name for out in outputs]
        assert len(self.output_names) == 1
        self.output_shape = outputs[0].shape
        input_cfg = inputs[0]
        input_shape = input_cfg.shape
        self.input_shape = input_shape
        self.input_size = tuple(input_shape[2:4][::-1])

    def forward(self, imgs, latents):
        preds = []
        for img, latent in zip(imgs, latents):
            img = (img - self.input_mean) / self.input_std
            pred = self.session.run(self.output_names, {self.input_names[0]: img, self.input_names[1]: latent})[0]
            preds.append(pred)

    def get(self, imgs, target_faces, source_faces):
        imgs = list(imgs)

        preds = [None] * len(imgs)
        aimgs = [None] * len(imgs)
        matrs = [None] * len(imgs)

        for idx, (img, target_face, source_face) in enumerate(zip(imgs, target_faces, source_faces)):
            aimg, M, blob, latent = self.prepare_data(img, target_face, source_face)
            aimgs[idx] = aimg
            matrs[idx] = M
            pred = self.session.run(self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent})[0]
            pred = pred.transpose((0, 2, 3, 1))[0]
            pred = np.clip(255 * pred, 0, 255).astype(np.uint8)[:, :, ::-1]
            preds[idx] = pred

        return (preds, aimgs, matrs)

    def prepare_data(self, img, target_face, source_face):
        if isinstance(img, str):
            img = cv2.imread(img)

        aimg, M = norm_crop2(img, target_face.kps, self.input_size[0])

        blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.input_std, self.input_size,
                (self.input_mean, self.input_mean, self.input_mean), swapRB=True)

        latent = source_face.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)

        return (aimg, M, blob, latent)

    def batch_forward(self, img_list, target_f_list, source_f_list):
        num_samples = len(img_list)
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size

        preds = []
        aimgs = []
        matrs = []

        for i in tqdm(range(num_batches), desc="Swapping face"):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, num_samples)

            batch_img = img_list[start_idx:end_idx]
            batch_target_f = target_f_list[start_idx:end_idx]
            batch_source_f = source_f_list[start_idx:end_idx]

            batch_pred, batch_aimg, batch_matr = self.get(batch_img, batch_target_f, batch_source_f)
            preds.extend(batch_pred)
            aimgs.extend(batch_aimg)
            matrs.extend(batch_matr)

        return (preds, aimgs, matrs)


def paste_to_whole(bgr_fake, aimg, M, whole_img, laplacian_blend=True, crop_mask=(0,0,0,0)):
    IM = cv2.invertAffineTransform(M)

    img_white = make_white_image(aimg.shape[:2], crop=crop_mask, white_value=255)

    bgr_fake = cv2.warpAffine(bgr_fake, IM, (whole_img.shape[1], whole_img.shape[0]), borderValue=0.0)
    img_white = cv2.warpAffine(img_white, IM, (whole_img.shape[1], whole_img.shape[0]), borderValue=0.0)

    img_white[img_white > 20] = 255
    img_mask = img_white
    mask_h_inds, mask_w_inds = np.where(img_mask == 255)
    mask_size = int(np.sqrt(np.ptp(mask_h_inds) * np.ptp(mask_w_inds)))

    k = max(mask_size // 10, 10)
    img_mask = cv2.erode(img_mask, np.ones((k, k), np.uint8), iterations=1)

    k = max(mask_size // 20, 5)
    kernel_size = (k, k)
    blur_size = tuple(2 * i + 1 for i in kernel_size)
    img_mask = cv2.GaussianBlur(img_mask, blur_size, 0) / 255
    img_mask = np.tile(np.expand_dims(img_mask, axis=-1), (1, 1, 3))

    if laplacian_blend:
        bgr_fake = laplacian_blending(bgr_fake.astype("float32").clip(0,255), whole_img.astype("float32").clip(0,255), img_mask.clip(0,1))
        bgr_fake = bgr_fake.astype("float32")

    fake_merged = img_mask * bgr_fake + (1 - img_mask) * whole_img.astype(np.float32)
    return fake_merged.astype("uint8")

def place_foreground_on_background(foreground, background, matrix):
    matrix = cv2.invertAffineTransform(matrix)
    mask = np.ones(foreground.shape, dtype="float32")
    foreground = cv2.warpAffine(foreground, matrix, (background.shape[1], background.shape[0]), borderValue=0.0)
    mask = cv2.warpAffine(mask, matrix, (background.shape[1], background.shape[0]), borderValue=0.0)
    composite_image = mask * foreground + (1 - mask) * background
    return composite_image