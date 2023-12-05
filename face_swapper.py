import time
import torch
import onnx
import cv2
import onnxruntime
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from onnx import numpy_helper
from skimage import transform as trans
import torchvision.transforms.functional as F
import torch.nn.functional as F
from utils import mask_crop, laplacian_blending


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

        self.session_options = onnxruntime.SessionOptions()
        self.session = onnxruntime.InferenceSession(self.model_file, sess_options=self.session_options, providers=providers)

    def forward(self, imgs, latents):
        preds = []
        for img, latent in zip(imgs, latents):
            img = img / 255
            pred = self.session.run(['output'], {'target': img, 'source': latent})[0]
            preds.append(pred)

    def get(self, imgs, target_faces, source_faces):
        imgs = list(imgs)

        preds = [None] * len(imgs)
        matrs = [None] * len(imgs)

        for idx, (img, target_face, source_face) in enumerate(zip(imgs, target_faces, source_faces)):
            matrix, blob, latent = self.prepare_data(img, target_face, source_face)
            pred = self.session.run(['output'], {'target': blob, 'source': latent})[0]
            pred = pred.transpose((0, 2, 3, 1))[0]
            pred = np.clip(255 * pred, 0, 255).astype(np.uint8)[:, :, ::-1]

            preds[idx] = pred
            matrs[idx] = matrix

        return (preds, matrs)

    def prepare_data(self, img, target_face, source_face):
        if isinstance(img, str):
            img = cv2.imread(img)

        aligned_img, matrix = norm_crop2(img, target_face.kps, 128)

        blob = cv2.dnn.blobFromImage(aligned_img, 1.0 / 255, (128, 128), (0., 0., 0.), swapRB=True)

        latent = source_face.normed_embedding.reshape((1, -1))
        latent = np.dot(latent, self.emap)
        latent /= np.linalg.norm(latent)

        return (matrix, blob, latent)

    def batch_forward(self, img_list, target_f_list, source_f_list):
        num_samples = len(img_list)
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size

        for i in tqdm(range(num_batches), desc="Generating face"):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, num_samples)

            batch_img = img_list[start_idx:end_idx]
            batch_target_f = target_f_list[start_idx:end_idx]
            batch_source_f = source_f_list[start_idx:end_idx]

            batch_pred, batch_matr = self.get(batch_img, batch_target_f, batch_source_f)

            yield batch_pred, batch_matr


def paste_to_whole(foreground, background, matrix, mask=None, crop_mask=(0,0,0,0), blur_amount=0.1, erode_amount = 0.15, blend_method='linear'):
    inv_matrix = cv2.invertAffineTransform(matrix)
    fg_shape = foreground.shape[:2]
    bg_shape = (background.shape[1], background.shape[0])
    foreground = cv2.warpAffine(foreground, inv_matrix, bg_shape, borderValue=0.0)

    if mask is None:
        mask = np.full(fg_shape, 1., dtype=np.float32)
        mask = mask_crop(mask, crop_mask)
        mask = cv2.warpAffine(mask, inv_matrix, bg_shape, borderValue=0.0)
    else:
        assert fg_shape == mask.shape[:2], "foreground & mask shape mismatch!"
        mask = mask_crop(mask, crop_mask).astype('float32')
        mask = cv2.warpAffine(mask, inv_matrix, (background.shape[1], background.shape[0]), borderValue=0.0)

    _mask = mask.copy()
    _mask[_mask > 0.05] = 1.
    non_zero_points = cv2.findNonZero(_mask)
    _, _, w, h = cv2.boundingRect(non_zero_points)
    mask_size = int(np.sqrt(w * h))

    if erode_amount > 0:
        kernel_size = max(int(mask_size * erode_amount), 1)
        structuring_element = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        mask = cv2.erode(mask, structuring_element)

    if blur_amount > 0:
        kernel_size = max(int(mask_size * blur_amount), 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)

    mask = np.tile(np.expand_dims(mask, axis=-1), (1, 1, 3))

    if blend_method == 'laplacian':
        composite_image = laplacian_blending(foreground, background, mask.clip(0,1), num_levels=4)
    else:
        composite_image = mask * foreground + (1 - mask) * background

    return composite_image.astype("uint8").clip(0, 255)
