import time
import torch
import onnx
import cv2
import onnxruntime
import numpy as np
from tqdm import tqdm
from onnx import numpy_helper
from skimage import transform as trans

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
        batch_preds = []
        for img, latent in zip(imgs, latents):
            img = (img - self.input_mean) / self.input_std
            pred = self.session.run(self.output_names, {self.input_names[0]: img, self.input_names[1]: latent})[0]
            batch_preds.append(pred)
        return batch_preds

    def get(self, imgs, target_faces, source_faces):
        batch_preds = []
        batch_aimgs = []
        batch_ms = []
        for img, target_face, source_face in zip(imgs, target_faces, source_faces):
            if isinstance(img, str):
                img = cv2.imread(img)
            aimg, M = norm_crop2(img, target_face.kps, self.input_size[0])
            blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.input_std, self.input_size,
                                         (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
            latent = source_face.normed_embedding.reshape((1, -1))
            latent = np.dot(latent, self.emap)
            latent /= np.linalg.norm(latent)
            pred = self.session.run(self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent})[0]
            pred = pred.transpose((0, 2, 3, 1))[0]
            pred = np.clip(255 * pred, 0, 255).astype(np.uint8)[:, :, ::-1]
            batch_preds.append(pred)
            batch_aimgs.append(aimg)
            batch_ms.append(M)
        return batch_preds, batch_aimgs, batch_ms

    def batch_forward(self, img_list, target_f_list, source_f_list):
        num_samples = len(img_list)
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size

        preds = []
        aimgs = []
        ms = []
        for i in tqdm(range(num_batches), desc="Swapping face by batch"):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, num_samples)

            batch_img = img_list[start_idx:end_idx]
            batch_target_f = target_f_list[start_idx:end_idx]
            batch_source_f = source_f_list[start_idx:end_idx]

            batch_pred, batch_aimg, batch_m = self.get(batch_img, batch_target_f, batch_source_f)
            preds.extend(batch_pred)
            aimgs.extend(batch_aimg)
            ms.extend(batch_m)
        return preds, aimgs, ms


def laplacian_blending(A, B, m, num_levels=4):
    assert A.shape == B.shape
    assert B.shape == m.shape
    height = m.shape[0]
    width = m.shape[1]
    size_list = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    size = size_list[np.where(size_list > max(height, width))][0]
    GA = np.zeros((size, size, 3), dtype=np.float32)
    GA[:height, :width, :] = A
    GB = np.zeros((size, size, 3), dtype=np.float32)
    GB[:height, :width, :] = B
    GM = np.zeros((size, size, 3), dtype=np.float32)
    GM[:height, :width, :] = m
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))
    lpA  = [gpA[num_levels-1]]
    lpB  = [gpB[num_levels-1]]
    gpMr = [gpM[num_levels-1]]
    for i in range(num_levels-1,0,-1):
        LA = np.subtract(gpA[i-1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i-1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i-1])
    LS = []
    for la,lb,gm in zip(lpA,lpB,gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)
    ls_ = LS[0]
    for i in range(1,num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    ls_ = np.clip(ls_[:height, :width, :], 0, 255)
    return ls_


def paste_to_whole(bgr_fake, aimg, M, whole_img, laplacian_blend=True, crop_mask=(0,0,0,0)):
    IM = cv2.invertAffineTransform(M)

    img_white = np.full((aimg.shape[0], aimg.shape[1]), 255, dtype=np.float32)

    top = int(crop_mask[0])
    bottom = int(crop_mask[1])
    if top + bottom < aimg.shape[1]:
        if top > 0: img_white[:top, :] = 0
        if bottom > 0: img_white[-bottom:, :] = 0

    left = int(crop_mask[2])
    right = int(crop_mask[3])
    if left + right < aimg.shape[0]:
        if left > 0: img_white[:, :left] = 0
        if right > 0: img_white[:, -right:] = 0

    bgr_fake = cv2.warpAffine(
        bgr_fake, IM, (whole_img.shape[1], whole_img.shape[0]), borderValue=0.0
    )
    img_white = cv2.warpAffine(
        img_white, IM, (whole_img.shape[1], whole_img.shape[0]), borderValue=0.0
    )
    img_white[img_white > 20] = 255
    img_mask = img_white
    mask_h_inds, mask_w_inds = np.where(img_mask == 255)
    mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
    mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
    mask_size = int(np.sqrt(mask_h * mask_w))

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
