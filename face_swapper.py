import time
import torch
import onnx
import cv2
import onnxruntime
import numpy as np
from tqdm import tqdm
from onnx import numpy_helper
from utils import add_logo_to_image
from insightface.utils import face_align


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
        print('inswapper-shape:', self.input_shape)
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
        for img, target_face, source_face in zip(imgs, target_faces, source_faces):
            if isinstance(img, str):
                img = cv2.imread(img)
            aimg, M = face_align.norm_crop2(img, target_face.kps, self.input_size[0])
            blob = cv2.dnn.blobFromImage(aimg, 1.0 / self.input_std, self.input_size,
                                         (self.input_mean, self.input_mean, self.input_mean), swapRB=True)
            latent = source_face.normed_embedding.reshape((1, -1))
            latent = np.dot(latent, self.emap)
            latent /= np.linalg.norm(latent)
            pred = self.session.run(self.output_names, {self.input_names[0]: blob, self.input_names[1]: latent})[0]
            pred = pred.transpose((0,2,3,1))[0]
            pred = np.clip(255 * pred, 0, 255).astype(np.uint8)[:,:,::-1]
            batch_preds.append((pred,aimg,M))
        return batch_preds

    def batch_forward(self, img_list, target_f_list, source_f_list):
        num_samples = len(img_list)
        num_batches = (num_samples + self.batch_size - 1) // self.batch_size

        preds = []
        for i in tqdm(range(num_batches), desc="Swapping face by batch"):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, num_samples)

            batch_img = img_list[start_idx:end_idx]
            batch_target_f = target_f_list[start_idx:end_idx]
            batch_source_f = source_f_list[start_idx:end_idx]

            batch_pred = self.get(batch_img, batch_target_f, batch_source_f)
            preds.extend(batch_pred)

        return preds


def paste_to_whole(bgr_fake, aimg, M, whole_img):
    IM = cv2.invertAffineTransform(M)

    img_white = np.full((aimg.shape[0], aimg.shape[1]), 255, dtype=np.float32)
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

    img_mask = np.reshape(img_mask, [img_mask.shape[0], img_mask.shape[1], 1])
    fake_merged = img_mask * bgr_fake + (1 - img_mask) * whole_img.astype(np.float32)
    fake_merged = add_logo_to_image(fake_merged.astype("uint8"))
    return fake_merged