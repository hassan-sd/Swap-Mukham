import cv2
import torch
import onnx
import onnxruntime
import numpy as np

import time

# codeformer converted to onnx
# using https://github.com/redthing1/CodeFormer


class CodeFormerEnhancer:
    def __init__(self, model_path="codeformer.onnx", device='cpu'):
        model = onnx.load(model_path)
        session_options = onnxruntime.SessionOptions()
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = ["CPUExecutionProvider"]
        if device == 'cuda':
            providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "DEFAULT"}),"CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)

    def enhance(self, img, w=0.9):
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32)[:,:,::-1] / 255.0
        img = img.transpose((2, 0, 1))
        nrm_mean = np.array([0.5, 0.5, 0.5]).reshape((-1, 1, 1))
        nrm_std = np.array([0.5, 0.5, 0.5]).reshape((-1, 1, 1))
        img = (img - nrm_mean) / nrm_std

        img = np.expand_dims(img, axis=0)

        out = self.session.run(None, {'x':img.astype(np.float32), 'w':np.array([w], dtype=np.double)})[0]
        out = (out[0].transpose(1,2,0).clip(-1,1) + 1) * 0.5
        out = (out * 255)[:,:,::-1]

        return out.astype('uint8')
