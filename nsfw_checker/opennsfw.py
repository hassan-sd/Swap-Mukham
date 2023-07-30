import cv2
import torch
import onnx
import onnxruntime
import numpy as np
from tqdm import tqdm

# https://github.com/yahoo/open_nsfw

class NSFWChecker:
    def __init__(self, model_path=None, providers=["CPUExecutionProvider"]):
        model = onnx.load(model_path)
        self.input_name = model.graph.input[0].name
        session_options = onnxruntime.SessionOptions()
        self.session = onnxruntime.InferenceSession(model_path, sess_options=session_options, providers=providers)

    def is_nsfw(self, img_paths, threshold = 0.85):
        skip_step = 1
        total_len = len(img_paths)
        if total_len < 100: skip_step = 1
        if total_len > 100 and total_len < 500: skip_step = 10
        if total_len > 500 and total_len < 1000: skip_step = 20
        if total_len > 1000 and total_len < 10000: skip_step = 50
        if total_len > 10000: skip_step = 100

        for idx in tqdm(range(0, total_len, skip_step), total=int(total_len // skip_step), desc="Checking for NSFW contents"):
            img = cv2.imread(img_paths[idx])
            img = cv2.resize(img, (224,224)).astype('float32')
            img -= np.array([104, 117, 123], dtype=np.float32)
            img = np.expand_dims(img, axis=0)

            score = self.session.run(None, {self.input_name:img})[0][0][1]

            if score > threshold:
                print(f"Detected nsfw score:{score}")
                return True
        return False
