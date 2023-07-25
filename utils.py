import os
import cv2
import time
import glob
import shutil
import platform
import datetime
import subprocess
import numpy as np
from threading import Thread
from moviepy.editor import VideoFileClip, ImageSequenceClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


logo_image = cv2.imread("./assets/images/logo.png", cv2.IMREAD_UNCHANGED)


quality_types = ["poor", "low", "medium", "high", "best"]


bitrate_quality_by_resolution = {
    240: {"poor": "300k", "low": "500k", "medium": "800k", "high": "1000k", "best": "1200k"},
    360: {"poor": "500k","low": "800k","medium": "1200k","high": "1500k","best": "2000k"},
    480: {"poor": "800k","low": "1200k","medium": "2000k","high": "2500k","best": "3000k"},
    720: {"poor": "1500k","low": "2500k","medium": "4000k","high": "5000k","best": "6000k"},
    1080: {"poor": "2500k","low": "4000k","medium": "6000k","high": "7000k","best": "8000k"},
    1440: {"poor": "4000k","low": "6000k","medium": "8000k","high": "10000k","best": "12000k"},
    2160: {"poor": "8000k","low": "10000k","medium": "12000k","high": "15000k","best": "20000k"}
}


crf_quality_by_resolution = {
    240: {"poor": 45, "low": 35, "medium": 28, "high": 23, "best": 20},
    360: {"poor": 35, "low": 28, "medium": 23, "high": 20, "best": 18},
    480: {"poor": 28, "low": 23, "medium": 20, "high": 18, "best": 16},
    720: {"poor": 23, "low": 20, "medium": 18, "high": 16, "best": 14},
    1080: {"poor": 20, "low": 18, "medium": 16, "high": 14, "best": 12},
    1440: {"poor": 18, "low": 16, "medium": 14, "high": 12, "best": 10},
    2160: {"poor": 16, "low": 14, "medium": 12, "high": 10, "best": 8}
}


def get_bitrate_for_resolution(resolution, quality):
    available_resolutions = list(bitrate_quality_by_resolution.keys())
    closest_resolution = min(available_resolutions, key=lambda x: abs(x - resolution))
    return bitrate_quality_by_resolution[closest_resolution][quality]


def get_crf_for_resolution(resolution, quality):
    available_resolutions = list(crf_quality_by_resolution.keys())
    closest_resolution = min(available_resolutions, key=lambda x: abs(x - resolution))
    return crf_quality_by_resolution[closest_resolution][quality]


def get_video_bitrate(video_file):
    ffprobe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=bit_rate', '-of', 'default=noprint_wrappers=1:nokey=1', video_file]
    result = subprocess.run(ffprobe_cmd, stdout=subprocess.PIPE)
    kbps = max(int(result.stdout) // 1000, 10)
    return str(kbps) + 'k'


def trim_video(video_path, output_path, start_frame, stop_frame):
    video_name, _ = os.path.splitext(os.path.basename(video_path))
    trimmed_video_filename = video_name + "_trimmed" + ".mp4"
    temp_path = os.path.join(output_path, "trim")
    os.makedirs(temp_path, exist_ok=True)
    trimmed_video_file_path = os.path.join(temp_path, trimmed_video_filename)

    video = VideoFileClip(video_path, fps_source="fps")
    fps = video.fps
    start_time = start_frame / fps
    duration = (stop_frame - start_frame) / fps

    bitrate = get_bitrate_for_resolution(min(*video.size), "high")

    trimmed_video = video.subclip(start_time, start_time + duration)
    trimmed_video.write_videofile(
        trimmed_video_file_path, codec="libx264", audio_codec="aac", bitrate=bitrate,
    )
    trimmed_video.close()
    video.close()

    return trimmed_video_file_path


def open_directory(path=None):
    if path is None:
        return
    try:
        os.startfile(path)
    except:
        subprocess.Popen(["xdg-open", path])


class StreamerThread(object):
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.FPS = 1 / 30
        self.FPS_MS = int(self.FPS * 1000)
        self.thread = None
        self.stopped = False
        self.frame = None

    def start(self):
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.stopped = True
        self.thread.join()
        print("stopped")

    def update(self):
        while not self.stopped:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(self.FPS)


class ProcessBar:
    def __init__(self, bar_length, total, before="⬛", after="🟨"):
        self.bar_length = bar_length
        self.total = total
        self.before = before
        self.after = after
        self.bar = [self.before] * bar_length
        self.start_time = time.time()

    def get(self, index):
        total = self.total
        elapsed_time = time.time() - self.start_time
        average_time_per_iteration = elapsed_time / (index + 1)
        remaining_iterations = total - (index + 1)
        estimated_remaining_time = remaining_iterations * average_time_per_iteration

        self.bar[int(index / total * self.bar_length)] = self.after
        info_text = f"({index+1}/{total}) {''.join(self.bar)} "
        info_text += f"(ETR: {int(estimated_remaining_time // 60)} min {int(estimated_remaining_time % 60)} sec)"
        return info_text


def add_logo_to_image(img, logo=logo_image):
    logo_size = int(img.shape[1] * 0.1)
    logo = cv2.resize(logo, (logo_size, logo_size))
    if logo.shape[2] == 4:
        alpha = logo[:, :, 3]
    else:
        alpha = np.ones_like(logo[:, :, 0]) * 255
    padding = int(logo_size * 0.1)
    roi = img.shape[0] - logo_size - padding, img.shape[1] - logo_size - padding
    for c in range(0, 3):
        img[roi[0] : roi[0] + logo_size, roi[1] : roi[1] + logo_size, c] = (
            alpha / 255.0
        ) * logo[:, :, c] + (1 - alpha / 255.0) * img[
            roi[0] : roi[0] + logo_size, roi[1] : roi[1] + logo_size, c
        ]
    return img


def split_list_by_lengths(data, length_list):
    split_data = []
    start_idx = 0
    for length in length_list:
        end_idx = start_idx + length
        sublist = data[start_idx:end_idx]
        split_data.append(sublist)
        start_idx = end_idx
    return split_data


def merge_img_sequence_from_ref(ref_video_path, image_sequence, output_file_name):
    video_clip = VideoFileClip(ref_video_path, fps_source="fps")
    fps = video_clip.fps
    duration = video_clip.duration
    total_frames = video_clip.reader.nframes
    audio_clip = video_clip.audio if video_clip.audio is not None else None
    edited_video_clip = ImageSequenceClip(image_sequence, fps=fps)

    if audio_clip is not None:
        edited_video_clip = edited_video_clip.set_audio(audio_clip)

    bitrate = get_bitrate_for_resolution(min(*edited_video_clip.size), "high")

    edited_video_clip.set_duration(duration).write_videofile(
        output_file_name, codec="libx264", bitrate=bitrate,
    )
    edited_video_clip.close()
    video_clip.close()


def scale_bbox_from_center(bbox, scale_width, scale_height, image_width, image_height):
    # Extract the coordinates of the bbox
    x1, y1, x2, y2 = bbox

    # Calculate the center point of the bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Calculate the new width and height of the bbox based on the scaling factors
    width = x2 - x1
    height = y2 - y1
    new_width = width * scale_width
    new_height = height * scale_height

    # Calculate the new coordinates of the bbox, considering the image boundaries
    new_x1 = center_x - new_width / 2
    new_y1 = center_y - new_height / 2
    new_x2 = center_x + new_width / 2
    new_y2 = center_y + new_height / 2

    # Adjust the coordinates to ensure the bbox remains within the image boundaries
    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(image_width - 1, new_x2)
    new_y2 = min(image_height - 1, new_y2)

    # Return the scaled bbox coordinates
    scaled_bbox = [new_x1, new_y1, new_x2, new_y2]
    return scaled_bbox


def laplacian_blending(A, B, m, num_levels=7):
    assert A.shape == B.shape
    assert B.shape == m.shape
    height = m.shape[0]
    width = m.shape[1]
    size_list = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
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
    ls_ = ls_[:height, :width, :]
    #ls_ = (ls_ - np.min(ls_)) * (255.0 / (np.max(ls_) - np.min(ls_)))
    return ls_.clip(0, 255)


def mask_crop(mask, crop):
    top, bottom, left, right = crop
    shape = mask.shape
    top = int(top)
    bottom = int(bottom)
    if top + bottom < shape[1]:
        if top > 0: mask[:top, :] = 0
        if bottom > 0: mask[-bottom:, :] = 0

    left = int(left)
    right = int(right)
    if left + right < shape[0]:
        if left > 0: mask[:, :left] = 0
        if right > 0: mask[:, -right:] = 0

    return mask

def create_image_grid(images, size=128):
    num_images = len(images)
    num_cols = int(np.ceil(np.sqrt(num_images)))
    num_rows = int(np.ceil(num_images / num_cols))
    grid = np.zeros((num_rows * size, num_cols * size, 3), dtype=np.uint8)

    for i, image in enumerate(images):
        row_idx = (i // num_cols) * size
        col_idx = (i % num_cols) * size
        image = cv2.resize(image.copy(), (size,size))
        if image.dtype != np.uint8:
            image = (image.astype('float32') * 255).astype('uint8')
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        grid[row_idx:row_idx + size, col_idx:col_idx + size] = image

    return grid
