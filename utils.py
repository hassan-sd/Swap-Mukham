import os
import cv2
import time
import glob
import shutil
import platform
import datetime
import subprocess
from threading import Thread
from moviepy.editor import VideoFileClip, ImageSequenceClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def trim_video(video_path, output_path, start_frame, stop_frame):
    video_name, _ = os.path.splitext(os.path.basename(video_path))
    trimmed_video_filename = video_name + "_trimmed" + ".mp4"
    temp_path = os.path.join(output_path, "trim")
    os.makedirs(temp_path, exist_ok=True)
    trimmed_video_file_path = os.path.join(temp_path, trimmed_video_filename)

    video = VideoFileClip(video_path)
    fps = video.fps
    start_time = start_frame / fps
    duration = (stop_frame - start_frame) / fps

    trimmed_video = video.subclip(start_time, start_time + duration)
    trimmed_video.write_videofile(
        trimmed_video_file_path, codec="libx264", audio_codec="aac"
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


logo_image = cv2.imread("./assets/images/logo.png", cv2.IMREAD_UNCHANGED)


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
