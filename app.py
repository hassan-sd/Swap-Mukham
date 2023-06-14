import os
import cv2
import glob
import time
# import torch #gpu
import shutil
import platform
import tempfile
import threading
import subprocess
import insightface
import onnxruntime
import gradio as gr
import numpy as np
from threading import Thread

WORKSPACE = None
OUTPUT_FILE = None
CURRENT_FRAME = None
STREAMER = None

### provider
available_providers = onnxruntime.get_available_providers()
#provider = ["CUDAExecutionProvider", "CPUExecutionProvider"] #gpu
provider = ["CPUExecutionProvider"]

### load swapping model
model_path = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "inswapper_128.onnx"
)
MODEL = insightface.model_zoo.get_model(model_path, providers=provider)

### load face analyser
FACE_ANALYSER = insightface.app.FaceAnalysis(name="buffalo_l", providers=provider)
FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

### ffmpeg
ffmpeg = "ffmpeg"
custom_ffmpeg_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "ffmpeg")
if os.path.exists(custom_ffmpeg_path):
    ffmpeg = custom_ffmpeg_path


def change_analyse_settings(detection_size, detection_threshold):
    yield "### \n Applying new values..."
    global FACE_ANALYSER
    FACE_ANALYSER = insightface.app.FaceAnalysis(name="buffalo_l", providers=provider)
    FACE_ANALYSER.prepare(
        ctx_id=0,
        det_size=(detection_size, detection_size),
        det_thresh=detection_threshold,
    )
    yield f"### \n Applied detection size: {detection_size} & detection threshold: {detection_threshold}"


def analyse_face(image, single_output=True):
    source_faces = FACE_ANALYSER.get(image)
    print(f"Number of faces detected {len(source_faces)}")
    if not single_output:
        return source_faces
    if len(source_faces) > 1:
        raise ValueError("More than one face")
        return
    if len(source_faces) == 0:
        raise ValueError("No face detected")
        return
    return source_faces[0]


swap_options_list = [
    "All face",
    "Age less than",
    "Age greater than",
    "All Male",
    "All Female",
]


def swap_face(source, target, condition, condition_value, skip_source_analyse=False):
    source_face = source
    if not skip_source_analyse:
        source_face = analyse_face(source, single_output=True)

    target_faces = analyse_face(target, single_output=False)
    swapped = target.copy()

    for face in target_faces:
        if condition == swap_options_list[0]:
            swapped = MODEL.get(swapped, face, source_face, paste_back=True)
        elif condition == swap_options_list[1] and face["age"] < condition_value:
            swapped = MODEL.get(swapped, face, source_face, paste_back=True)
        elif condition == swap_options_list[2] and face["age"] > condition_value:
            swapped = MODEL.get(swapped, face, source_face, paste_back=True)
        elif condition == swap_options_list[3] and face["gender"] == 1:
            swapped = MODEL.get(swapped, face, source_face, paste_back=True)
        elif condition == swap_options_list[4] and face["gender"] == 0:
            swapped = MODEL.get(swapped, face, source_face, paste_back=True)
    return swapped


def trim_video(video_path, output_path, start_frame, stop_frame):
    video_name, video_extension = os.path.splitext(os.path.basename(video_path))
    trimmed_video_filename = video_name + "_trimmed" + video_extension
    trimmed_video_file_path = os.path.join(output_path, trimmed_video_filename)
    command = [
        ffmpeg,
        "-i",
        video_path,
        "-ss",
        start_frame,
        "-to",
        stop_frame,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-strict",
        "-2",
        trimmed_video_file_path,
        "-y",
    ]
    out = subprocess.call(
        " ".join(command),
        shell=platform.system() != "Windows",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if out == 0:
        return trimmed_video_file_path, True
    return None, False


def get_audio_from_video(video_path, output_directory):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    audio = os.path.join(output_directory, f"{video_name}_audio.wav")
    command = [ffmpeg, "-v", "error", "-i", video_path, "-map", "0:a", audio, "-y"]
    out = subprocess.call(
        " ".join(command),
        shell=platform.system() != "Windows",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if out == 0:
        return audio, True
    return None, False


def image_sequence_to_video(
    image_sequence_path, output_directory, audio=None, fps=30, filename="result.mp4"
):
    output = os.path.join(output_directory, filename)
    command = [
        ffmpeg,
        "-v",
        "error",
        "-framerate",
        str(fps),
        "-i",
        image_sequence_path,
        f"-i {audio}" if audio is not None else "",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-pix_fmt",
        "yuv420p",
        "-shortest",
        output,
        "-y",
    ]
    out = subprocess.call(
        " ".join(command),
        shell=platform.system() != "Windows",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if out == 0:
        return output, True
    return None, False


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


def process(
    input_type,
    image_path,
    video_path,
    directory_path,
    source_path,
    output_path,
    output_name,
    condition,
    condition_value,
    trim,
    trim_start,
    trim_end,
):
    global WORKSPACE
    global OUTPUT_FILE
    global PREVIEW
    WORKSPACE, OUTPUT_FILE, PREVIEW = None, None, None

    def ui_before():
        return (
            gr.update(visible=True, value=PREVIEW),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(visible=False),
        )

    def ui_after():
        return (
            gr.update(visible=True, value=PREVIEW),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(visible=False),
        )

    def ui_after_vid():
        return (
            gr.update(visible=False),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(value=OUTPUT_FILE, visible=True),
        )

    if input_type == "Image":
        yield "### \n Swapping...", *ui_before()
        source = cv2.imread(source_path)
        target = cv2.imread(image_path)
        swapped = swap_face(source, target, condition, condition_value)
        filename = os.path.join(output_path, output_name + ".png")
        cv2.imwrite(filename, swapped)
        OUTPUT_FILE = filename
        WORKSPACE = output_path
        PREVIEW = swapped[:, :, ::-1]
        yield "Done!", *ui_after()

    elif input_type == "Video":
        yield "### \n Starting...", *ui_before()

        trimmed_video = None
        if trim:
            yield "### \n Trimming video...", *ui_before()
            trimmed_video, success = trim_video(
                video_path, output_path, trim_start, trim_end
            )
            if not success:
                yield "### \n Trimming video failed", *ui_before()
                return
            video_path = trimmed_video

        yield "### \n Analysing face...", *ui_before()
        source = cv2.imread(source_path)
        source = analyse_face(source, single_output=True)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        temp_path = os.path.join(output_path, output_name)
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        os.mkdir(temp_path)
        img_format = "image-%03d.jpg"
        swapped_seq_path = os.path.join(temp_path, img_format)

        start_time = time.time()
        bar_length = 20
        bar = ["⬛"] * bar_length

        for frame_index in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            swapped = frame
            swapped = swap_face(
                source, frame, condition, condition_value, skip_source_analyse=True
            )
            cv2.imwrite(swapped_seq_path % frame_index, swapped)

            elapsed_time = time.time() - start_time
            average_time_per_iteration = elapsed_time / (frame_index + 1)
            remaining_iterations = total_frames - (frame_index + 1)
            estimated_remaining_time = remaining_iterations * average_time_per_iteration

            bar[int(frame_index / total_frames * bar_length)] = "🟨"
            info_text = f"### \n({frame_index+1}/{total_frames}) {''.join(bar)} "
            info_text += f"(ETR: {int(estimated_remaining_time // 60)} min {int(estimated_remaining_time % 60)} sec)"

            PREVIEW = swapped[:, :, ::-1]
            yield info_text, *ui_before()

        cap.release()

        yield "### \n Merging image sequence...", *ui_before()
        audio, success = get_audio_from_video(video_path, output_path)
        merged_output, success = image_sequence_to_video(
            swapped_seq_path, output_path, audio, fps=fps, filename=output_name + ".mp4"
        )

        if not success:
            yield "### \n Merging image sequence failed", *ui_before()
            return

        yield "### \n Removing temp files...", *ui_before()
        if audio is not None and os.path.exists(audio):
            os.remove(audio)
        if trim and os.path.exists(trimmed_video):
            os.remove(trimmed_video)
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)

        WORKSPACE = output_path
        OUTPUT_FILE = merged_output

        yield "Done!", *ui_after_vid()

    elif input_type == "Directory":
        yield "### \n Starting...", *ui_before()
        source = cv2.imread(source_path)
        source = analyse_face(source, single_output=True)
        extensions = ["jpg", "jpeg", "png", "bmp", "tiff", "ico", "webp"]
        temp_path = os.path.join(output_path, output_name)
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        os.mkdir(temp_path)
        swapped = None

        files = []
        for file_path in glob.glob(os.path.join(directory_path, "*")):
            if any(file_path.lower().endswith(ext) for ext in extensions):
                files.append(file_path)

        files_length = len(files)
        filename = None
        for i, file_path in enumerate(files):
            target = cv2.imread(file_path)
            swapped = swap_face(
                source, target, condition, condition_value, skip_source_analyse=True
            )
            filename = os.path.join(temp_path, os.path.basename(file_path))
            cv2.imwrite(filename, swapped)
            info_text = f"### \n Processing file {i+1} of {files_length}"
            PREVIEW = swapped[:, :, ::-1]
            yield info_text, *ui_before()

        WORKSPACE = temp_path
        OUTPUT_FILE = filename

        yield "Done!", *ui_after()

    elif input_type == "Stream":
        yield "Starting...", *ui_before()
        source = cv2.imread(source_path)
        source = analyse_face(source, single_output=True)

        global STREAMER
        STREAMER = StreamerThread(src=directory_path)
        STREAMER.start()
        while True:
            try:
                frame = STREAMER.frame
                swapped = swap_face(
                    source, frame, condition, condition_value, skip_source_analyse=True
                )
                PREVIEW = swapped[:, :, ::-1]
                yield f"Streaming...", *ui_before()
            except AttributeError:
                yield "Streaming...", *ui_before()
        STREAMER.stop()


### Gradio
def update_radio(value):
    if value == "Image":
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )
    elif value == "Video":
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )
    elif value == "Directory":
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
        )
    elif value == "Stream":
        return (
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
        )


def update_swap_option(value):
    if value == swap_options_list[1] or value == swap_options_list[2]:
        return gr.update(visible=True)
    return gr.update(visible=False)


def stop_running():
    if hasattr(STREAMER, "stop"):
        STREAMER.stop()
        del STREAMER
    return "Cancelled"


with gr.Blocks() as interface:
    gr.Markdown("# 🗿 Swap Mukham")
    gr.Markdown("A simple face swapper based on insightface inswapper")
    with gr.Row():
        with gr.Row():
            with gr.Column(scale=0.4):
                source_image_input = gr.Image(
                    label="Source face", type="filepath", interactive=True
                )

                with gr.Group():
                    input_type = gr.Radio(
                        ["Image", "Video", "Directory", "Stream"],
                        label="Target type",
                        value="Video",
                    )

                    with gr.Box(visible=False) as input_image_group:
                        image_input = gr.Image(
                            label="Target Image", interactive=True, type="filepath"
                        )

                    with gr.Box(visible=True) as input_video_group:
                        video_input = gr.Video(label="Target Video", interactive=True)
                        with gr.Accordion("✂️ Trim video", open=False):
                            enable_trim = gr.Checkbox(label="Enable", value=False)
                            with gr.Row():
                                trim_start = gr.Text(
                                    label="Trim Start",
                                    placeholder="HH:MM:SS",
                                    interactive=True,
                                )
                                trim_end = gr.Text(
                                    label="Trim End",
                                    placeholder="HH:MM:SS",
                                    interactive=True,
                                )

                    with gr.Box(visible=False) as input_directory_group:
                        direc_input = gr.Text(label="Path", interactive=True)

                info = gr.Markdown(show_label=False, visible=True)

            with gr.Column(scale=0.6):
                with gr.Accordion("🎚️ Detection Settings", open=False):
                    detection_size = gr.Number(
                        label="Detection Size", value=640, interactive=True
                    )
                    detection_threshold = gr.Number(
                        label="Detection Threshold", value=0.5, interactive=True
                    )
                    apply_detection_settings = gr.Button("Apply settings")

                with gr.Accordion("📄 Swap Options", open=False):
                    swap_option = gr.Radio(
                        swap_options_list,
                        label="Condition",
                        value=swap_options_list[0],
                        interactive=True,
                    )
                    condition_value = gr.Number(
                        value=25, label="Value", interactive=True, visible=False
                    )

                with gr.Accordion("📤 Output Settings", open=False):
                    output_directory = gr.Text(
                        label="Output Directory", value=os.getcwd(), interactive=True
                    )
                    output_name = gr.Text(
                        label="Output Name", value="Result", interactive=True
                    )

                with gr.Row():
                    swap_button = gr.Button("✨ Swap", variant="primary")
                    cancel_button = gr.Button("⛔ Cancel")

                preview_image = gr.Image(label="Output", interactive=False)
                preview_video = gr.Video(
                    label="Output", interactive=False, visible=False
                )

                with gr.Row():
                    output_directory_button = gr.Button("📂", interactive=False)
                    output_video_button = gr.Button("🎬", interactive=False)

    input_type.change(
        update_radio,
        inputs=[input_type],
        outputs=[input_image_group, input_video_group, input_directory_group],
    )
    swap_option.change(
        update_swap_option, inputs=[swap_option], outputs=[condition_value]
    )
    apply_detection_settings.click(
        change_analyse_settings,
        inputs=[detection_size, detection_threshold],
        outputs=[info],
    )

    swap_inputs = [
        input_type,
        image_input,
        video_input,
        direc_input,
        source_image_input,
        output_directory,
        output_name,
        swap_option,
        condition_value,
        enable_trim,
        trim_start,
        trim_end,
    ]
    swap_outputs = [
        info,
        preview_image,
        output_directory_button,
        output_video_button,
        preview_video,
    ]
    swap_event = swap_button.click(fn=process, inputs=swap_inputs, outputs=swap_outputs)

    cancel_button.click(
        fn=stop_running, inputs=None, outputs=[info], cancels=[swap_event]
    )
    output_directory_button.click(
        lambda: open_directory(path=WORKSPACE), inputs=None, outputs=None
    )
    output_video_button.click(
        lambda: open_directory(path=OUTPUT_FILE), inputs=None, outputs=None
    )

if __name__ == "__main__":
    interface.queue(concurrency_count=2, max_size=20).launch()
