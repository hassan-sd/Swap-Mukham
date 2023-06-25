import os
import cv2
import glob
import time
import torch
import shutil
import platform
import datetime
import subprocess
import insightface
import onnxruntime
import numpy as np
import gradio as gr
from threading import Thread
from moviepy.editor import VideoFileClip, ImageSequenceClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

WORKSPACE = None
OUTPUT_FILE = None
CURRENT_FRAME = None
STREAMER = None
DETECT_CONDITION = "left most"
NUM_OF_SRC_SPECIFIC = 10

### provider
provider = ["CPUExecutionProvider"]
available_providers = onnxruntime.get_available_providers()
if "CUDAExecutionProvider" in available_providers:
    provider = ["CUDAExecutionProvider", "CPUExecutionProvider"]

### load swapping model
model_path = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "inswapper_128.onnx"
)
MODEL = insightface.model_zoo.get_model(model_path, providers=provider)

### load face analyser
FACE_ANALYSER = insightface.app.FaceAnalysis(name="buffalo_l", providers=provider)
FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)

detect_conditions = [
    "left most",
    "right most",
    "top most",
    "bottom most",
    "most width",
    "most height",
]


def analyse_face(image, return_single_face=True):
    faces = FACE_ANALYSER.get(image)
    if not return_single_face:
        return faces

    total_faces = len(faces)
    if total_faces == 1:
        return faces[0]

    global DETECT_CONDITION
    condition = DETECT_CONDITION

    print(f"{total_faces} face detected. Using {condition} face.")
    if condition == "left most":
        return sorted(faces, key=lambda face: face["bbox"][0])[0]
    elif condition == "right most":
        return sorted(faces, key=lambda face: face["bbox"][0])[-1]
    elif condition == "top most":
        return sorted(faces, key=lambda face: face["bbox"][1])[0]
    elif condition == "bottom most":
        return sorted(faces, key=lambda face: face["bbox"][1])[-1]
    elif condition == "most width":
        return sorted(faces, key=lambda face: face["bbox"][2])[-1]
    elif condition == "most height":
        return sorted(faces, key=lambda face: face["bbox"][3])[-1]


swap_options_list = [
    "All face",
    "Age less than",
    "Age greater than",
    "All Male",
    "All Female",
    "Specific Face",
]


def swap_face(source_face, target_faces, target, condition, age):
    swapped = target.copy()

    for target_face in target_faces:
        if condition == "All face":
            swapped = MODEL.get(swapped, target_face, source_face, paste_back=True)
        elif condition == "Age less than" and target_face["age"] < age:
            swapped = MODEL.get(swapped, target_face, source_face, paste_back=True)
        elif condition == "Age greater than" and target_face["age"] > age:
            swapped = MODEL.get(swapped, target_face, source_face, paste_back=True)
        elif condition == "All Male" and target_face["gender"] == 1:
            swapped = MODEL.get(swapped, target_face, source_face, paste_back=True)
        elif condition == "All Female" and target_face["gender"] == 0:
            swapped = MODEL.get(swapped, target_face, source_face, paste_back=True)

    return swapped


def swap_specific(source_specifics, target_faces, target, threshold=0.6):
    swapped = target.copy()

    for source_face, specific_face in source_specifics:
        specific_embed = specific_face["embedding"]
        specific_embed /= np.linalg.norm(specific_embed)

        for target_face in target_faces:
            target_embed = target_face["embedding"]
            target_embed /= np.linalg.norm(target_embed)
            cosine_distance = 1 - np.dot(specific_embed, target_embed)
            if cosine_distance > threshold:
                continue
            swapped = MODEL.get(swapped, target_face, source_face, paste_back=True)

    return swapped


def trim_video(video_path, output_path, start_frame, stop_frame):
    video_name, video_extension = os.path.splitext(os.path.basename(video_path))
    trimmed_video_filename = video_name + "_trimmed" + video_extension
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
        info_text = f"### \n({index+1}/{total}) {''.join(self.bar)} "
        info_text += f"(ETR: {int(estimated_remaining_time // 60)} min {int(estimated_remaining_time % 60)} sec)"
        return info_text


def process(
    input_type,
    image_path,
    video_path,
    directory_path,
    source_path,
    output_path,
    output_name,
    keep_output_sequence,
    condition,
    age,
    distance,
    *specifics,
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

    start_time = time.time()
    specifics = list(specifics)
    half = len(specifics) // 2
    sources = specifics[:half]
    specifics = specifics[half:]

    yield "### \n Analysing Face...", *ui_before()

    analysed_source_specific = []
    if condition == "Specific Face":
        for source, specific in zip(sources, specifics):
            if source is None or specific is None:
                continue
            analysed_source = analyse_face(source, return_single_face=True)
            analysed_specific = analyse_face(specific, return_single_face=True)
            analysed_source_specific.append([analysed_source, analysed_specific])
    else:
        source = cv2.imread(source_path)
        analysed_source = analyse_face(source, return_single_face=True)

    if input_type == "Image":
        target = cv2.imread(image_path)
        analysed_target = analyse_face(target, return_single_face=False)
        if condition == "Specific Face":
            swapped = swap_specific(
                analysed_source_specific, analysed_target, target, threshold=distance
            )
        else:
            swapped = swap_face(
                analysed_source, analysed_target, target, condition, age
            )

        filename = os.path.join(output_path, output_name + ".png")
        cv2.imwrite(filename, swapped)
        OUTPUT_FILE = filename
        WORKSPACE = output_path
        PREVIEW = swapped[:, :, ::-1]

        tot_exec_time = time.time() - start_time
        _min, _sec = divmod(tot_exec_time, 60)

        yield f"Completed in {int(_min)} min {int(_sec)} sec.", *ui_after()

    elif input_type == "Video":
        temp_path = os.path.join(output_path, output_name, "sequence")
        os.makedirs(temp_path, exist_ok=True)

        video_clip = VideoFileClip(video_path)
        duration = video_clip.duration
        fps = video_clip.fps
        audio_clip = video_clip.audio if video_clip.audio is not None else None

        image_sequence = []
        process_bar = ProcessBar(30, video_clip.reader.nframes)

        for i, frame in enumerate(video_clip.iter_frames()):
            swapped = frame
            analysed_target = analyse_face(frame, return_single_face=False)

            if condition == "Specific Face":
                swapped = swap_specific(
                    analysed_source_specific, analysed_target, frame, threshold=distance
                )
            else:
                swapped = swap_face(
                    analysed_source, analysed_target, frame, condition, age
                )

            image_path = os.path.join(temp_path, f"frame_{i}.png")
            cv2.imwrite(image_path, swapped[:, :, ::-1])
            image_sequence.append(image_path)

            info_text = process_bar.get(i)
            PREVIEW = swapped
            yield info_text, *ui_before()

        yield "### \nMerging...", *ui_before()
        edited_video_clip = ImageSequenceClip(image_sequence, fps=fps)

        if audio_clip is not None:
            edited_video_clip = edited_video_clip.set_audio(audio_clip)

        output_video_path = os.path.join(output_path, output_name + ".mp4")
        edited_video_clip.set_duration(duration).write_videofile(
            output_video_path, codec="libx264"
        )
        edited_video_clip.close()
        video_clip.close()

        if os.path.exists(temp_path) and not keep_output_sequence:
            yield "### \nRemoving temporary files...", *ui_before()
            shutil.rmtree(temp_path)

        WORKSPACE = output_path
        OUTPUT_FILE = output_video_path

        tot_exec_time = time.time() - start_time
        _min, _sec = divmod(tot_exec_time, 60)

        yield f"✔️ Completed in {int(_min)} min {int(_sec)} sec.", *ui_after_vid()

    elif input_type == "Directory":
        source = cv2.imread(source_path)
        source = analyse_face(source, return_single_face=True)
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
            analysed_target = analyse_face(target, return_single_face=False)

            if condition == "Specific Face":
                swapped = swap_specific(
                    analysed_source_specific,
                    analysed_target,
                    target,
                    threshold=distance,
                )
            else:
                swapped = swap_face(
                    analysed_source, analysed_target, target, condition, age
                )

            filename = os.path.join(temp_path, os.path.basename(file_path))
            cv2.imwrite(filename, swapped)
            info_text = f"### \n Processing file {i+1} of {files_length}"
            PREVIEW = swapped[:, :, ::-1]
            yield info_text, *ui_before()

        WORKSPACE = temp_path
        OUTPUT_FILE = filename

        tot_exec_time = time.time() - start_time
        _min, _sec = divmod(tot_exec_time, 60)

        yield f"✔️ Completed in {int(_min)} min {int(_sec)} sec.", *ui_after()

    elif input_type == "Stream":
        yield "Starting...", *ui_before()
        source = cv2.imread(source_path)
        source = analyse_face(source, return_single_face=True)

        global STREAMER
        STREAMER = StreamerThread(src=directory_path)
        STREAMER.start()
        while True:
            try:
                target = STREAMER.frame
                analysed_target = analyse_face(target, return_single_face=False)
                if condition == "Specific Face":
                    swapped = swap_specific(
                        analysed_source_specific,
                        analysed_target,
                        target,
                        threshold=distance,
                    )
                else:
                    swapped = swap_face(
                        analysed_source,
                        analysed_target,
                        target,
                        condition,
                        age,
                    )
                PREVIEW = swapped[:, :, ::-1]
                yield f"Streaming...", *ui_before()
            except AttributeError:
                yield "Streaming...", *ui_before()
        STREAMER.stop()


### Gradio change functions
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


def swap_option_changed(value):
    if value == swap_options_list[1] or value == swap_options_list[2]:
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
        )
    elif value == swap_options_list[5]:
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
        )
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


def video_changed(video_path):
    sliders_update = gr.Slider.update
    button_update = gr.Button.update
    number_update = gr.Number.update

    if video_path is None:
        return (
            sliders_update(minimum=0, maximum=0, value=0),
            sliders_update(minimum=1, maximum=1, value=1),
            number_update(value=1),
        )
    try:
        clip = VideoFileClip(video_path)
        fps = clip.fps
        total_frames = clip.reader.nframes
        clip.close()
        return (
            sliders_update(minimum=0, maximum=total_frames, value=0, interactive=True),
            sliders_update(
                minimum=0, maximum=total_frames, value=total_frames, interactive=True
            ),
            number_update(value=fps),
        )
    except:
        return (
            sliders_update(value=0),
            sliders_update(value=0),
            number_update(value=1),
        )


def analyse_settings_changed(detect_condition, detection_size, detection_threshold):
    yield "### \n ⌛ Applying new values..."
    global FACE_ANALYSER
    global DETECT_CONDITION
    DETECT_CONDITION = detect_condition
    FACE_ANALYSER = insightface.app.FaceAnalysis(name="buffalo_l", providers=provider)
    FACE_ANALYSER.prepare(
        ctx_id=0,
        det_size=(int(detection_size), int(detection_size)),
        det_thresh=float(detection_threshold),
    )
    yield f"### \n ✔️ Applied detect condition:{detect_condition}, detection size: {detection_size}, detection threshold: {detection_threshold}"


def stop_running():
    global STREAMER
    if hasattr(STREAMER, "stop"):
        STREAMER.stop()
        STREAMER = None
    return "Cancelled"


def slider_changed(show_frame, video_path, frame_index):
    if not show_frame:
        return None, None
    if video_path is None:
        return None, None
    clip = VideoFileClip(video_path)
    frame = clip.get_frame(frame_index / clip.fps)
    frame_array = np.array(frame)
    clip.close()
    return gr.Image.update(value=frame_array, visible=True), gr.Video.update(
        visible=False
    )


def trim_and_reload(video_path, output_path, output_name, start_frame, stop_frame):
    yield video_path, f"### \n ⌛ Trimming video frame {start_frame} to {stop_frame}..."
    try:
        output_path = os.path.join(output_path, output_name)
        trimmed_video = trim_video(video_path, output_path, start_frame, stop_frame)
        yield trimmed_video, "### \n ✔️ Video trimmed and reloaded."
    except Exception as e:
        print(e)
        yield video_path, "### \n ❌ Video trimming failed. See console for more info."


css = """
footer{display:none !important}
"""

### Gradio interface
with gr.Blocks(css=css) as interface:
    gr.Markdown("# 🗿 Swap Mukham")
    gr.Markdown("### Face swap app based on insightface inswapper.")
    with gr.Row():
        with gr.Row():
            with gr.Column(scale=0.4):
                with gr.Tab("📄 Swap Condition"):
                    swap_option = gr.Radio(
                        swap_options_list,
                        show_label=False,
                        value=swap_options_list[0],
                        interactive=True,
                    )
                    age = gr.Number(
                        value=25, label="Value", interactive=True, visible=False
                    )

                with gr.Tab("🎚️ Detection Settings"):
                    detect_condition_dropdown = gr.Dropdown(
                        detect_conditions,
                        label="Condition",
                        value=DETECT_CONDITION,
                        interactive=True,
                        info="This condition is only used when multiple faces are detected on source or specific image.",
                    )
                    detection_size = gr.Number(
                        label="Detection Size", value=640, interactive=True
                    )
                    detection_threshold = gr.Number(
                        label="Detection Threshold", value=0.5, interactive=True
                    )
                    apply_detection_settings = gr.Button("Apply settings")

                with gr.Tab("📤 Output Settings"):
                    output_directory = gr.Text(
                        label="Output Directory", value=os.getcwd(), interactive=True
                    )
                    output_name = gr.Text(
                        label="Output Name", value="Result", interactive=True
                    )
                    keep_output_sequence = gr.Checkbox(
                        label="Keep output sequence", value=False, interactive=True
                    )

                source_image_input = gr.Image(
                    label="Source face", type="filepath", interactive=True
                )

                with gr.Box(visible=False) as specific_face:
                    for i in range(NUM_OF_SRC_SPECIFIC):
                        idx = i + 1
                        code = "\n"
                        code += f"with gr.Tab(label='({idx})'):"
                        code += "\n\twith gr.Row():"
                        code += f"\n\t\tsrc{idx} = gr.Image(interactive=True, type='numpy', label='Source Face {idx}')"
                        code += f"\n\t\ttrg{idx} = gr.Image(interactive=True, type='numpy', label='Specific Face {idx}')"
                        exec(code)

                    distance_slider = gr.Slider(
                        minimum=0,
                        maximum=2,
                        value=0.6,
                        interactive=True,
                        label="Distance",
                        info="Lower distance is more similar and higher distance is less similar to the target face.",
                    )

                with gr.Group():
                    input_type = gr.Radio(
                        ["Image", "Video", "Directory", "Stream"],
                        label="Target Type",
                        value="Video",
                    )

                    with gr.Box(visible=False) as input_image_group:
                        image_input = gr.Image(
                            label="Target Image", interactive=True, type="filepath"
                        )

                    with gr.Box(visible=True) as input_video_group:
                        video_input = gr.Text(
                            label="Target Video Path", interactive=True
                        )
                        with gr.Accordion("✂️ Trim video", open=False):
                            with gr.Column():
                                with gr.Row():
                                    set_slider_range_btn = gr.Button(
                                        "Set frame range", interactive=True
                                    )
                                    show_trim_preview_btn = gr.Checkbox(
                                        label="Show frame when slider change",
                                        value=True,
                                        interactive=True,
                                    )

                                video_fps = gr.Number(
                                    value=30,
                                    interactive=False,
                                    label="Fps",
                                    visible=False,
                                )
                                start_frame = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    value=0,
                                    step=1,
                                    interactive=True,
                                    label="Start Frame",
                                    info="",
                                )
                                end_frame = gr.Slider(
                                    minimum=0,
                                    maximum=1,
                                    value=1,
                                    step=1,
                                    interactive=True,
                                    label="End Frame",
                                    info="",
                                )
                            trim_and_reload_btn = gr.Button(
                                "Trim and Reload", interactive=True
                            )

                    with gr.Box(visible=False) as input_directory_group:
                        direc_input = gr.Text(label="Path", interactive=True)

            with gr.Column(scale=0.6):
                info = gr.Markdown(value="...")

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

                with gr.Column():
                    gr.Markdown(
                        '[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/harisreedhar)'
                    )
                    gr.Markdown(
                        "### [Source code](https://github.com/harisreedhar/Swap-Mukham) . [Disclaimer](https://github.com/harisreedhar/Swap-Mukham#disclaimer) . [Gradio](https://gradio.app/)"
                    )

    set_slider_range_event = set_slider_range_btn.click(
        video_changed,
        inputs=[video_input],
        outputs=[start_frame, end_frame, video_fps],
    )

    trim_and_reload_event = trim_and_reload_btn.click(
        fn=trim_and_reload,
        inputs=[video_input, output_directory, output_name, start_frame, end_frame],
        outputs=[video_input, info],
    )

    start_frame_event = start_frame.release(
        fn=slider_changed,
        inputs=[show_trim_preview_btn, video_input, start_frame],
        outputs=[preview_image, preview_video],
        show_progress=False,
    )

    end_frame_event = end_frame.release(
        fn=slider_changed,
        inputs=[show_trim_preview_btn, video_input, end_frame],
        outputs=[preview_image, preview_video],
        show_progress=False,
    )

    input_type.change(
        update_radio,
        inputs=[input_type],
        outputs=[input_image_group, input_video_group, input_directory_group],
    )
    swap_option.change(
        swap_option_changed,
        inputs=[swap_option],
        outputs=[age, specific_face, source_image_input],
    )

    apply_detection_settings.click(
        analyse_settings_changed,
        inputs=[detect_condition_dropdown, detection_size, detection_threshold],
        outputs=[info],
    )

    src_specific_inputs = []
    gen_variable_txt = ",".join(
        [f"src{i+1}" for i in range(NUM_OF_SRC_SPECIFIC)]
        + [f"trg{i+1}" for i in range(NUM_OF_SRC_SPECIFIC)]
    )
    exec(f"src_specific_inputs = ({gen_variable_txt})")
    swap_inputs = [
        input_type,
        image_input,
        video_input,
        direc_input,
        source_image_input,
        output_directory,
        output_name,
        keep_output_sequence,
        swap_option,
        age,
        distance_slider,
        *src_specific_inputs,
    ]

    swap_outputs = [
        info,
        preview_image,
        output_directory_button,
        output_video_button,
        preview_video,
    ]

    swap_event = swap_button.click(
        fn=process, inputs=swap_inputs, outputs=swap_outputs, show_progress=False
    )

    cancel_button.click(
        fn=stop_running,
        inputs=None,
        outputs=[info],
        cancels=[
            swap_event,
            trim_and_reload_event,
            set_slider_range_event,
            start_frame_event,
            end_frame_event,
        ],
        show_progress=False,
    )
    output_directory_button.click(
        lambda: open_directory(path=WORKSPACE), inputs=None, outputs=None
    )
    output_video_button.click(
        lambda: open_directory(path=OUTPUT_FILE), inputs=None, outputs=None
    )

if __name__ == "__main__":
    interface.queue(concurrency_count=2, max_size=20).launch()
