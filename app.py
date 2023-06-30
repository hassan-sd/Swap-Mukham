import os
import cv2
import glob
import time
import torch
import shutil
import gfpgan
import argparse
import platform
import datetime
import subprocess
import insightface
import onnxruntime
import numpy as np
import gradio as gr
from moviepy.editor import VideoFileClip, ImageSequenceClip

from face_analyser import detect_conditions, analyse_face
from utils import trim_video, StreamerThread, ProcessBar, open_directory
from face_parsing import init_parser, swap_regions, mask_regions, mask_regions_to_list
from swapper import (
    swap_face,
    swap_face_with_condition,
    swap_specific,
    swap_options_list,
)

## ------------------------------ USER ARGS ------------------------------

parser = argparse.ArgumentParser(description="Swap-Mukham Face Swapper")
parser.add_argument("--out_dir", help="Default Output directory", default=os.getcwd())
parser.add_argument("--cuda", action="store_true", help="Enable cuda", default=False)
parser.add_argument(
    "--colab", action="store_true", help="Enable colab mode", default=False
)
user_args = parser.parse_args()

## ------------------------------ DEFAULTS ------------------------------

USE_COLAB = user_args.colab
USE_CUDA = user_args.cuda
DEF_OUTPUT_PATH = user_args.out_dir
WORKSPACE = None
OUTPUT_FILE = None
CURRENT_FRAME = None
STREAMER = None
DETECT_CONDITION = "left most"
DETECT_SIZE = 640
DETECT_THRESH = 0.6
NUM_OF_SRC_SPECIFIC = 10
MASK_INCLUDE = [
    "Skin",
    "R-Eyebrow",
    "L-Eyebrow",
    "L-Eye",
    "R-Eye",
    "Nose",
    "Mouth",
    "L-Lip",
    "U-Lip"
]
MASK_EXCLUDE = ["R-Ear", "L-Ear", "Hair", "Hat"]
MASK_BLUR = 25

FACE_SWAPPER = None
FACE_ANALYSER = None
FACE_ENHANCER = None
FACE_PARSER = None

## ------------------------------ SET EXECUTION PROVIDER ------------------------------
# Note: For AMD,MAC or non CUDA users, change settings here

PROVIDER = ["CPUExecutionProvider"]

if USE_CUDA:
    available_providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        print("\n********** Running on CUDA **********\n")
        PROVIDER = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        USE_CUDA = False
        print("\n********** CUDA unavailable running on CPU **********\n")
else:
    USE_CUDA = False
    print("\n********** Running on CPU **********\n")


## ------------------------------ LOAD MODELS ------------------------------

def load_face_analyser_model(name="buffalo_l"):
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name=name, providers=PROVIDER)
        FACE_ANALYSER.prepare(
            ctx_id=0, det_size=(DETECT_SIZE, DETECT_SIZE), det_thresh=DETECT_THRESH
        )


def load_face_swapper_model(name="./assets/pretrained_models/inswapper_128.onnx"):
    global FACE_SWAPPER
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), name)
    if FACE_SWAPPER is None:
        FACE_SWAPPER = insightface.model_zoo.get_model(path, providers=PROVIDER)


def load_face_enhancer_model(name="./assets/pretrained_models/GFPGANv1.4.pth"):
    global FACE_ENHANCER
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), name)
    if FACE_ENHANCER is None:
        FACE_ENHANCER = gfpgan.GFPGANer(model_path=path, upscale=1)


def load_face_parser_model(name="./assets/pretrained_models/79999_iter.pth"):
    global FACE_PARSER
    path = os.path.join(os.path.abspath(os.path.dirname(__file__)), name)
    if FACE_PARSER is None:
        FACE_PARSER = init_parser(name, use_cuda=USE_CUDA)


load_face_analyser_model()
load_face_swapper_model()

## ------------------------------ MAIN PROCESS ------------------------------


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
    face_enhance,
    enable_face_parser,
    mask_include,
    mask_exclude,
    mask_blur,
    *specifics,
):
    global WORKSPACE
    global OUTPUT_FILE
    global PREVIEW
    WORKSPACE, OUTPUT_FILE, PREVIEW = None, None, None

    ## ------------------------------ GUI UPDATE FUNC ------------------------------

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

    ## ------------------------------ LOAD PENDING MODELS ------------------------------
    start_time = time.time()
    specifics = list(specifics)
    half = len(specifics) // 2
    sources = specifics[:half]
    specifics = specifics[half:]

    yield "### \n ⌛ Loading face analyser model...", *ui_before()
    load_face_analyser_model()

    yield "### \n ⌛ Loading face swapper model...", *ui_before()
    load_face_swapper_model()

    if face_enhance:
        yield "### \n ⌛ Loading face enhancer model...", *ui_before()
        load_face_enhancer_model()

    if enable_face_parser:
        yield "### \n ⌛ Loading face parsing model...", *ui_before()
        load_face_parser_model()

    yield "### \n ⌛ Analysing Face...", *ui_before()

    mi = mask_regions_to_list(mask_include)
    me = mask_regions_to_list(mask_exclude)
    models = {
        "swap": FACE_SWAPPER,
        "enhance": FACE_ENHANCER,
        "enhance_sett": face_enhance,
        "face_parser": FACE_PARSER,
        "face_parser_sett": (enable_face_parser, mi, me, int(mask_blur)),
    }

    ## ------------------------------ ANALYSE SOURCE & SPECIFIC ------------------------------

    analysed_source_specific = []
    if condition == "Specific Face":
        for source, specific in zip(sources, specifics):
            if source is None or specific is None:
                continue
            analysed_source = analyse_face(
                source,
                FACE_ANALYSER,
                return_single_face=True,
                detect_condition=DETECT_CONDITION,
            )
            analysed_specific = analyse_face(
                specific,
                FACE_ANALYSER,
                return_single_face=True,
                detect_condition=DETECT_CONDITION,
            )
            analysed_source_specific.append([analysed_source, analysed_specific])
    else:
        source = cv2.imread(source_path)
        analysed_source = analyse_face(
            source,
            FACE_ANALYSER,
            return_single_face=True,
            detect_condition=DETECT_CONDITION,
        )

    ## ------------------------------ IMAGE ------------------------------

    if input_type == "Image":
        target = cv2.imread(image_path)
        analysed_target = analyse_face(target, FACE_ANALYSER, return_single_face=False)
        if condition == "Specific Face":
            swapped = swap_specific(
                analysed_source_specific,
                analysed_target,
                target,
                models,
                threshold=distance,
            )
        else:
            swapped = swap_face_with_condition(
                target, analysed_target, analysed_source, condition, age, models
            )

        filename = os.path.join(output_path, output_name + ".png")
        cv2.imwrite(filename, swapped)
        OUTPUT_FILE = filename
        WORKSPACE = output_path
        PREVIEW = swapped[:, :, ::-1]

        tot_exec_time = time.time() - start_time
        _min, _sec = divmod(tot_exec_time, 60)

        yield f"Completed in {int(_min)} min {int(_sec)} sec.", *ui_after()

    ## ------------------------------ VIDEO ------------------------------

    elif input_type == "Video":
        temp_path = os.path.join(output_path, output_name, "sequence")
        os.makedirs(temp_path, exist_ok=True)

        video_clip = VideoFileClip(video_path)
        duration = video_clip.duration
        fps = video_clip.fps
        total_frames = video_clip.reader.nframes

        analysed_targets = []
        process_bar = ProcessBar(30, total_frames)
        yield "### \n ⌛ Analysing...", *ui_before()
        for i, frame in enumerate(video_clip.iter_frames()):
            analysed_targets.append(
                analyse_face(frame, FACE_ANALYSER, return_single_face=False)
            )
            info_text = "Analysing Faces || "
            info_text += process_bar.get(i)
            print("\033[1A\033[K", end="", flush=True)
            print(info_text)
            if i % 10 == 0:
                yield "### \n" + info_text, *ui_before()
        video_clip.close()

        image_sequence = []
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio if video_clip.audio is not None else None
        process_bar = ProcessBar(30, total_frames)
        yield "### \n ⌛ Swapping...", *ui_before()
        for i, frame in enumerate(video_clip.iter_frames()):
            swapped = frame
            analysed_target = analysed_targets[i]

            if condition == "Specific Face":
                swapped = swap_specific(
                    frame,
                    analysed_target,
                    analysed_source_specific,
                    models,
                    threshold=distance,
                )
            else:
                swapped = swap_face_with_condition(
                    frame, analysed_target, analysed_source, condition, age, models
                )

            image_path = os.path.join(temp_path, f"frame_{i}.png")
            cv2.imwrite(image_path, swapped[:, :, ::-1])
            image_sequence.append(image_path)

            info_text = "Swapping Faces || "
            info_text += process_bar.get(i)
            print("\033[1A\033[K", end="", flush=True)
            print(info_text)
            if i % 6 == 0:
                PREVIEW = swapped
                yield "### \n" + info_text, *ui_before()

        yield "### \n ⌛ Merging...", *ui_before()
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
            yield "### \n ⌛ Removing temporary files...", *ui_before()
            shutil.rmtree(temp_path)

        WORKSPACE = output_path
        OUTPUT_FILE = output_video_path

        tot_exec_time = time.time() - start_time
        _min, _sec = divmod(tot_exec_time, 60)

        yield f"✔️ Completed in {int(_min)} min {int(_sec)} sec.", *ui_after_vid()

    ## ------------------------------ DIRECTORY ------------------------------

    elif input_type == "Directory":
        source = cv2.imread(source_path)
        source = analyse_face(
            source,
            FACE_ANALYSER,
            return_single_face=True,
            detect_condition=DETECT_CONDITION,
        )
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
            analysed_target = analyse_face(
                target, FACE_ANALYSER, return_single_face=False
            )

            if condition == "Specific Face":
                swapped = swap_specific(
                    target,
                    analysed_target,
                    analysed_source_specific,
                    models,
                    threshold=distance,
                )
            else:
                swapped = swap_face_with_condition(
                    target, analysed_target, analysed_source, condition, age, models
                )

            filename = os.path.join(temp_path, os.path.basename(file_path))
            cv2.imwrite(filename, swapped)
            info_text = f"### \n ⌛ Processing file {i+1} of {files_length}"
            PREVIEW = swapped[:, :, ::-1]
            yield info_text, *ui_before()

        WORKSPACE = temp_path
        OUTPUT_FILE = filename

        tot_exec_time = time.time() - start_time
        _min, _sec = divmod(tot_exec_time, 60)

        yield f"✔️ Completed in {int(_min)} min {int(_sec)} sec.", *ui_after()

    ## ------------------------------ STREAM ------------------------------

    elif input_type == "Stream":
        yield "### \n ⌛ Starting...", *ui_before()
        global STREAMER
        STREAMER = StreamerThread(src=directory_path)
        STREAMER.start()

        while True:
            try:
                target = STREAMER.frame
                analysed_target = analyse_face(
                    target, FACE_ANALYSER, return_single_face=False
                )
                if condition == "Specific Face":
                    swapped = swap_specific(
                        target,
                        analysed_target,
                        analysed_source_specific,
                        models,
                        threshold=distance,
                    )
                else:
                    swapped = swap_face_with_condition(
                        target, analysed_target, analysed_source, condition, age, models
                    )
                PREVIEW = swapped[:, :, ::-1]
                yield f"Streaming...", *ui_before()
            except AttributeError:
                yield "Streaming...", *ui_before()
        STREAMER.stop()


## ------------------------------ GRADIO FUNC ------------------------------


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
    FACE_ANALYSER = insightface.app.FaceAnalysis(name="buffalo_l", providers=PROVIDER)
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


## ------------------------------ GRADIO GUI ------------------------------

css = """
footer{display:none !important}
"""

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
                        label="Detection Size", value=DETECT_SIZE, interactive=True
                    )
                    detection_threshold = gr.Number(
                        label="Detection Threshold",
                        value=DETECT_THRESH,
                        interactive=True,
                    )
                    apply_detection_settings = gr.Button("Apply settings")

                with gr.Tab("📤 Output Settings"):
                    output_directory = gr.Text(
                        label="Output Directory",
                        value=DEF_OUTPUT_PATH,
                        interactive=True,
                    )
                    output_name = gr.Text(
                        label="Output Name", value="Result", interactive=True
                    )
                    keep_output_sequence = gr.Checkbox(
                        label="Keep output sequence", value=False, interactive=True
                    )

                with gr.Tab("🪄 Other Settings"):
                    with gr.Accordion("Enhance Face", open=True):
                        enable_face_enhance = gr.Checkbox(
                            label="Enable GFPGAN", value=False, interactive=True
                        )
                    with gr.Accordion("Advanced Mask", open=False):
                        enable_face_parser_mask = gr.Checkbox(
                            label="Enable Face Parsing",
                            value=False,
                            interactive=True,
                        )

                        mask_include = gr.Dropdown(
                            mask_regions.keys(),
                            value=MASK_INCLUDE,
                            multiselect=True,
                            label="Include",
                            interactive=True,
                        )
                        mask_exclude = gr.Dropdown(
                            mask_regions.keys(),
                            value=MASK_EXCLUDE,
                            multiselect=True,
                            label="Exclude",
                            interactive=True,
                        )
                        mask_blur = gr.Number(
                            label="Blur Mask",
                            value=MASK_BLUR,
                            minimum=0,
                            interactive=True,
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
                        vid_widget = gr.Video if USE_COLAB else gr.Text
                        video_input = vid_widget(
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
                    output_directory_button = gr.Button(
                        "📂", interactive=False, visible=not USE_COLAB
                    )
                    output_video_button = gr.Button(
                        "🎬", interactive=False, visible=not USE_COLAB
                    )

                with gr.Column():
                    gr.Markdown(
                        '[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/harisreedhar)'
                    )
                    gr.Markdown(
                        "### [Source code](https://github.com/harisreedhar/Swap-Mukham) . [Disclaimer](https://github.com/harisreedhar/Swap-Mukham#disclaimer) . [Gradio](https://gradio.app/)"
                    )

    ## ------------------------------ GRADIO EVENTS ------------------------------

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
        enable_face_enhance,
        enable_face_parser_mask,
        mask_include,
        mask_exclude,
        mask_blur,
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
    if USE_COLAB:
        print("Running in colab mode")

    interface.queue(concurrency_count=2, max_size=20).launch(share=USE_COLAB)
