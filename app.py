import os
import cv2
import glob
import time
import torch
import shutil
import argparse
import platform
import datetime
import subprocess
import insightface
import onnxruntime
import numpy as np
import gradio as gr
import threading
import queue
from tqdm import tqdm
import concurrent.futures
from moviepy.editor import VideoFileClip

from nsfw_checker import NSFWChecker
from face_swapper import Inswapper, paste_to_whole
from face_analyser import detect_conditions, get_analysed_data, swap_options_list
from face_parsing import init_parsing_model, get_parsed_mask, mask_regions, mask_regions_to_list
from face_enhancer import get_available_enhancer_names, load_face_enhancer_model, cv2_interpolations
from utils import trim_video, StreamerThread, ProcessBar, open_directory, split_list_by_lengths, merge_img_sequence_from_ref, create_image_grid

## ------------------------------ USER ARGS ------------------------------

parser = argparse.ArgumentParser(description="Swap-Mukham Face Swapper")
parser.add_argument("--out_dir", help="Default Output directory", default=os.getcwd())
parser.add_argument("--batch_size", help="Gpu batch size", default=32)
parser.add_argument("--cuda", action="store_true", help="Enable cuda", default=False)
parser.add_argument(
    "--colab", action="store_true", help="Enable colab mode", default=False
)
user_args = parser.parse_args()

## ------------------------------ DEFAULTS ------------------------------

USE_COLAB = user_args.colab
USE_CUDA = user_args.cuda
DEF_OUTPUT_PATH = user_args.out_dir
BATCH_SIZE = int(user_args.batch_size)
WORKSPACE = None
OUTPUT_FILE = None
CURRENT_FRAME = None
STREAMER = None
DETECT_CONDITION = "best detection"
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
MASK_SOFT_KERNEL = 17
MASK_SOFT_ITERATIONS = 10
MASK_BLUR_AMOUNT = 0.1
MASK_ERODE_AMOUNT = 0.15

FACE_SWAPPER = None
FACE_ANALYSER = None
FACE_ENHANCER = None
FACE_PARSER = None
NSFW_DETECTOR = None
FACE_ENHANCER_LIST = ["NONE"]
FACE_ENHANCER_LIST.extend(get_available_enhancer_names())
FACE_ENHANCER_LIST.extend(cv2_interpolations)

## ------------------------------ SET EXECUTION PROVIDER ------------------------------
# Note: Non CUDA users may change settings here

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

device = "cuda" if USE_CUDA else "cpu"
EMPTY_CACHE = lambda: torch.cuda.empty_cache() if device == "cuda" else None

## ------------------------------ LOAD MODELS ------------------------------

def load_face_analyser_model(name="buffalo_l"):
    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name=name, providers=PROVIDER)
        FACE_ANALYSER.prepare(
            ctx_id=0, det_size=(DETECT_SIZE, DETECT_SIZE), det_thresh=DETECT_THRESH
        )


def load_face_swapper_model(path="./assets/pretrained_models/inswapper_128.onnx"):
    global FACE_SWAPPER
    if FACE_SWAPPER is None:
        batch = int(BATCH_SIZE) if device == "cuda" else 1
        FACE_SWAPPER = Inswapper(model_file=path, batch_size=batch, providers=PROVIDER)


def load_face_parser_model(path="./assets/pretrained_models/79999_iter.pth"):
    global FACE_PARSER
    if FACE_PARSER is None:
        FACE_PARSER = init_parsing_model(path, device=device)

def load_nsfw_detector_model(path="./assets/pretrained_models/open-nsfw.onnx"):
    global NSFW_DETECTOR
    if NSFW_DETECTOR is None:
        NSFW_DETECTOR = NSFWChecker(model_path=path, providers=PROVIDER)


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
    face_enhancer_name,
    enable_face_parser,
    mask_includes,
    mask_soft_kernel,
    mask_soft_iterations,
    blur_amount,
    erode_amount,
    face_scale,
    enable_laplacian_blend,
    crop_top,
    crop_bott,
    crop_left,
    crop_right,
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

    start_time = time.time()
    total_exec_time = lambda start_time: divmod(time.time() - start_time, 60)
    get_finsh_text = lambda start_time: f"✔️ Completed in {int(total_exec_time(start_time)[0])} min {int(total_exec_time(start_time)[1])} sec."

    ## ------------------------------ PREPARE INPUTS & LOAD MODELS ------------------------------

    yield "### \n ⌛ Loading NSFW detector model...", *ui_before()
    load_nsfw_detector_model()

    yield "### \n ⌛ Loading face analyser model...", *ui_before()
    load_face_analyser_model()

    yield "### \n ⌛ Loading face swapper model...", *ui_before()
    load_face_swapper_model()

    if face_enhancer_name != "NONE":
        if face_enhancer_name not in cv2_interpolations:
            yield f"### \n ⌛ Loading {face_enhancer_name} model...", *ui_before()
        FACE_ENHANCER = load_face_enhancer_model(name=face_enhancer_name, device=device)
    else:
        FACE_ENHANCER = None

    if enable_face_parser:
        yield "### \n ⌛ Loading face parsing model...", *ui_before()
        load_face_parser_model()

    includes = mask_regions_to_list(mask_includes)
    specifics = list(specifics)
    half = len(specifics) // 2
    sources = specifics[:half]
    specifics = specifics[half:]
    if crop_top > crop_bott:
        crop_top, crop_bott = crop_bott, crop_top
    if crop_left > crop_right:
        crop_left, crop_right = crop_right, crop_left
    crop_mask = (crop_top, 511-crop_bott, crop_left, 511-crop_right)

    def swap_process(image_sequence):
        ## ------------------------------ CONTENT CHECK ------------------------------

        yield "### \n ⌛ Checking contents...", *ui_before()
        nsfw = NSFW_DETECTOR.is_nsfw(image_sequence)
        if nsfw:
            message = "NSFW Content detected !!!"
            yield f"### \n 🔞 {message}", *ui_before()
            assert not nsfw, message
            return False
        EMPTY_CACHE()

        ## ------------------------------ ANALYSE FACE ------------------------------

        yield "### \n ⌛ Analysing face data...", *ui_before()
        if condition != "Specific Face":
            source_data = source_path, age
        else:
            source_data = ((sources, specifics), distance)
        analysed_targets, analysed_sources, whole_frame_list, num_faces_per_frame = get_analysed_data(
            FACE_ANALYSER,
            image_sequence,
            source_data,
            swap_condition=condition,
            detect_condition=DETECT_CONDITION,
            scale=face_scale
        )

        ## ------------------------------ SWAP FUNC ------------------------------

        yield "### \n ⌛ Generating faces...", *ui_before()
        preds = []
        matrs = []
        count = 0
        global PREVIEW
        for batch_pred, batch_matr in FACE_SWAPPER.batch_forward(whole_frame_list, analysed_targets, analysed_sources):
            preds.extend(batch_pred)
            matrs.extend(batch_matr)
            EMPTY_CACHE()
            count += 1

            if USE_CUDA:
                image_grid = create_image_grid(batch_pred, size=128)
                PREVIEW = image_grid[:, :, ::-1]
                yield f"### \n ⌛ Generating face Batch {count}", *ui_before()

        ## ------------------------------ FACE ENHANCEMENT ------------------------------

        generated_len = len(preds)
        if face_enhancer_name != "NONE":
            yield f"### \n ⌛ Upscaling faces with {face_enhancer_name}...", *ui_before()
            for idx, pred in tqdm(enumerate(preds), total=generated_len, desc=f"Upscaling with {face_enhancer_name}"):
                enhancer_model, enhancer_model_runner = FACE_ENHANCER
                pred = enhancer_model_runner(pred, enhancer_model)
                preds[idx] = cv2.resize(pred, (512,512))
        EMPTY_CACHE()

        ## ------------------------------ FACE PARSING ------------------------------

        if enable_face_parser:
            yield "### \n ⌛ Face-parsing mask...", *ui_before()
            masks = []
            count = 0
            for batch_mask in get_parsed_mask(FACE_PARSER, preds, classes=includes, device=device, batch_size=BATCH_SIZE, softness=int(mask_soft_iterations)):
                masks.append(batch_mask)
                EMPTY_CACHE()
                count += 1

                if len(batch_mask) > 1:
                    image_grid = create_image_grid(batch_mask, size=128)
                    PREVIEW = image_grid[:, :, ::-1]
                    yield f"### \n ⌛ Face parsing Batch {count}", *ui_before()
            masks = np.concatenate(masks, axis=0) if len(masks) >= 1 else masks
        else:
            masks = [None] * generated_len

        ## ------------------------------ SPLIT LIST ------------------------------

        split_preds = split_list_by_lengths(preds, num_faces_per_frame)
        del preds
        split_matrs = split_list_by_lengths(matrs, num_faces_per_frame)
        del matrs
        split_masks = split_list_by_lengths(masks, num_faces_per_frame)
        del masks

        ## ------------------------------ PASTE-BACK ------------------------------

        yield "### \n ⌛ Pasting back...", *ui_before()
        def post_process(frame_idx, frame_img, split_preds, split_matrs, split_masks, enable_laplacian_blend, crop_mask, blur_amount, erode_amount):
            whole_img_path = frame_img
            whole_img = cv2.imread(whole_img_path)
            blend_method = 'laplacian' if enable_laplacian_blend else 'linear'
            for p, m, mask in zip(split_preds[frame_idx], split_matrs[frame_idx], split_masks[frame_idx]):
                p = cv2.resize(p, (512,512))
                mask = cv2.resize(mask, (512,512)) if mask is not None else None
                m /= 0.25
                whole_img = paste_to_whole(p, whole_img, m, mask=mask, crop_mask=crop_mask, blend_method=blend_method, blur_amount=blur_amount, erode_amount=erode_amount)
            cv2.imwrite(whole_img_path, whole_img)

        def concurrent_post_process(image_sequence, *args):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for idx, frame_img in enumerate(image_sequence):
                    future = executor.submit(post_process, idx, frame_img, *args)
                    futures.append(future)

                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Pasting back"):
                    result = future.result()

        concurrent_post_process(
            image_sequence,
            split_preds,
            split_matrs,
            split_masks,
            enable_laplacian_blend,
            crop_mask,
            blur_amount,
            erode_amount
        )


    ## ------------------------------ IMAGE ------------------------------

    if input_type == "Image":
        target = cv2.imread(image_path)
        output_file = os.path.join(output_path, output_name + ".png")
        cv2.imwrite(output_file, target)

        for info_update in swap_process([output_file]):
            yield info_update

        OUTPUT_FILE = output_file
        WORKSPACE = output_path
        PREVIEW = cv2.imread(output_file)[:, :, ::-1]

        yield get_finsh_text(start_time), *ui_after()

    ## ------------------------------ VIDEO ------------------------------

    elif input_type == "Video":
        temp_path = os.path.join(output_path, output_name, "sequence")
        os.makedirs(temp_path, exist_ok=True)

        yield "### \n ⌛ Extracting video frames...", *ui_before()
        image_sequence = []
        cap = cv2.VideoCapture(video_path)
        curr_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:break
            frame_path = os.path.join(temp_path, f"frame_{curr_idx}.jpg")
            cv2.imwrite(frame_path, frame)
            image_sequence.append(frame_path)
            curr_idx += 1
        cap.release()
        cv2.destroyAllWindows()

        for info_update in swap_process(image_sequence):
            yield info_update

        yield "### \n ⌛ Merging sequence...", *ui_before()
        output_video_path = os.path.join(output_path, output_name + ".mp4")
        merge_img_sequence_from_ref(video_path, image_sequence, output_video_path)

        if os.path.exists(temp_path) and not keep_output_sequence:
            yield "### \n ⌛ Removing temporary files...", *ui_before()
            shutil.rmtree(temp_path)

        WORKSPACE = output_path
        OUTPUT_FILE = output_video_path

        yield get_finsh_text(start_time), *ui_after_vid()

    ## ------------------------------ DIRECTORY ------------------------------

    elif input_type == "Directory":
        extensions = ["jpg", "jpeg", "png", "bmp", "tiff", "ico", "webp"]
        temp_path = os.path.join(output_path, output_name)
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        os.mkdir(temp_path)

        file_paths =[]
        for file_path in glob.glob(os.path.join(directory_path, "*")):
            if any(file_path.lower().endswith(ext) for ext in extensions):
                img = cv2.imread(file_path)
                new_file_path = os.path.join(temp_path, os.path.basename(file_path))
                cv2.imwrite(new_file_path, img)
                file_paths.append(new_file_path)

        for info_update in swap_process(file_paths):
            yield info_update

        PREVIEW = cv2.imread(file_paths[-1])[:, :, ::-1]
        WORKSPACE = temp_path
        OUTPUT_FILE = file_paths[-1]

        yield get_finsh_text(start_time), *ui_after()

    ## ------------------------------ STREAM ------------------------------

    elif input_type == "Stream":
        pass


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
    if value.startswith("Age"):
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=True),
        )
    elif value == "Specific Face":
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
                    swap_option = gr.Dropdown(
                        swap_options_list,
                        info="Choose which face or faces in the target image to swap.",
                        multiselect=False,
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
                    face_scale = gr.Slider(
                        label="Face Scale",
                        minimum=0,
                        maximum=2,
                        value=1,
                        interactive=True,
                    )

                    face_enhancer_name = gr.Dropdown(
                        FACE_ENHANCER_LIST, label="Face Enhancer", value="NONE", multiselect=False, interactive=True
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
                        mask_soft_kernel = gr.Number(
                            label="Soft Erode Kernel",
                            value=MASK_SOFT_KERNEL,
                            minimum=3,
                            interactive=True,
                            visible = False
                        )
                        mask_soft_iterations = gr.Number(
                            label="Soft Erode Iterations",
                            value=MASK_SOFT_ITERATIONS,
                            minimum=0,
                            interactive=True,

                        )


                    with gr.Accordion("Crop Mask", open=False):
                        crop_top = gr.Slider(label="Top", minimum=0, maximum=511, value=0, step=1, interactive=True)
                        crop_bott = gr.Slider(label="Bottom", minimum=0, maximum=511, value=511, step=1, interactive=True)
                        crop_left = gr.Slider(label="Left", minimum=0, maximum=511, value=0, step=1, interactive=True)
                        crop_right = gr.Slider(label="Right", minimum=0, maximum=511, value=511, step=1, interactive=True)


                    erode_amount = gr.Slider(
                            label="Mask Erode",
                            minimum=0,
                            maximum=1,
                            value=MASK_ERODE_AMOUNT,
                            step=0.05,
                            interactive=True,
                        )

                    blur_amount = gr.Slider(
                            label="Mask Blur",
                            minimum=0,
                            maximum=1,
                            value=MASK_BLUR_AMOUNT,
                            step=0.05,
                            interactive=True,
                        )

                    enable_laplacian_blend = gr.Checkbox(
                        label="Laplacian Blending",
                        value=True,
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
                        ["Image", "Video", "Directory"],
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

                with gr.Box():
                    with gr.Row():
                        gr.Markdown(
                            "### [🤝 Sponsor](https://github.com/sponsors/harisreedhar)"
                        )
                        gr.Markdown(
                            "### [👨‍💻 Source code](https://github.com/harisreedhar/Swap-Mukham)"
                        )
                        gr.Markdown(
                            "### [⚠️ Disclaimer](https://github.com/harisreedhar/Swap-Mukham#disclaimer)"
                        )
                        gr.Markdown(
                            "### [🌐 Run in Colab](https://colab.research.google.com/github/harisreedhar/Swap-Mukham/blob/main/swap_mukham_colab.ipynb)"
                        )
                        gr.Markdown(
                            "### [🤗 Acknowledgements](https://github.com/harisreedhar/Swap-Mukham#acknowledgements)"
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
        show_progress=True,
    )

    end_frame_event = end_frame.release(
        fn=slider_changed,
        inputs=[show_trim_preview_btn, video_input, end_frame],
        outputs=[preview_image, preview_video],
        show_progress=True,
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
        face_enhancer_name,
        enable_face_parser_mask,
        mask_include,
        mask_soft_kernel,
        mask_soft_iterations,
        blur_amount,
        erode_amount,
        face_scale,
        enable_laplacian_blend,
        crop_top,
        crop_bott,
        crop_left,
        crop_right,
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
        fn=process, inputs=swap_inputs, outputs=swap_outputs, show_progress=True
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
        show_progress=True,
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
