import os
import cv2
import numpy as np
from tqdm import tqdm
from utils import scale_bbox_from_center

detect_conditions = [
    "best detection",
    "left most",
    "right most",
    "top most",
    "bottom most",
    "middle",
    "biggest",
    "smallest",
]

swap_options_list = [
    "All Face",
    "Specific Face",
    "Age less than",
    "Age greater than",
    "All Male",
    "All Female",
    "Left Most",
    "Right Most",
    "Top Most",
    "Bottom Most",
    "Middle",
    "Biggest",
    "Smallest",
]

def get_single_face(faces, method="best detection"):
    total_faces = len(faces)
    if total_faces == 1:
        return faces[0]

    print(f"{total_faces} face detected. Using {method} face.")
    if method == "best detection":
        return sorted(faces, key=lambda face: face["det_score"])[-1]
    elif method == "left most":
        return sorted(faces, key=lambda face: face["bbox"][0])[0]
    elif method == "right most":
        return sorted(faces, key=lambda face: face["bbox"][0])[-1]
    elif method == "top most":
        return sorted(faces, key=lambda face: face["bbox"][1])[0]
    elif method == "bottom most":
        return sorted(faces, key=lambda face: face["bbox"][1])[-1]
    elif method == "middle":
        return sorted(faces, key=lambda face: (
                (face["bbox"][0] + face["bbox"][2]) / 2 - 0.5) ** 2 +
                ((face["bbox"][1] + face["bbox"][3]) / 2 - 0.5) ** 2)[len(faces) // 2]
    elif method == "biggest":
        return sorted(faces, key=lambda face: (face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1]))[-1]
    elif method == "smallest":
        return sorted(faces, key=lambda face: (face["bbox"][2] - face["bbox"][0]) * (face["bbox"][3] - face["bbox"][1]))[0]


def analyse_face(image, model, return_single_face=True, detect_condition="best detection", scale=1.0):
    faces = model.get(image)
    if scale != 1: # landmark-scale
        for i, face in enumerate(faces):
            landmark = face['kps']
            center = np.mean(landmark, axis=0)
            landmark = center + (landmark - center) * scale
            faces[i]['kps'] = landmark

    if not return_single_face:
        return faces

    return get_single_face(faces, method=detect_condition)


def cosine_distance(a, b):
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return 1 - np.dot(a, b)


def get_analysed_data(face_analyser, image_sequence, source_data, swap_condition="All face", detect_condition="left most", scale=1.0):
    if swap_condition != "Specific Face":
        source_path, age = source_data
        source_image = cv2.imread(source_path)
        analysed_source = analyse_face(source_image, face_analyser, return_single_face=True, detect_condition=detect_condition, scale=scale)
    else:
        analysed_source_specifics = []
        source_specifics, threshold = source_data
        for source, specific in zip(*source_specifics):
            if source is None or specific is None:
                continue
            analysed_source = analyse_face(source, face_analyser, return_single_face=True, detect_condition=detect_condition, scale=scale)
            analysed_specific = analyse_face(specific, face_analyser, return_single_face=True, detect_condition=detect_condition, scale=scale)
            analysed_source_specifics.append([analysed_source, analysed_specific])

    analysed_target_list = []
    analysed_source_list = []
    whole_frame_eql_list = []
    num_faces_per_frame = []

    total_frames = len(image_sequence)
    curr_idx = 0
    for curr_idx, frame_path in tqdm(enumerate(image_sequence), total=total_frames, desc="Analysing face data"):
        frame = cv2.imread(frame_path)
        analysed_faces = analyse_face(frame, face_analyser, return_single_face=False, detect_condition=detect_condition, scale=scale)

        n_faces = 0
        for analysed_face in analysed_faces:
            if swap_condition == "All Face":
                analysed_target_list.append(analysed_face)
                analysed_source_list.append(analysed_source)
                whole_frame_eql_list.append(frame_path)
                n_faces += 1
            elif swap_condition == "Age less than" and analysed_face["age"] < age:
                analysed_target_list.append(analysed_face)
                analysed_source_list.append(analysed_source)
                whole_frame_eql_list.append(frame_path)
                n_faces += 1
            elif swap_condition == "Age greater than" and analysed_face["age"] > age:
                analysed_target_list.append(analysed_face)
                analysed_source_list.append(analysed_source)
                whole_frame_eql_list.append(frame_path)
                n_faces += 1
            elif swap_condition == "All Male" and analysed_face["gender"] == 1:
                analysed_target_list.append(analysed_face)
                analysed_source_list.append(analysed_source)
                whole_frame_eql_list.append(frame_path)
                n_faces += 1
            elif swap_condition == "All Female" and analysed_face["gender"] == 0:
                analysed_target_list.append(analysed_face)
                analysed_source_list.append(analysed_source)
                whole_frame_eql_list.append(frame_path)
                n_faces += 1
            elif swap_condition == "Specific Face":
                for analysed_source, analysed_specific in analysed_source_specifics:
                    distance = cosine_distance(analysed_specific["embedding"], analysed_face["embedding"])
                    if distance < threshold:
                        analysed_target_list.append(analysed_face)
                        analysed_source_list.append(analysed_source)
                        whole_frame_eql_list.append(frame_path)
                        n_faces += 1

        if swap_condition == "Left Most":
            analysed_face = get_single_face(analysed_faces, method="left most")
            analysed_target_list.append(analysed_face)
            analysed_source_list.append(analysed_source)
            whole_frame_eql_list.append(frame_path)
            n_faces += 1

        elif swap_condition == "Right Most":
            analysed_face = get_single_face(analysed_faces, method="right most")
            analysed_target_list.append(analysed_face)
            analysed_source_list.append(analysed_source)
            whole_frame_eql_list.append(frame_path)
            n_faces += 1

        elif swap_condition == "Top Most":
            analysed_face = get_single_face(analysed_faces, method="top most")
            analysed_target_list.append(analysed_face)
            analysed_source_list.append(analysed_source)
            whole_frame_eql_list.append(frame_path)
            n_faces += 1

        elif swap_condition == "Bottom Most":
            analysed_face = get_single_face(analysed_faces, method="bottom most")
            analysed_target_list.append(analysed_face)
            analysed_source_list.append(analysed_source)
            whole_frame_eql_list.append(frame_path)
            n_faces += 1

        elif swap_condition == "Middle":
            analysed_face = get_single_face(analysed_faces, method="middle")
            analysed_target_list.append(analysed_face)
            analysed_source_list.append(analysed_source)
            whole_frame_eql_list.append(frame_path)
            n_faces += 1

        elif swap_condition == "Biggest":
            analysed_face = get_single_face(analysed_faces, method="biggest")
            analysed_target_list.append(analysed_face)
            analysed_source_list.append(analysed_source)
            whole_frame_eql_list.append(frame_path)
            n_faces += 1

        elif swap_condition == "Smallest":
            analysed_face = get_single_face(analysed_faces, method="smallest")
            analysed_target_list.append(analysed_face)
            analysed_source_list.append(analysed_source)
            whole_frame_eql_list.append(frame_path)
            n_faces += 1

        num_faces_per_frame.append(n_faces)

    return analysed_target_list, analysed_source_list, whole_frame_eql_list, num_faces_per_frame
