import cv2
import numpy as np
from insightface.utils import face_align
from face_parsing.swap import swap_regions
from utils import add_logo_to_image

swap_options_list = [
    "All face",
    "Age less than",
    "Age greater than",
    "All Male",
    "All Female",
    "Specific Face",
]


def swap_face(whole_img, target_face, source_face, models):
    inswapper = models.get("swap")
    face_enhancer = models.get("enhance", None)
    face_parser = models.get("face_parser", None)
    fe_enable = models.get("enhance_sett", False)

    bgr_fake, M = inswapper.get(whole_img, target_face, source_face, paste_back=False)
    image_size = 128 if not fe_enable else 512
    aimg, _ = face_align.norm_crop2(whole_img, target_face.kps, image_size=image_size)

    if face_parser is not None:
        fp_enable, includes, smooth_mask, blur_amount = models.get("face_parser_sett")
        if fp_enable:
            bgr_fake = swap_regions(
                bgr_fake, aimg, face_parser, smooth_mask, includes=includes, blur=blur_amount
            )

    if fe_enable:
        _, bgr_fake, _ = face_enhancer.enhance(
            bgr_fake, paste_back=True, has_aligned=True
        )
        bgr_fake = bgr_fake[0]
        M /= 0.25

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


def swap_face_with_condition(
    whole_img, target_faces, source_face, condition, age, models
):
    swapped = whole_img.copy()

    for target_face in target_faces:
        if condition == "All face":
            swapped = swap_face(swapped, target_face, source_face, models)
        elif condition == "Age less than" and target_face["age"] < age:
            swapped = swap_face(swapped, target_face, source_face, models)
        elif condition == "Age greater than" and target_face["age"] > age:
            swapped = swap_face(swapped, target_face, source_face, models)
        elif condition == "All Male" and target_face["gender"] == 1:
            swapped = swap_face(swapped, target_face, source_face, models)
        elif condition == "All Female" and target_face["gender"] == 0:
            swapped = swap_face(swapped, target_face, source_face, models)

    return swapped


def swap_specific(source_specifics, target_faces, whole_img, models, threshold=0.6):
    swapped = whole_img.copy()

    for source_face, specific_face in source_specifics:
        specific_embed = specific_face["embedding"]
        specific_embed /= np.linalg.norm(specific_embed)

        for target_face in target_faces:
            target_embed = target_face["embedding"]
            target_embed /= np.linalg.norm(target_embed)
            cosine_distance = 1 - np.dot(specific_embed, target_embed)
            if cosine_distance > threshold:
                continue
            swapped = swap_face(swapped, target_face, source_face, models)

    return swapped
