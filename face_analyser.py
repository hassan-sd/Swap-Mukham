detect_conditions = [
    "left most",
    "right most",
    "top most",
    "bottom most",
    "most width",
    "most height",
]


def analyse_face(image, model, return_single_face=True, detect_condition="left most"):
    faces = model.get(image)
    if not return_single_face:
        return faces

    total_faces = len(faces)
    if total_faces == 1:
        return faces[0]

    print(f"{total_faces} face detected. Using {detect_condition} face.")
    if detect_condition == "left most":
        return sorted(faces, key=lambda face: face["bbox"][0])[0]
    elif detect_condition == "right most":
        return sorted(faces, key=lambda face: face["bbox"][0])[-1]
    elif detect_condition == "top most":
        return sorted(faces, key=lambda face: face["bbox"][1])[0]
    elif detect_condition == "bottom most":
        return sorted(faces, key=lambda face: face["bbox"][1])[-1]
    elif detect_condition == "most width":
        return sorted(faces, key=lambda face: face["bbox"][2])[-1]
    elif detect_condition == "most height":
        return sorted(faces, key=lambda face: face["bbox"][3])[-1]
