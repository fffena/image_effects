import base64
from enum import IntEnum

import cv2
import numpy as np
import dlib
from imutils import face_utils

import exceptions


def b64_to_cv2_img(data: str):
    data = data.replace("-", "+").replace("_", "/")
    if "," in data:
        data = data.split(",")[1]
    decoded_bytes = base64.b64decode(data)
    ndarray = np.frombuffer(decoded_bytes, np.uint8)
    img = cv2.imdecode(ndarray, cv2.IMREAD_COLOR)
    return img


def b64_to_video(data: str):
    data = data.replace("-", "+").replace("_", "/")
    if "," in data:
        data = data.split(",")[1]
    decoded = base64.b64decode(data)
    ndarray = np.frombuffer(decoded, np.uint8)
    return cv2.imdecode(ndarray, cv2.IMREAD_COLOR)


def img_to_b64(data):
    # cv2.imwrite(f"../dist/{uuid4()}.jpg", data)  # test

    ret, dst = cv2.imencode(".jpg", data.astype(np.uint8))
    return base64.b64encode(dst).decode("utf-8", "strict")


def crop(img, x, y, width, height):
    # はみ出すとエラーが出るのでいい感じに収める
    tx = min(x + width, img.shape[1])
    ty = min(y + height, img.shape[0])
    width = tx - x
    height = ty - y

    return img[y:ty, x:tx]


def replace_img(img, new, x, y):
    width, height = new.shape[:2][::-1]
    result = img.copy()
    result[y:y + height, x:x + width] = new
    return result


def mosaic(
    img,
    ratio: int | float = 0.5,
    x: int = None,
    y: int = None,
    width: int = None,
    height: int = None,
):
    if isinstance(ratio, float) and ratio > 1.0:
        ratio = 1.0
    elif isinstance(ratio, int):
        ratio = ratio / 100
    area = (x, y, width, height)
    if any([i is None for i in area]):
        result = cv2.resize(img, None, fx=ratio, fy=ratio,
                            interpolation=cv2.INTER_NEAREST)

        result = cv2.resize(result, img.shape[:2][::-1],
                            interpolation=cv2.INTER_NEAREST)
        return result
    else:
        result = img.copy()
        trimming_area = crop(result, x, y, width, height)
        size = trimming_area.shape[:2][::-1]
        filtered_area = cv2.resize(
            trimming_area, None, fx=ratio, fy=ratio,
            interpolation=cv2.INTER_NEAREST
        )
        filtered_area = cv2.resize(filtered_area, size,
                                   interpolation=cv2.INTER_NEAREST)
        return replace_img(result, filtered_area, x, y)


def blur(
    img, radius,
    x: int = None, y: int = None,
    width: int = None, height: int = None
):
    if isinstance(radius, float):
        radius = int(radius * 10)
    area = (x, y, width, height)
    radius = (radius, radius)
    if any([i is None for i in area]):
        return cv2.blur(img, radius)
    else:
        cropped_area = crop(img, x, y, width, height)
        filtered = cv2.blur(cropped_area, radius)
        return replace_img(img, filtered, x, y)


class Flip(IntEnum):
    HORIZONTALLY = 0
    VERTICALLY = 1
    HORIZONTALLY_AND_VERTICALLY = -1


def flip(img, mode: Flip | int):
    if isinstance(mode, Flip):
        mode_value = mode.value
    else:
        mode_value = mode
    return cv2.flip(img, mode_value)


def rotate(img, angle, center=None):
    if center is None:
        center = (img.shape[1] // 2, img.shape[0] // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, img.shape[:2][::-1])
    return rotated


def oilpainting(img, size, dynRatio):
    result = cv2.xphoto.oilPainting(img, size=size, dynRatio=dynRatio)
    return result


def eye(landmarks, side):
    landmarks = face_utils.shape_to_np(landmarks)
    if side == "left":
        eye_left = 36
        eye_right = 39
        eye_top = 38
        eye_bottom = 41
    elif side == "right":
        eye_left = 42
        eye_right = 45
        eye_top = 43
        eye_bottom = 46
    else:
        raise ValueError("left or right")

    left_x = landmarks[eye_left][0]
    right_x = landmarks[eye_right][0]
    eye_width_half = (right_x - left_x) // 2

    top_y = landmarks[eye_top][1]
    bottom_y = landmarks[eye_bottom][1]

    top_left = (left_x - eye_width_half, top_y - eye_width_half)
    bottom_right = (right_x + eye_width_half, bottom_y + eye_width_half)
    return top_left, bottom_right


def detect_eye(img, start_level: int = 0, mark: bool = False):
    face_detector = dlib.get_frontal_face_detector()
    for i in range(start_level, start_level + 3):
        faces = face_detector(img, i)
        if faces:
            break
    else:
        raise exceptions.NoFaceDetected("No Face Detected")
    predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")  # NOQA
    eyes = []
    for face in faces:
        left_eye = eye(predictor(img, face), "left")
        right_eye = eye(predictor(img, face), "right")
        coordinate = (left_eye, right_eye)

        if mark:
            cv2.rectangle(img, left_eye[0], left_eye[1], (0, 255, 0), 2)
            cv2.rectangle(img, right_eye[0], right_eye[1], (0, 255, 0), 2)
        else:
            eyes.append(coordinate)
    if mark:
        return img
    else:
        return tuple(eyes)
