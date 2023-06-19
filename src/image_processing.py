import base64
from enum import IntEnum

import cv2
import numpy as np


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
