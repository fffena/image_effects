import base64
from os.path import join as pathjoin
from uuid import uuid4

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


def img_to_b64(data: str):
    # テスト用
    cv2.imwrite(f"out_img/{uuid4()}.jpg", data)

    ret, dst = cv2.imencode(".jpg", data)
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
    result[y : y + height, x : x + width] = new
    return result


def mosaic(
    img,
    ratio: int | float = None,
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
        result = cv2.resize(
            img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST
        )
        result = cv2.resize(
            result, img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST
        )
    else:
        result = img.copy()
        trimming_area = crop(result, x, y, width, height)
        size = trimming_area.shape[:2][::-1]
        filtered_area = cv2.resize(
            trimming_area, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST
        )
        filtered_area = cv2.resize(filtered_area, size, interpolation=cv2.INTER_NEAREST)
        result = replace_img(result, filtered_area, x, y)
    return result


def blur(
    img, radius, x: int = None, y: int = None, width: int = None, height: int = None
):
    area = (x, y, width, height)
    radius = (radius, radius)
    if any([i is None for i in area]):
        return cv2.blur(img, radius)
    else:
        cropped_area = crop(img, x, y, width, height)
        size = cropped_area.shape[:2][::-1]
        filtered = cv2.blur(cropped_area, radius)
        return replace_img(img, filtered, size[0], size[1])


def oilpainting(img, size, dynRatio):
    result = cv2.xphoto.oilPainting(img, size=size, dynRatio=dynRatio)
    return result


def detect_eye(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        pathjoin(cv2.data.haarcascades, "haarcascade_eye.xml")
    )

    eye = cascade.detectMultiScale(gray_img)
    for ex, ey, ew, eh in eye:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return img
