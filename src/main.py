import cv2
import uvicorn
import numpy as np

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

import image_processing as imp
import request_models as model

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def web_app():
    return "<h1>Test</h1>"

@app.post("/api/grayscale")
def grayscale(img: model.ImgData):
    img = imp.b64_to_cv2_img(img.img)
    if isinstance(img, JSONResponse):
        return img
    result_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return imp.img_to_b64(result_img)

@app.post("/api/noise")
def noise(d: model.ImgNoiseData):
    img = imp.b64_to_cv2_img(d.img)
    if isinstance(img, JSONResponse):
        return img
    if d.convert_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise = np.random.randint(0, d.level, img.shape[:2])
    filtered_img = img + noise
    return imp.img_to_b64(filtered_img)

@app.post("/api/resize")
def resize(d: model.ImgSize):
    img = imp.b64_to_cv2_img(d.img)
    if isinstance(img, JSONResponse):
        return img
    if d.keep_aspect_ratio:
        height, width = img.shape[:2]
        after_height = int(d.width / width * height)
    else:
        after_height = d.height
    result_img = cv2.resize(img, (d.width, after_height))
    return imp.img_to_b64(result_img)

@app.post("/api/crop")
def crop(d: model.ImgArea):
    img = imp.b64_to_cv2_img(d.img)
    if isinstance(img, JSONResponse):
        return img
    return imp.img_to_b64(imp.crop(img, d.x, d.y, d.width, d.height))

@app.post("/api/mosaic")
def mosaic(d: model.ImgAreaWithLevel):
    img = imp.b64_to_cv2_img(d.img)
    if isinstance(img, JSONResponse):
        return img
    result = imp.mosaic(img, d.level, d.x, d.y, d.width, d.height)
    return imp.img_to_b64(result)

@app.post("/api/blur")
def blur(d: model.ImgAreaWithLevel):
    img = imp.b64_to_cv2_img(d.img)
    if isinstance(img, JSONResponse):
        return img
    result = imp.blur(img, d.level, d.x, d.y, d.width, d.height)
    return imp.img_to_b64(result)


@app.post("/api/oilpainting")
def oilpainting(data: model.ImgOilPainting):
    img = imp.b64_to_cv2_img(data.img)
    if isinstance(img, JSONResponse):
        return img
    return imp.img_to_b64(imp.oilpainting(img, data.size, data.ratio))


# ------------ Deep Learning --------------
@app.post("/api/detection/eye")
def detection_eye(data: model.ImgData):
    img = imp.b64_to_cv2_img(data.img)
    result = imp.detect_eye(img)
    return imp.img_to_b64(result)
# ---------------- end ----------------

@app.post("/api/test")
def test(img: model.TestData):
    base = img.img.replace("-", "+").replace("_", "/")
    print(base)
    print(img.test_bool)
    decoded_img = imp.b64_to_cv2_img(base)
    result_img =  cv2.cvtColor(decoded_img, cv2.COLOR_BGR2GRAY)
    return imp.img_to_b64(result_img)

if __name__ == "__main__":
    # web.run(app)
    uvicorn.run(app, host="0.0.0.0")