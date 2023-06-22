import binascii
import traceback
from time import time

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import image_processing as imp
import request_models as model
import insightface_utils as inu
from exceptions import NoFaceDetected


app = FastAPI()
app.mount("/demo", StaticFiles(directory="templates/demo", html=True), name="demo")
templates = Jinja2Templates(directory="./templates")


face_d = inu.FaceDetector(
    name=inu.Models.LITE,
    allowed_modules=[inu.Modules.DETECTION, inu.Modules.LANDMARK_2D],
    providers=["CPUExecutionProvider"])
face_d.prepare()


@app.middleware("http")
async def add_exception_handling(request, call_next):
    start = time()
    try:
        response = await call_next(request)
    except (binascii.Error, cv2.error) as e:
        t = traceback.format_exception_only(type(e), e)
        if ("cv2" in t[-1]) and (".empty()" not in t[-1]):
            raise e
        else:
            response = JSONResponse({"msg": "base64 decode failed"}, 400)
    except NoFaceDetected:
        response = JSONResponse({"msg": "not face detected"}, 400)
    # except Exception as e:
    #     response = JSONResponse({"msg": "internal server error!!", 500})
    response.headers["X-Process-Time"] = str(round(time() - start, 3))
    return response


@app.get("/", response_class=HTMLResponse)
def web_app(req: Request):
    return templates.TemplateResponse("index.html", {"request": req})


@app.post("/api/grayscale")
def grayscale(img: model.ImgData):
    img = imp.b64_to_cv2_img(img.img)
    result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return imp.img_to_b64(result)


@app.post("/api/noise")
def noise(d: model.ImgNoiseData):
    img = imp.b64_to_cv2_img(d.img)
    if d.convert_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise = np.random.randint(0, d.level, img.shape[:2])
    filtered_img = img + noise
    return imp.img_to_b64(filtered_img)


@app.post("/api/resize")
def resize(d: model.ImgSize):
    img = imp.b64_to_cv2_img(d.img)
    if d.keep_aspect_ratio:
        height, width = img.shape[:2]
        after_height = int(d.width / width * height)
    else:
        after_height = d.height
    result = cv2.resize(img, (d.width, after_height))
    return imp.img_to_b64(result)


@app.post("/api/crop")
def crop(d: model.ImgArea):
    img = imp.b64_to_cv2_img(d.img)
    result = imp.crop(img, d.x, d.y, d.width, d.height)
    return imp.img_to_b64(result)


@app.post("/api/mosaic")
def mosaic(d: model.ImgAreaWithLevel):
    img = imp.b64_to_cv2_img(d.img)
    result = imp.mosaic(img, d.level, d.x, d.y, d.width, d.height)
    return imp.img_to_b64(result)


@app.post("/api/blur")
def blur(d: model.ImgAreaWithLevel):
    img = imp.b64_to_cv2_img(d.img)
    result = imp.blur(img, d.level, d.x, d.y, d.width, d.height)
    return imp.img_to_b64(result)


@app.post("/api/flip")
def flip(d: model.ImgFlip):
    img = imp.b64_to_cv2_img(d.img)
    flip_modes = [i for i in imp.Flip.__members__.items()]
    flip_value = list(filter(lambda x: x[0].lower() == d.mode.value, flip_modes))[0][1]  # NOQA

    result = imp.flip(img, flip_value)
    return imp.img_to_b64(result)


@app.post("/api/rotate")
def rotate(d: model.ImgRotate):
    img = imp.b64_to_cv2_img(d.img)
    print(d.direction)
    result = imp.rotate(img, d.direction)
    return imp.img_to_b64(result)


@app.post("/api/oilpainting")
def oilpainting(data: model.ImgOilPainting):
    img = imp.b64_to_cv2_img(data.img)
    result = imp.oilpainting(img, data.size, data.ratio)
    return imp.img_to_b64(result)


# ------------ Deep Learning ----------


@app.post("/api/detection/eye")
def detection_eye(data: model.Detection):
    img = imp.b64_to_cv2_img(data.img)
    mark = data.return_type.value == "mark"
    for i in face_d.detect(img):
        if mark:
            result = i.left_eye.draw_frame(img)
            result = i.right_eye.draw_frame(img)
        else:
            TYPES = {"mosaic": imp.mosaic, "blur": imp.blur}
            result = TYPES[data.return_type](img, "")
    return imp.img_to_b64(result)


@app.post("/api/detection/landmark")
def landmark(d: model.Detection):
    img = imp.b64_to_cv2_img(d.img)
    faces = face_d.detect(img)
    for i in faces:
        result = i.draw_face_frame(img)
        result = i.draw_all_landmarks(img, text=True)
    return imp.img_to_b64(result)
# -------------------------------------


if __name__ == "__main__":
    # web.run(app)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
