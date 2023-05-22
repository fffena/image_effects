import binascii
import traceback
from multiprocessing import Pool, TimeoutError
from time import time

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

import image_processing as imp
import request_models as model
from exceptions import NoFaceDetected

app = FastAPI()
templates = Jinja2Templates(directory="./templates")


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
    flip_value = list(filter(lambda x: x[0].lower() == d.mode.value, flip_modes))[0][1]

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
    if data.img:
        image = True
        video = False
    elif data.video:
        image = False
        video = True
    else:
        return {"msg": "どっちか入れて"}

    img = imp.b64_to_cv2_img(data.img)
    mark = True if data.return_type.value == "mark" else False
    try:
        with Pool(processes=1) as p:
            apply_result = p.apply_async(imp.detect_eye, (img, data.start_level, mark))
            result = apply_result.get(timeout=20)
    except NoFaceDetected:
        return {"msg": "No Face detected"}
    except TimeoutError:
        return {"msg": "timeouted"}
    if not mark:
        types = {"mosaic": imp.mosaic, "blur": imp.blur}
        for eyes in result:
            for eye in eyes:
                x = eye[0][0]
                y = eye[0][1]
                width = eye[1][0] - x
                height = eye[1][1] - y
                if isinstance(result, tuple):
                    result = types[data.return_type.value](
                        img, 0.2, x, y, width, height
                    )
                else:
                    print("he")
                    result = types[data.return_type.value](
                        result, 0.2, x, y, width, height
                    )

    return imp.img_to_b64(result)


# -------------------------------------


if __name__ == "__main__":
    # web.run(app)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
