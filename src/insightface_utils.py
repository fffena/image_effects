import cv2
import numpy as np
import insightface  # NOQA
from insightface.app import FaceAnalysis
from insightface.app.common import Face
from insightface.utils import DEFAULT_MP_NAME

import exceptions
from time import time


class Models:
    NORMAL = "buffalo_l"
    LITE = "buffalo_s"
    DETECTION_ONLY = "buffalo_sc"


class Modules:
    DETECTION = "detection"
    LANDMARK_2D = "landmark_2d_106"
    LANDMARK_3D = "landmark_3d_68"
    AGE_AND_GENDER = "genderage"


class DetectedFace:
    def __init__(self, face: Face, process_time=None) -> None:
        self._face = face
        self.process_time = round(process_time, 5)

        self.have_landmark_2d = hasattr(face, "landmark_2d_106")
        self.have_landmark_3d = hasattr(face, "landmark_3d_68")
        self.have_age = hasattr(face, "age")

    @property
    def face_frame(self):
        return self._get_face_frame()

    @property
    def age(self):
        return self._get_age()

    def draw_all_landmarks(self, img, text):
        for i, point in enumerate(self._face.landmark_2d_106.astype(int)):
            co = (point[0], point[1])
            cv2.circle(img, co, 1, (0, 255, 0), 2)
            if text:
                cv2.putText(img, str(i), co, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 1)
        return img

    def draw_face_frame(self, img, draw_metadata=True):
        x1, y1, x2, y2 = self.face_frame
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if draw_metadata:
            cv2.rectangle(img, (x1, y1-40), (x2+10, y1), (255, 0, 0), -1)
            cv2.putText(img, f"age: {self.age}", (x1+5, y1-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
            cv2.putText(img, text=f"gender: {self._face.sex}", org=(x1+100, y1-20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7,
                        color=(0, 255, 0), thickness=2)

        return img

    def draw_process_time(self, img):
        cv2.putText(img, f"process time: {self.process_time}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img

    def _get_face_frame(self):
        return self._face.bbox.astype(int)

    def _get_age(self):
        if not self.have_age:
            raise Exception('face object does not have age')
        return self._face.age

    def get_landmark_2d_point(self, type):
        if not self.have_landmark_2d:
            raise Exception('face object does not have landmark_2d_106')
        x, y = self._face.landmark_2d_106[type].astype(int)
        return Point(x, y)

    def get_landmark_3d_point(self, type):
        if not self.have_landmark_3d:
            raise Exception('face object does not have landmark_3d_68')
        x, y = self._face.landmark_3d_68[type].astype(int)
        return Point(x, y)

    @property
    def left_eye(self):
        s, y = LANDMARKS_2D["left_eye"]
        return Landmark(self._face.landmark_2d_106[s:y])

    @property
    def right_eye(self):
        s, y = LANDMARKS_2D["right_eye"]
        return Landmark(self._face.landmark_2d_106[s:y])

    @property
    def left_eyebrow(self):
        s, y = LANDMARKS_2D["left_eyebrow"]
        return Landmark(self._face.landmark_2d_106[s:y])

    @property
    def right_eyebrow(self):
        s, y = LANDMARKS_2D["right_eyebrow"]
        return Landmark(self._face.landmark_2d_106[s:y])


class Landmark:
    def __init__(self, obj: np.ndarray) -> None:
        all_x = [i[0] for i in obj.astype(int)]
        all_y = [i[1] for i in obj.astype(int)]
        self.left = Point(min(all_x), min(all_y))
        self.right = Point(max(all_x), max(all_y))

    def draw_frame(self, img: np.ndarray):
        width = self.right.x - self.left.x
        height = self.right.y - self.left.y
        thinkness = 2 if (width + height) > 75 else 1
        cv2.rectangle(img, self.left.point_tuple, self.right.point_tuple,
                      (0, 255, 0), thinkness)
        return img


class Point:
    def __init__(self, x, y) -> None:
        self.x = int(x)
        self.y = int(y)

    def __add__(self, obj):
        if not isinstance(obj, Point):
            raise TypeError
        return Point(self.x + obj.x, self.y + obj.y)

    def __sub__(self, obj):
        if not isinstance(obj, Point):
            raise TypeError
        return Point(self.x - obj.x, self.y - obj.y)

    def __mul__(self, obj):
        if not isinstance(obj, Point):
            raise TypeError
        return Point(self.x * obj.x, self.y * obj.y)

    def __truediv__(self, obj):
        if not isinstance(obj, Point):
            raise TypeError
        return Point(self.x / obj.x, self.y / obj.y)

    @property
    def point_tuple(self):
        return (self.x, self.y)


class FaceDetector(FaceAnalysis):
    def __init__(self, name=DEFAULT_MP_NAME, root="~/.insightface",
                 allowed_modules=None, **kwargs):
        super().__init__(name, root, allowed_modules, **kwargs)

    def prepare(self, gpu=False, det_thresh=0.5, det_size=(640, 640)):
        ctx_id = 0 if gpu else -1
        super().prepare(ctx_id, det_thresh, det_size)

    def detect(self, img: np.ndarray) -> list[DetectedFace]:
        start = time()
        faces = super().get(img)
        end = time()
        if not faces:
            raise exceptions.NoFaceDetected()
        return [DetectedFace(i, end-start) for i in faces]


LANDMARKS_2D = {
    "left_eye": (33, 42),
    "right_eye": (87, 96),
    "mouth": (52, 71),
    "nose": (),
    "left_eyebrow": (43, 51),
    "right_eyebrow": (97, 105),
    "contour": (),
}

LANDMARKS_3D = {
    "left_eye": (),
    "right_eye": (),
    "mouth": (),
    "nose": (),
    "left_eyebrow": (),
    "right_eyebrow": (),
    "contour": (),
}


if __name__ == "__main__":
    a = FaceDetector(name=Models.DETECTION_ONLY,
                     allowed_modules=[Modules.DETECTION])
    a.prepare()
    img = cv2.imread("./img.jpg")
    faces = a.detect(img)
    for i in faces:
        img = i.draw_face_frame(img)
    cv2.imwrite("a.jpg", img)
