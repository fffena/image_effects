import cv2
import insightface
import numpy as np

import exceptions


face_detector = insightface.app.FaceAnalysis(
    name="buffalo_s",
    providers=["CPUExecutionProvider"],
    allowed_modules=["detection", "landmark_2d_106"]
)
face_detector.prepare(ctx_id=-1, det_size=(640, 640))


class DetectedFace:
    def __init__(self, face) -> None:
        self._face = face

        self.have_landmark_2d = False
        self.have_landmark_3d = False
        self.have_age = False
        if hasattr(face, "landmark_2d_106"):
            self.have_landmark_2d = True
        if hasattr(face, "landmark_3d_68"):
            self.have_landmark_3d = True
        if hasattr(face, "age"):
            self.have_age = True

    @property
    def face_frame(self):
        return self.get_face_frame()

    @property
    def age(self):
        return self.get_age()

    def draw_all_landmark(self, img, text):
        for i, point in enumerate(self._face.landmark_2d_106.astype(int)):
            co = (point[0], point[1])
            cv2.circle(img, co, 1, (0, 255, 0), 2)
            if text:
                cv2.putText(img, str(i), co, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 1)
        return img

    def draw_face_frame(self, img):
        x1, y1, x2, y2 = self.face_frame
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return img

    def get_face_frame(self):
        return self._face.bbox.astype(int)

    def get_landmark_2d_point(self, type):
        if not self.have_landmark_2d:
            raise Exception("no has landmark_2d")
        x, y = self._face.landmark_2d_106[type].astype(int)
        return Point(x, y)

    def get_landmark_3d_point(self, type):
        if not self.have_landmark_3d:
            raise Exception("aa")
        x, y = self._face.landmark_3d_68[type].astype(int)
        return Point(x, y)

    def get_age(self):
        if not self.have_age:
            raise Exception("hello")
        return self._face.age

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
        self.__x = int(x)
        self.__y = int(y)

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
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def point_tuple(self):
        return (self.x, self.y)


LANDMARKS_2D = {
    "left_eye": (33, 42),
    "right_eye": (87, 96),
    "mouth": (),
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


def detect_face(img: np.ndarray) -> list[DetectedFace]:
    faces = face_detector.get(img)
    if not faces:
        raise exceptions.NoFaceDetected()
    return [DetectedFace(i) for i in faces]