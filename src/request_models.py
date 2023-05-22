from pydantic import BaseModel, Field

from enum import Enum


class Flip(str, Enum):
    HORIZONTALLY = "horizontally"
    VERTICALLY = "vertically"
    HORIZONTALLY_AND_VERTICALLY = "horizontally_and_vertically"


class DetectionReturnType(str, Enum):
    MARK = "mark"
    MOSAIC = "mosaic"
    BLUR = "blur"


class ImgData(BaseModel):
    img: str


class Detection(BaseModel):
    img: str = None
    video: str = None
    start_level: int = Field(0, alias="start-level")
    return_type: DetectionReturnType = Field(
        DetectionReturnType.MARK, alias="return-type"
    )


class ImgNoiseData(BaseModel):
    img: str
    convert_grayscale: bool = Field(False, alias="convert-grayscale")
    level: int = 100


class ImgSize(BaseModel):
    img: str
    width: int = None
    height: int = None
    keep_aspect_ratio: bool = Field(True, alias="keep-aspect-ratio")


class ImgArea(BaseModel):
    img: str
    x: int = None
    y: int = None
    width: int = None
    height: int = None


class ImgAreaWithLevel(ImgArea):
    level: float


class ImgFlip(BaseModel):
    img: str
    mode: Flip


class ImgRotate(BaseModel):
    img: str
    direction: int = Field(..., ge=0, le=360)


class ImgOilPainting(BaseModel):
    img: str
    size: int = 7
    ratio: int = 1
