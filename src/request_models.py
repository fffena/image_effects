from pydantic import BaseModel, Field

class ImgData(BaseModel):
    img: str

class ImgNoiseData(BaseModel):
    img: str
    convert_grayscale: bool = Field(False, alias="convert-grayscale")
    level: int = 100

class ImgSizeSelectData(BaseModel):
    img: str
    width: int = None
    height: int = None
    aspect_ratio: bool = Field(True, alias="aspect-ratio")

class ImgAreaSelect(BaseModel):
    img: str
    x: int = None
    y: int = None
    width: int = None
    height: int = None
    level: int

class ImgOilPainting(BaseModel):
    img: str
    size: int = 7
    ratio: int = 1

class TestData(ImgData, BaseModel):
    test_bool: bool
    test_str: str