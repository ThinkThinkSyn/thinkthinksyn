if __name__ == "__main__":  # for debugging
    import os, sys
    _proj_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
    sys.path.append(_proj_path)
    __package__ = 'common_utils.data_structs.file_models'

import os
import base64
import logging
import requests
import numpy as np

from io import BytesIO
from pathlib import Path
from PIL import Image as PILImage
from pydantic_core import core_schema
from typing import cast, Self, Literal, TYPE_CHECKING, Coroutine, TypeAlias
from typing_extensions import override

from ...decorators import cache
from ...type_utils import bytes_to_base64
from ...concurrent_utils import run_any_func, is_async_callable

from ..geometry import Box2D, Point2D
from .loader import save_get, AcceptableFileSource
from ._utils import _hash_md5, _try_get_from_dict, _get_media_json_schema, _dump_media_dict

CommonImgFormat: TypeAlias = Literal['jpg', 'png', 'gif', 'bmp', 'tiff', 'webp']
ImageColorMode: TypeAlias = Literal['rgb', 'rgba', 'l', 'p', '1', 'cmyk']

_logger = logging.getLogger(__name__)

def _tidy_color_mode(mode: ImageColorMode)->str:
    return mode.upper() 

def _tidy_format(format: CommonImgFormat)->str:
    if format == 'jpg':
        return 'jpeg'
    return format.lower() 

def _crop_img(
    img: AcceptableFileSource | np.ndarray | PILImage.Image,
    region: Box2D,
    return_mode: Literal["bytes", "base64", "image"] = "image",
    color_mode: Literal["unchange", "L", "RGB", "RGBA"] = "unchange",
):
    if isinstance(img, bytes):
        img_obj = PILImage.open(BytesIO(img))
    elif isinstance(img, str):
        if img.startswith("http://") or img.startswith("https://"):
            img_obj = PILImage.open(BytesIO(requests.get(img).content))
        elif os.path.exists(img):
            img_obj = PILImage.open(img)
        else:
            img_obj = PILImage.open(base64.b64decode(img))
    elif isinstance(img, Path):
        img_obj = PILImage.open(img)
    elif isinstance(img, np.ndarray):
        img_obj = PILImage.fromarray(img)
    elif isinstance(img, PILImage.Image):
        img_obj = img
    else:
        raise ValueError("Unexpected image type. It should be a url, base64 string, bytes, path or numpy array.")

    if region.mode == "relative":
        region = region.to_absolute(img_obj.size)
    img_obj = img_obj.crop((region.left_top.x, region.left_top.y, region.right_bottom.x, region.right_bottom.y))  # type: ignore
    if color_mode != "unchange":
        img_obj = img_obj.convert(color_mode)

    if return_mode == "bytes":
        buf = BytesIO()
        img_obj.save(buf, format="PNG")
        return buf.getvalue()
    elif return_mode == "base64":
        buf = BytesIO()
        img_obj.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    else:
        return img_obj

def _is_svg(data: str|Path|bytes)->bool:
    if isinstance(data, Path):
        if not data.is_file():
            return False
        if data.suffix.lower() == '.svg':
            return True
        try:
            with open(data, 'r', encoding='utf-8') as f:
                header = f.read(512)
            if '<svg' in header:
                return True
        except:
            return False
    elif isinstance(data, str):
        if '<svg' in data[:512]:
            return True
    elif isinstance(data, bytes):
        if b'<svg' in data[:512]:
            return True
    return False

def _svg_convert(source: str|Path, format='png')-> bytes:
    import pyvips
    if isinstance(source, Path):
        if not source.is_file():
            raise ValueError(f'SVG file not found: {source}')
        source = str(source.resolve())
        vips_img: pyvips.Image = pyvips.Image.new_from_file(source, access='sequential')    # type: ignore
    elif len(source) < 1024 and os.path.isfile(source):
        vips_img = pyvips.Image.new_from_file(source, access='sequential')  # type: ignore
    else:
        vips_img = pyvips.Image.new_from_buffer(source.encode('utf-8'), '', access='sequential')    # type: ignore
    if not vips_img:
        raise ValueError(f'Failed to load SVG image from `...{source[-64:]}`')  # type: ignore
    return vips_img.write_to_buffer(f'.{format}')   # type: ignore

class _ImageMeta(type(PILImage.Image)):
    def __subclasscheck__(self, subclass: type)-> bool:
        if subclass.__name__ in (Image.__name__, _DeferImageLoader.__name__):
            return True
        return super().__subclasscheck__(subclass)
    
    def __instancecheck__(self, instance: object) -> bool:
        if instance.__class__.__name__ in (Image.__name__, _DeferImageLoader.__name__):
            return True
        return super().__instancecheck__(instance)

class Image(PILImage.Image, metaclass=_ImageMeta):
    '''Advanced Image class with pydantic support'''
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        def validator(data):
            if isinstance(data, dict):
                data = _try_get_from_dict(data, 'data', 'content', 'img', 'image', 'source', 'url')
            if not isinstance(data, cls):
                data = cls.Load(data)   # type: ignore
            return data
        
        def serializer(img: 'Image'):
            if isinstance(img, _DeferImageLoader):
                if img.__real_image__:
                    return _dump_media_dict(img.__real_image__.to_base64(), Image)
                elif isinstance(img.__image_source__, (str, Path)):
                    if isinstance(img.__image_source__, Path):
                        source = str(img.__image_source__)
                    else:
                        source = img.__image_source__
                    return _dump_media_dict(source, Image)
                else:
                    image_obj = img._defer_load_image()
                    return _dump_media_dict(image_obj.to_base64(), Image)
            else:
                if img.channel_count <= 3:
                    format = 'jpg'
                else:
                    format = 'png'
                return _dump_media_dict(img.to_base64(format=format), cls)

        validate_schema = core_schema.no_info_after_validator_function(validator, core_schema.any_schema())
        serialize_schema = core_schema.plain_serializer_function_ser_schema(serializer)
        return core_schema.json_or_python_schema(
            json_schema=validate_schema,
            python_schema=validate_schema,
            serialization=serialize_schema
        )
        
    @classmethod
    def __get_pydantic_json_schema__(cls, cs, handler):
        return _get_media_json_schema(cls)
    
    @property
    def pixel_count(self)->int:
        return self.size[0] * self.size[1]
    
    @property
    def channel_count(self):
        return len(self.getbands())    
    
    @cache
    def size_in_bytes(
        self, 
        mode: ImageColorMode|None = None,
        format: Literal['pil']|CommonImgFormat|str|None = None,
    )->int:
        format = _tidy_format(format) if format else self.format    # type: ignore
        mode = _tidy_color_mode(mode) if mode else self.mode        # type: ignore
        if mode in ('pil', None):
            b = super().tobytes(encoder_name, *args)    # type: ignore
            return len(b)
        else:
            if not mode:
                mode = 'RGBA' if self.channel_count > 3 else 'RGB'  
            if not format:
                format = 'jpeg' if self.channel_count <= 3 else 'png'
            bytes_io = BytesIO()
            if mode != self.mode:
                img = self.convert(mode)
            else:
                img = self
            img.save(bytes_io, format=format)
            return bytes_io.tell()

    @override
    @cache(maxsize=2)
    def tobytes(
        self, 
        encoder_name: str ="raw", 
        *args,
        format: Literal['pil']|CommonImgFormat|str|None=None,
        mode: ImageColorMode|None = None,
    ):
        '''
        get the data of this image in bytes format
        Args:
            - encoder_name: the encoder name, default is 'raw'.
            - format: the image format, default is 'pil'(original format of PLI image).
                    note that 'jpg' and 'jpeg' are the same. If None, remain current format.
            - mode: the image color mode. If None, remain current mode.
        '''
        format = _tidy_format(format) if format else self.format    # type: ignore
        mode = _tidy_color_mode(mode) if mode else self.mode        # type: ignore
        if mode == 'pil':
            return super().tobytes(encoder_name, *args) 
        else:
            if not mode:
                mode = 'RGBA' if self.channel_count > 3 else 'RGB'      # type: ignore
            if not format:
                format = 'jpeg' if self.channel_count <= 3 else 'png'
            bytes_io = BytesIO()
            if mode != self.mode:
                img = self.convert(mode)
            else:
                img = self
            img.save(bytes_io, format=format)
            return bytes_io.getvalue()
    
    to_bytes = tobytes   # alias

    def to_base64(
        self, 
        format: Literal['pil']|CommonImgFormat|str|None = None,
        mode: ImageColorMode|None = None,
        url_scheme: bool = False,
    )->str:
        '''
        Get the data of this image in base64 format.
        Args:
            - format: the image format, default is 'pil'(original format of PLI image).
                    note that 'jpg' and 'jpeg' are the same. If None, remain current format.
            - mode: the image color mode. If None, remain current mode.
            - url_scheme: if True, return the base64 string with data URL scheme, e.g. 'data:image/png;base64,...'.
                         This is only available when `format`!='pil'.
        '''
        format = _tidy_format(format) if format else self.format        # type: ignore
        b64 = bytes_to_base64(self.tobytes(format=format, mode=mode))
        if url_scheme and format != 'pil':
            if not format:
                format = 'jpg' if self.channel_count <= 3 else 'png'
            format = format.lower().strip()
            b64 = f'data:image/{format};base64,{b64}'
        return b64

    def copy(self)->Self:
        '''return a copy of this image'''
        return self.CastPILImage(super().copy())

    @cache
    def to_md5_hash(
        self,
        format: Literal['pil']|CommonImgFormat|str|None = None,
        mode: ImageColorMode|None = None,
    )->str:
        '''
        Get the md5 hash of this image.
        Args:
            - format: the image format, default is 'pil'(original format of PLI image).
                    note that 'jpg' and 'jpeg' are the same. If None, remain current format.
            - mode: the image color mode. If None, remain current mode.
        '''
        return _hash_md5(self.tobytes(format=format, mode=mode))
        
    @override
    def crop(self, region: Box2D)->Self:
        """
        Cut the image with a given region.

        Args:
            - region: the target region to crop. You can set the mode `relative`/`absolute` to define the region,
                    e.g. box=Box(left_top=(0.1, 0.1), right_bottom=(0.9, 0.9), mode='relative')
        """
        if not isinstance(region, Box2D):
            return super().crop(region)
        img = _crop_img(self, region, return_mode='image')
        return self.CastPILImage(img)       # type: ignore
    
    def crop_into(
        self, 
        pieces: int,
        method: Literal['horizontal', 'vertical', 'square'] = 'horizontal',
        overlap: int|float = 0.5
    )->list[Self]:
        '''
        Crop this image into multiple pieces.
        Args:
            - pieces: the number of pieces to crop. NOTE: when mode==`square`, pieces should be event number>=4
            - method: the cropping method, can be 'horizontal', 'vertical' or 'square':
                    ```
                    `horizontal`:
                       -------    -------
                       |     | -> |     |
                       |     |    -------
                       -------    |     |
                                  -------
                    `vertical`:
                        --------    -------
                        |      |    |  |  |
                        |      | -> |  |  |
                        --------    -------
                    `square`:
                        --------    -------
                        |      |    |  |  |
                        |      | -> ------
                        --------    |  |  |
                                    ------
                    ```
            - overlap: the overlap ratio or pixels between pieces. If float, 
                    it means the ratio (0~1); if int, it means the pixels.
        '''
        assert pieces >= 1, 'pieces must be greater than or equal to 1'
        if method == 'horizontal':
            max_len = self.size[0]
        elif method == 'vertical':
            max_len = self.size[1]
        else:
            max_len = min(self.size[0], self.size[1])
        assert pieces <= max_len, f'too many pieces: {pieces} > max_len({max_len})'
        if pieces == 1:
            return [self.copy()]
        
        if method == 'square' and pieces!=1:
            if pieces <4:
                _logger.warning(f'When method is `square`, pieces should be square number(1,4,9,16,...). Got {pieces}. Fallback to `horizontal` method.')
                method = 'horizontal'
            elif pieces % 2 !=0:
                _logger.warning(f'When method is `square`, pieces should be even number(4,8,16,...). Got {pieces}. Fallback to `horizontal` method.')
                method = 'horizontal'
        
        def calc_range(r, n)->list[tuple[float, float]]:
            x = 2 / (2*n-((n-1)*r))
            xo = x * r / 2
            ranges = []
            for i in range(n):
                to_l = (i+1)*x - i*xo
                from_l = max(0.0, to_l - x)
                ranges.append((from_l, to_l))
            return ranges
        
        if overlap > 1:
            overlap = overlap / max_len
                
        if method in ('horizontal', 'vertical'):
            r = max(0.0, min(overlap, 1.0))
            ranges = calc_range(r, pieces)
            boxes = []
            for _range in ranges:
                from_l, to_l = _range
                boxes.append(Box2D(
                    left_top=Point2D(0.0, from_l) if method=='horizontal' else Point2D(from_l, 0.0),
                    right_bottom=Point2D(1.0, to_l) if method=='horizontal' else Point2D(to_l, 1.0),
                    mode='relative'
                ))
            return [self.crop(box) for box in boxes]
        elif method == 'square':
            ratio = self.size[0] / self.size[1]
            n_cols = int((pieces * ratio) ** 0.5)
            n_rows = pieces // n_cols
            r = max(0.0, min(overlap, 1.0))
            col_ranges = calc_range(r, n_cols)
            row_ranges = calc_range(r, n_rows)
            boxes = []
            for row_range in row_ranges:
                for col_range in col_ranges:
                    from_lx, to_lx = col_range
                    from_ly, to_ly = row_range
                    boxes.append(Box2D(
                        left_top=Point2D(from_lx, from_ly),
                        right_bottom=Point2D(to_lx, to_ly),
                        mode='relative'
                    ))
            return [self.crop(box) for box in boxes]
        else:
            raise ValueError(f'Invalid method: {method}')
    
    @classmethod
    def _Load(cls, img: AcceptableFileSource|PILImage.Image, /)->Self:
        '''load image from file bytes, path or url'''
        if not isinstance(img, cls):
            if not isinstance(img, PILImage.Image):
                if _is_svg(img):    # type: ignore
                    img_bytes = _svg_convert(img, format='png')  # type: ignore
                    img = PILImage.open(BytesIO(img_bytes))   # type: ignore
                else:
                    img = run_any_func(save_get, img)    # type: ignore
                    img = PILImage.open(img)    # type: ignore
                img.load()  # type: ignore
            img = cls.CastPILImage(img)  # type: ignore
        return img  # type: ignore
    
    @classmethod
    async def _ALoad(cls, img: AcceptableFileSource|PILImage.Image, /)->Self:
        '''asynchronously load image from file bytes, path or url'''
        if not isinstance(img, cls):
            if not isinstance(img, PILImage.Image):
                if _is_svg(img):    # type: ignore
                    img_bytes = _svg_convert(img, format='png')  # type: ignore
                    img = PILImage.open(BytesIO(img_bytes))   # type: ignore
                else:
                    img = await save_get(img)   # type: ignore
                    img = PILImage.open(img)    # type: ignore
                img.load()  # type: ignore
            img = cls.CastPILImage(img)  # type: ignore
        return img  # type: ignore
    
    @classmethod
    def Load(cls, source: AcceptableFileSource|Self, /)->Self:
        '''load image from file bytes, path or url.
        NOTE: a deferred loader will be returned, the image will be loaded when first used.'''
        if isinstance(source, PILImage.Image):
            return cls.CastPILImage(source)
        return _DeferImageLoader(source)    # type: ignore
    
    @classmethod
    async def ALoad(cls, source: AcceptableFileSource|Self, /)->Self:
        '''asynchronously load image from file bytes, path or url.
        NOTE: a deferred loader will be returned, the image will be loaded when first used.'''
        if isinstance(source, PILImage.Image):
            return cls.CastPILImage(source)
        return await _DeferImageLoader(source)  # type: ignore
    
    @classmethod
    def New(
        cls, 
        width: int=512,
        height: int=512,
        color: int|tuple[int,int,int]|tuple[int,int,int,int]=(255, 255, 255),
        mode:ImageColorMode|None=None,
    )->Self:
        '''
        Create a new image with given width, height and color.
        Args:
            - width: the width of the image.
            - height: the height of the image.
            - color: the background color of the image. It can be an integer (grayscale) or a tuple of 3(RGB) or 4(RGBA) integers.
            - mode: the color mode of the image. If None, it will be 'RGB' for 3-channel color and 'RGBA' for 4-channel color.
        '''
        if not isinstance(color, int):
            assert len(color) in (3, 4), 'color must be a tuple of 3(RGB) or 4(RGBA) integers'
        elif isinstance(color, int):
            color = (color, color, color)
        if mode:
            mode = _tidy_color_mode(mode)                   # type: ignore
        else:
            mode = 'RGB' if len(color) == 3 else 'RGBA'     # type: ignore
        img = PILImage.new(mode, (width, height), color)    # type: ignore
        return cls.CastPILImage(img)
            
    @classmethod
    def CastPILImage(cls, img: PILImage.Image)->Self:   
        '''change origin PIL Image type to this Image type'''
        if isinstance(img, cls):
            return img
        setattr(img, '__class__', cls)
        return cast(cls, img)    # type: ignore

    def __repr__(self):
        return f'<{self.__class__.__name__} shape={self.size[0]}x{self.size[1]} mode={self.mode}>'

    if not TYPE_CHECKING:
        def __getattribute__(self, name: str):
            attr = super().__getattribute__(name)
            if not (name.startswith('__') and name.endswith('__')):
                if not isinstance(attr, type) and callable(attr):
                    return _ImgRetWrapper(attr)
            return attr

class _ImgRetWrapper:
    def __init__(self, f):
        self.f = f
        if hasattr(self.f, '__doc__'):
            self.__doc__ = self.f.__doc__
        
    def __getattr__(self, name: str):
        return getattr(self.f, name)
    
    def __is_async_func__(self)->bool:
        # `for `is_async_callable` to work
        return is_async_callable(self.f)
    
    @staticmethod
    def _recursive_cast_image(r):
        if isinstance(r, PILImage.Image) and not isinstance(r, Image):
            return Image.CastPILImage(r)
        elif isinstance(r, (list, tuple, set)):
            return type(r)(_ImgRetWrapper._recursive_cast_image(i) for i in r)
        elif isinstance(r, dict):
            return type(r)({k: _ImgRetWrapper._recursive_cast_image(v) for k, v in r.items()})
        else:
            return r
    
    def __call__(self, *args, **kwargs):
        r = self.f(*args, **kwargs)
        if isinstance(r, Coroutine):
            async def wrapper():
                coro_r = await r
                return _ImgRetWrapper._recursive_cast_image(coro_r)
            return wrapper()
        else:
            r = _ImgRetWrapper._recursive_cast_image(r)
            return r

_no_need_init_image_attrs = ('__image_source__', '__real_image__', '__dict__', '__weakref__', '__module__', '__doc__',
                             '__dir__', '__getattribute__', '__setattr__', '__init__', '__annotations__',
                             '__class__', '__getattr__', '__get_pydantic_core_schema__', 
                             '__get_pydantic_json_schema__', '_defer_load_image')

_image_dir = set(dir(Image))
_image_dir.update(PILImage.Image.__annotations__.keys())

class _DeferImageLoader:
    __image_source__: AcceptableFileSource
    __real_image__: Image|None = None
    
    def __init__(self, source: AcceptableFileSource):
        if isinstance(source, _DeferImageLoader):
            self.__real_image__ = source.__real_image__
            self.__image_source__ = source.__image_source__
        else:
            self.__image_source__ = source
    
    def _defer_load_image(self)->Image:        
        if not self.__real_image__:
            self.__real_image__ = Image._Load(self.__image_source__)
        return self.__real_image__
    
    def __getattr__(self, name):
        if name not in _no_need_init_image_attrs:
            image = self.__real_image__
            if not image and name in _image_dir:
                image = self._defer_load_image()
            if image:
                return getattr(image, name)
        raise AttributeError(f"'{Image.__name__}' object has no attribute '{name}'")  # type: ignore
    
    @classmethod
    def __get_pydantic_json_schema__(cls, cs, handler):
        return Image.__get_pydantic_json_schema__(cs, handler)
    
    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        def validator(data):
            raise ValueError('DeferImageLoader cannot be directly validated. It can only be used as a placeholder for deferred loading of Image.')
        
        def serializer(image: '_DeferImageLoader'):
            if image.__real_image__:
                return _dump_media_dict(image.__real_image__.to_base64(), Image)
            elif isinstance(image.__image_source__, (str, Path)):
                if isinstance(image.__image_source__, Path):
                    source = str(image.__image_source__)
                else:
                    source = image.__image_source__
                return _dump_media_dict(source, Image)
            else:
                image_obj = image._defer_load_image()
                return _dump_media_dict(image_obj.to_base64(), Image)
            
        validate_schema = core_schema.no_info_after_validator_function(validator, core_schema.any_schema())
        serialize_schema = core_schema.plain_serializer_function_ser_schema(serializer)
        return core_schema.json_or_python_schema(
            json_schema=validate_schema,
            python_schema=validate_schema,
            serialization=serialize_schema
        )        
    
__all__ = ['Image', 'CommonImgFormat', 'ImageColorMode']


if __name__ == '__main__':
    def test_load_svg():
        svg = '''<svg height="100" width="100">
        <circle r="45" cx="50" cy="50" stroke="green" stroke-width="3" fill="red" />
        </svg>'''
        img = Image.Load(svg)
        print(img.size, img.mode)
        
    def test():
        from pydantic import BaseModel
        class A(BaseModel):
            img: Image
        
        img_url = 'https://api.thinkthinksyn.com/resources/tts/logo512.png'
        
        img = Image.Load(img_url)
        print(isinstance(img, _DeferImageLoader))   # True
        print(isinstance(img, Image))               # True
        
        a = A(img=img)
        
        print(len(str(a.model_dump())))      # in this moment, img is not loaded yet, will dump as url
        print(img.size, img.mode)                   # this will trigger loading
        print(len(str(a.model_dump())))      # now img is loaded, so the dump is different
    
    test_load_svg()
    test()