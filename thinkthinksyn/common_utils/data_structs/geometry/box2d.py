from typing import Literal, TYPE_CHECKING, Self, overload
from pydantic import Field, AliasChoices, model_validator, BaseModel

from .point2d import Point2D


class Box2D(BaseModel):
    '''
    Box2d is a BaseModel class for parsing a box area from input.
    This is different from Rect2D, whom is not a BaseModel class(only a tuple with 4 numbers). 
    Thus, this class is more suitable for validation and serialization.
     
    Note: If both left_top & right_bottom is provided within [0, 1] and `mode` is not given,
          it will be set to 'relative' automatically.
    '''

    left_top: Point2D = Field(default=Point2D(0.0,0.0),  
                              validation_alias=AliasChoices('left_top', 'top_left', 'lefttop', 'leftop', 'topleft',
                                                            'start', 'begin', 'left_corner', 'from'))
    '''The left-top point of the box. '''
    right_bottom: Point2D = Field(default=Point2D(1.0,1.0), 
                                  validation_alias=AliasChoices('right_bottom', 'bottom_right', 'rightbottom', 'bottomright',
                                                                'end', 'right_corner', 'to'))
    '''The right-bottom point of the box. '''
    
    if TYPE_CHECKING:
        mode: Literal['absolute', 'relative'] = 'absolute'
        '''
        Mode of the box. 
            - Absolute: means positions are absolute coordinates.
            - Relative: means positions([0,1]) & size([0,1]) are relative to the parent, e.g. image
        
        Note: If both left_top & right_bottom is provided within [0, 1] and `mode` is not given,
            it will be set to 'relative' automatically.
        '''
    else:
        mode: Literal['absolute', 'relative']|None = None
        '''
        Mode of the box. 
            - Absolute: means positions are absolute coordinates.
            - Relative: means positions([0,1]) & size([0,1]) are relative to the parent, e.g. image
        
        Note: If both left_top & right_bottom is provided within [0, 1] and `mode` is not given,
            it will be set to 'relative' automatically.
        '''
        
    @model_validator(mode='before')
    @classmethod
    def _PreValidator(cls, data):
        if isinstance(data, (list, tuple)):
            if len(data) == 4:  # [x1, y1, x2, y2]
                return {
                    'left_top': Point2D(data[0], data[1]),
                    'right_bottom': Point2D(data[2], data[3])
                }
            elif len(data) == 2 and \
                isinstance(data[0], (list, tuple)) and len(data[0])==2 and \
                    isinstance(data[1], (list, tuple)) and len(data[1])==2:    # [[x1, y1], [x2, y2]]
                return {
                    'left_top': Point2D(data[0][0], data[0][1]),
                    'right_bottom': Point2D(data[1][0], data[1][1])
                }
            else:
                raise ValueError('Invalid input data for Box2D')
        return data
    
    @model_validator(mode='after')
    def _post_validate(self, info):
        start_x = max(self.left_top.x, 0)
        start_y = max(self.left_top.y, 0)
        self.left_top = Point2D(start_x, start_y)
        
        end_x = max(self.right_bottom.x, 0)
        end_y = max(self.right_bottom.y, 0)
        self.right_bottom = Point2D(end_x, end_y)
        
        if (0.0, 0.0) <= self.left_top <= (1.0, 1.0) and \
            (0.0, 0.0) <= self.right_bottom <= (1.0, 1.0) and \
               self.mode is None:
            self.mode = 'relative'
        if self.mode is None:
            self.mode = 'absolute' 
        return self
    
    @property
    def width(self)->float:
        return self.right_bottom.x - self.left_top.x
    
    @property
    def height(self)->float:
        return self.right_bottom.y - self.left_top.y
    
    @overload
    def to_absolute(self, width: float|int, height: float|int)->Self:...
    @overload
    def to_absolute(self, size: tuple[int|float, int|float]|list[float|int])->Self:...

    def to_absolute(self, *args, **kwargs):
        '''convert the relative box to absolute box'''
        arg_count = len(args) + len(kwargs)
        assert arg_count <= 2, 'Arguments must be 2 at most'
        if arg_count == 1:
            size = args[0] if len(args) == 1 else kwargs.get('size')
            assert size is not None and isinstance(size, (tuple, list)), 'Size must be a tuple or list'
            assert len(size) == 2, 'Size must be a tuple or list with 2 elements'
            return self.to_absolute(size[0], size[1])
        else:
            width = kwargs.get('width', args[0])
            height = kwargs.get('height', args[1])
            return self.model_copy(update={
                'left_top': Point2D(self.left_top.x*width, self.left_top.y*height),
                'right_bottom': Point2D(self.right_bottom.x*width, self.right_bottom.y*height),
                'mode': 'absolute'
            })
            
    
    
    
__all__ = ['Box2D']