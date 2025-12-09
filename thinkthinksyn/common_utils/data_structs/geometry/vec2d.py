import math
import numpy as np  

from .point2d import Point2D

class Vec2D(Point2D):
    '''2D vector'''
    
    @property
    def length(self)->float:
        return (self.x**2+self.y**2)**0.5
    
    @property
    def angle(self)->float:
        '''angle in radian, from x axis to the vector.'''
        return math.atan2(self.y, self.x)
    
    @property
    def normalized(self)->"Vec2D":
        length = self.length
        return Vec2D((self.x/length, self.y/length))
    
    @property
    def slope(self)->float|None:
        if self.x==0:
            return None
        return self.y/self.x
   
    @property
    def np_array(self)->np.ndarray:
        return np.array([self.x, self.y])         


__all__ = ['Vec2D']