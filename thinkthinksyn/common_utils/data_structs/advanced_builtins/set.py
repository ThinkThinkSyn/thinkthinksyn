# -*- coding: utf-8 -*-
'''Collections of some special data structures.'''
from typing import Callable, Any, TypeAlias, TypeVar, Generic
from functools import partial

def _fuzzy_simplify(s: str):
    return s.strip().lower().replace(' ', '').replace('_', '').replace('-', '')

_T = TypeVar('_T')

class ContainerSet(Generic[_T], set[_T]):
    '''
    Special set which could be easily to customize the __contain__ method.
    Helpful in define special checking
    '''
    def __init__(
        self, 
        *args, 
        validator:Callable[[set[_T], _T], bool]|None=None, 
        iter_validator:Callable[[_T, _T], bool]|None=None, 
        **kwargs
    ): 
        '''
        Args:
            - validator: the function to check if the item is in the set.
            - iter_validator: if validator returns False, then use this function to check if the item is in the set one-by-one.
        '''
        super().__init__(*args, **kwargs)
        self.validator = validator
        self.iter_validator = iter_validator
        
    def contains(self, item:Any) -> bool:
        if self.validator or self.iter_validator:
            if self.validator:
                result = self.validator(self, item)
            else:
                result = False
            if not result:
                if self.iter_validator:
                    for self_item in self:
                        if self.iter_validator(self_item, item):
                            return True
            return result
        return item in self
    
    def __contains__(self, item) -> bool:
        return self.contains(item)

_CaseIgnoreStrSet: type[ContainerSet] = partial(ContainerSet, iter_validator=lambda item, target: item.lower() == target.lower())   # type: ignore
CaseIgnoreStrSet: TypeAlias = ContainerSet[str]  # just for type hint
'''
Check if item.lower() == target.lower()
E.g.
```
string_set = CaseIgnoreStrSet(['a', 'b'])
print(string_set.contains('A'))   # True
```
'''
globals()['CaseIgnoreStrSet'] = _CaseIgnoreStrSet

_RangeSet:ContainerSet = partial(ContainerSet, iter_validator=lambda item, target: target in item)   # type: ignore
RangeSet: TypeAlias = ContainerSet
'''
Items should be range. Check if an item in any range.
E.g.:
```
range_set = RangeSet([range(1, 3), range(3, 5)])
print(range_set.contains(2))    # True
```
'''
globals()['RangeSet'] = _RangeSet

class FuzzySet(set[str]):
    '''
    A special string set which will simplify the input string before adding into it.
    Underscores, spaces and case will be ignored, e.g. `Hello World` == `hello_world` == `helloworld`.
    '''
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cur_items = tuple(self)
        self.clear()
        for item in cur_items:
            self.add(item)

    def add(self, item:str):
        simplified = _fuzzy_simplify(item)
        set.add(self, simplified)

    def remove(self, item:str):
        simplified = _fuzzy_simplify(item)
        set.remove(self, simplified)
        
    def discard(self, item:str):
        simplified = _fuzzy_simplify(item)
        set.discard(self, simplified)
        
    def __contains__(self, item:str):
        simplified = _fuzzy_simplify(item)
        return set.__contains__(self, simplified)



__all__ = [
    'ContainerSet', 
    'CaseIgnoreStrSet', 
    'RangeSet', 
    'FuzzySet'
]