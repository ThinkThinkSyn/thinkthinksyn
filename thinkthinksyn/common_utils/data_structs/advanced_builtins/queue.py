from typing import Iterable, Generator, overload, Self, Protocol, Any, TypeVar, Generic

class _Comparable(Protocol):
    """Comparable protocol, for types that can be compared."""
    def __lt__(self, __other: Any) -> bool: ...
    def __eq__(self, __other: Any) -> bool: ...

def _insert(lst: list[tuple], item: tuple, start: int=0, end: int=-1):
    '''binary insert'''
    if end == -1:
        end = len(lst)
    if start == end:
        lst.insert(start, item)
        return
    mid = (start + end) // 2
    if item[0] < lst[mid][0]:
        _insert(lst, item, start, mid)
    else:
        _insert(lst, item, mid+1, end)

def _find_order(lst: list[tuple], order: "_Comparable", start: int=0, end: int=-1):
    '''binary search for the order in the list'''
    if end == -1:
        end = len(lst)
    if start >= end:
        return -1
    mid = (start + end) // 2
    if lst[mid][0] == order:
        return mid
    elif lst[mid][0] < order:
        return _find_order(lst, order, mid+1, end)
    else:
        return _find_order(lst, order, start, mid)

def _find(lst: list[tuple], obj):
    for i, (_, item) in enumerate(lst):
        if item == obj:
            return i
    return -1

_T = TypeVar('_T')
_OT = TypeVar('_OT', bound=_Comparable)

class AutoSortedQueue(Generic[_T, _OT]):
    '''
    AutoSortedQueue is a priority queue that automatically sorts the elements
    based on the priority. You can push elements with or without priority,
    in that case, the priority will be the element itself..
    
    Args:
        - objs: an iterable of objects to be added to the queue.
        - orders: an iterable of priorities corresponding to the objects.
    '''
    
    elements: list[tuple[_OT, _T]]
    '''Elements in the queue, each element is a tuple of (priority, item)'''
    
    @overload
    def __init__(self):...
    @overload
    def __init__(self, objs: Iterable[_T|_OT]):...
    @overload
    def __init__(self, objs: Iterable[_T], orders: Iterable[_OT]):...
    
    def __init__(   # type: ignore
        self, 
        objs: Iterable[_T]|None=None, 
        orders: Iterable[_OT]|None=None
    ):
        self.elements = []
        if objs:
            if orders is not None:
                for obj, order in zip(objs, orders):
                    self.push(obj, order)
            else:
                for obj in objs:
                    self.push(obj)
        elif orders:
            raise ValueError('If orders is provided, objs must also be provided.')

    def empty(self)->bool:
        '''check if the queue is empty'''
        return len(self.elements) == 0

    def push(self, item: _T, priority: _OT|None=None)->None:
        '''
        Push an item with a priority. If priority is None, the item will be
        the priority itself.
        '''
        if priority is None:
            priority = item     # type: ignore
        _insert(self.elements, (priority, item))

    def pop(self)->_T:
        '''pop the item with the smallest priority,
        i.e. pop from left'''
        return self.elements.pop(0)[1]
    
    def rpop(self)->_T:
        '''pop the item with the largest priority,
        i.e. from right'''
        return self.elements.pop()[1]
    
    def clear(self)->None:
        '''clear the queue'''
        self.elements.clear()
        
    def items(self)->Generator[tuple[_OT, _T], None, None]:
        '''return the items in the queue'''
        def gen():
            for element in self.elements:
                yield element
        return gen()
    
    def remove(self, item: _T)->int|None:
        '''remove the item from the queue.
        Return the origin index of the item if found, otherwise None.'''
        idx = _find(self.elements, item)
        if idx != -1:
            self.elements.pop(idx)
            return idx
        return None
    
    def find_order(self, order: _OT)->tuple[int, _T]|None:
        '''
        Find the index of the first item with the specified order.
        Returns:
            - the index of the item
            - 
        '''
        idx = _find_order(self.elements, order)
        if idx == -1:
            return None
        for i in range(idx, 0, -1):
            if self.elements[i][0] != order:
                break
            idx = i
        return idx, self.elements[idx][1] # type: ignore    
        
    def remove_order(self, order: _OT, max_remove: int|None=None)->int:
        '''
        Removing items in the queue which having the specified order.
        if `remove_order` is None, it will remove all items with the same order. 
        
        Return the number of items removed.
        '''
        if max_remove is not None and max_remove < 1:
            return 0
        if r := self.find_order(order):
            idx, _ = r
            count = 0
            prev = self.elements[:idx]
            for o, _ in self.elements[idx:]:
                if o == order:
                    count += 1
                    if max_remove is not None and count >= max_remove:
                        break
                else:
                    break
            new = prev + self.elements[idx+count:]
            self.elements = new
            return count
        else:
            return 0
    
    def to_list(self)->list[_T]:
        '''return the items in the queue as a list'''
        return [element[1] for element in self.elements]
    
    def __len__(self)->int:
        return len(self.elements)
    
    def __contains__(self, item)->bool:
        idx = _find(self.elements, item)
        return idx != -1
    
    def __iter__(self)-> Generator[_T, None, None]:
        for element in self.elements:
            yield element[1]
    
    @overload
    def __getitem__(self, index: int) -> _T: ...
    @overload
    def __getitem__(self, index: slice)->Self:...
    
    def __getitem__(self, index: slice|int):
        if isinstance(index, slice):
            elements = self.elements[index]
            orders, items = zip(*elements)
            return self.__class__(items, orders)
        return self.elements[index][1]
    
    def __str__(self):
        return str([element[1] for element in self.elements])
    
    def __repr__(self):
        return f'PriorityQueue({[element[1] for element in self.elements]})'
    
    def __bool__(self):
        return bool(self.elements)
    
    def __eq__(self, other)->bool:
        if not isinstance(other, self.__class__):
            return self.elements == other
        return self.elements == other.elements
    
    def __ne__(self, other):
        if not isinstance(other, self.__class__):
            return self.elements != other
        return self.elements != other.elements
    
    def __iadd__(self, other):
        if not isinstance(other, self.__class__):
            for element in other:
                self.push(element)
        else:
            for order, element in other.elements:
                self.push(element, order)
        return self

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            # will return a list if other is NOT a AutoSortedQueue
            return list(self) + other    
        else:
            # will return a AutoSortedQueue if other is a AutoSortedQueue
            return self.__class__(list(self) + other.elements)      # type: ignore  


__all__ = ['AutoSortedQueue']


if __name__ == '__main__':
    q = AutoSortedQueue([4,1,2])
    
    q.push(3)
    print(q)
    print(3 in q, 5 in q)
    lst = q+[1,2]
    print(lst, type(lst))
    q += [5,6]
    print(q)
    
    q2 = AutoSortedQueue(['o4', 'o1', 'o-2', 'o2'], [4,1, -2, 2])
    print(q2, list(q2.items()))
    print(q2.find_order(2))
    print(q2.remove_order(2))
    print(q2)