import re
import inspect
import logging
import numpy as np

from attr import attrs
from functools import partial
from pydantic import BaseModel as BaseModelV2
from pydantic.v1 import BaseModel as BaseModelV1
from sympy import (And, Or, simplify_logic, Expr, Eq, Ne, symbols, false as _False, true as _True, 
                   solve, sstr)
from typing import (Any, Self, Mapping, Iterable, Set, no_type_check, TYPE_CHECKING,
                    Final, overload, get_args)
from typing_extensions import override

from .parser import (BaseConditionOperator, ConditionOperator, _inverse_base_operator, _operator_map, 
                     _parse_base_condition, _valid_base_ops, _parse_condition)
from ._utils import hash_md5

_logger = logging.getLogger(__name__)
_inverse_operator = {
    'and': 'or',
    'or': 'and',
    '&': 'or',
    '|': 'and',
    '&&': 'or',
    '||': 'and',
}
_operators_with_not_prefix = set([f'not {k}' for k in _inverse_base_operator.keys() if k not in ('not in', 'in', 'not contains', 'contains')])
_fuzzy_simplify_str = lambda x: x.replace('_', '').lower().strip()
_empty = inspect.Parameter.empty
_NullVals = (None, 'null', 'None', 'none', 'NoneType', 'NULL', 'Null', _empty, 'nan', 'NaN', float('nan'))
_LooseNullVals = _NullVals + ('', 0, False, Ellipsis)

def _get_fields(obj: object, loose: bool=True)->tuple[dict[str, str], int]|tuple[None, None]:
    try:
        if isinstance(obj, dict):
            all_fuzzy_keys = FuzzySet(obj.keys())
            if len(all_fuzzy_keys) != len(obj) or not loose:
                d = {}
            else:
                d = FuzzyDict()
            for k in obj:
                d[k] = k
            return d, len(d)
        
        d = {} if not loose else FuzzyDict()
        if isinstance(obj, (BaseModelV1, BaseModelV2)):
            count = 0
            fields = obj.__fields__ if isinstance(obj, BaseModelV1) else obj.model_fields
            for f in fields:
                count += 1
                aliases = get_pydantic_model_field_aliases(obj, f)
                for a in aliases:
                    d[a] = f
            return d, count
        if hasattr(obj, '__dataclass_fields__'):
            for f in obj.__dataclass_fields__:   # type: ignore
                d[f] = f
            return d, len(d)    # type: ignore
        if hasattr(obj, '__attrs_attrs__'):
            for f in obj.__attrs_attrs__:   # type: ignore
                d[f.name] = f.name
            return d, len(d) # type: ignore
    except:
        ...
    return None, None

def _tidy_str_for_compare(s: str, fuzzy):
    if fuzzy:
        s = s.lower().strip()
        if s.startswith('`') and s.endswith('`'):
            s = s[1:-1]
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    return s

def _tidy_str(s):
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")) or\
            (s.startswith('`') and s.endswith('`')):
        return s[1:-1]
    return s

def _tidy_num(s: str):
    s = _tidy_str(s.strip()).strip()
    return get_num_from_text(s) if s else None

def _equal(x, y, loose_compare: bool = True):
    if isinstance(x, (bytes, bytearray)):
        x = x.decode()
    if isinstance(y, (bytes, bytearray)):
        y = y.decode()
    if isinstance(x, str):
        x = _tidy_str_for_compare(x, loose_compare)
    if isinstance(y, str):
        y = _tidy_str_for_compare(y, loose_compare)
    try:
        if x == y:
            return True
    except:
        ...
    
    # compare with null values
    null_vals = _LooseNullVals if loose_compare else _NullVals
    x_in_null = x in null_vals
    y_in_null = y in null_vals
    if not loose_compare:
        if x_in_null and y_in_null:
            return True
        if x_in_null and y not in null_vals:
            return False
    else:
        if x_in_null:
            if not y:
                return True
            return False
        if y_in_null:
            if not x:
                return True
            return False
    
    modified_x, modified_y = False, False
    # try to deserialize the value
    if isinstance(x, str):
        maybe_x = _tidy_num(x)
        if maybe_x is not None:
            x = maybe_x
            modified_x = True
        else:
            try:
                x = deserialize(x)
                modified_x = True
            except:
                ...
    if isinstance(y, str):
        maybe_y = _tidy_num(y)
        if maybe_y is not None:
            y = maybe_y
            modified_y = True
        else:
            try:
                y = deserialize(y)
                modified_y = True
            except:
                ...
    if modified_x or modified_y:
        try:
            if x == y:
                return True
        except:
            ...
    elif isinstance(x, str) or isinstance(y, str):
        return False # impossible to be equal if any str not modified
    
    # compare with length
    if hasattr(x, '__len__') and hasattr(y, '__len__'):
        if len(x) != len(y):    # type: ignore
            return False
    
    # check for set & list
    if isinstance(x, Set) or isinstance(y, Set):
        if not isinstance(x, Set):
            try:
                x = set(x)  # type: ignore
            except:
                return False
        if not isinstance(y, Set):
            try:
                y = set(y)  # type: ignore
            except:
                return False
        r = (x == y)
        if r or not loose_compare:
            return r    # if loose_compare, keep going in list
    
    if isinstance(x, Iterable) and not isinstance(x, str):
        x = list(x)
    if isinstance(y, Iterable) and not isinstance(y, str):
        y = list(y)
    if isinstance(x, list) and isinstance(y, list):
        if loose_compare:
            y_remains = y.copy()
            for x_v in x:
                for y_v in y_remains:
                    if _equal(x_v, y_v, loose_compare=loose_compare):
                        y_remains.remove(y_v)
                        break
                else:
                    return False
            return True    
        else:
            for i, v in enumerate(x):
                try:
                    if not _equal(v, y[i]):
                        return False
                except IndexError:
                    return False
            return True

    # final compare: check dict/data model classes
    x_fields, x_field_count = _get_fields(x, loose=loose_compare)
    y_fields, y_field_count = _get_fields(y, loose=loose_compare)
    if x_fields is None and y_fields is None:
        return False
    else:
        if x_field_count == 0 and y_field_count == 0:
            return True # 2 empty data classes, always equal
        if (x_field_count == 0 and y_field_count) or (y_field_count == 0 and x_field_count):
            return False # 1 empty data class, 1 not empty
        if not loose_compare and (x_field_count is not None and y_field_count is not None):
            if x_field_count != y_field_count:
                return False
        
        if x_fields is not None and (y_fields or isinstance(y, Mapping)):
            try:
                x = recursive_dump_to_basic_types(x, ignore_err=loose_compare, include_non_dumpable=not loose_compare)
            except: # fail to dump, try compare directly
                ...
        if y_fields is not None and (x_fields or isinstance(x, Mapping)):
            try:
                y = recursive_dump_to_basic_types(y, ignore_err=loose_compare, include_non_dumpable=not loose_compare)
            except: # fail to dump, try compare directly
                ...
        
        # has model fields, but final dump is empty. invalid for comparison
        if x_field_count and isinstance(x, dict) and not x:
            return False
        if y_field_count and isinstance(y, dict) and not y:
            return False
        
        get_obj_attr = lambda obj, k: obj.get(k, _empty) if isinstance(obj, dict) else getattr(obj, k, _empty)
        
        if (x_fields or isinstance(x, Mapping)) and (y_fields or isinstance(y, Mapping)):
            real_x_field_count = len(x) if isinstance(x, Mapping) else x_field_count
            real_y_field_count = len(y) if isinstance(y, Mapping) else y_field_count
            if real_x_field_count != real_y_field_count:
                return False
            if isinstance(x, BaseModelV1):
                x_keys = list(x.__fields__.keys())  # type: ignore
            elif isinstance(x, BaseModelV2):
                x_keys = list(x.model_fields.keys())    # type: ignore
            elif hasattr(x, '__dataclass_fields__'):
                x_keys = list(x.__dataclass_fields__.keys())    # type: ignore
            elif hasattr(x, '__attrs_attrs__'):
                x_keys = [f.name for f in x.__attrs_attrs__]   # type: ignore
            elif isinstance(x, dict):
                x_keys = list(x.keys())
            else:
                return False
            for k in x_keys:
                y_key = y_fields.get(k, _empty) if y_fields else k
                if y_key is _empty:
                    return False
                x_attr = get_obj_attr(x, k)
                y_attr = get_obj_attr(y, y_key)
                if x_attr is _empty or y_attr is _empty:
                    return False
                if not _equal(x_attr, y_attr, loose_compare=loose_compare):
                    return False
            return True
        return False
    
def _contains(container, value, loose_compare: bool = True):
    if hasattr(container, '__contains__'):
        try:
            return container.__contains__(value)
        except:
            ...
    if isinstance(container, Iterable):
        for v in container:
            if _equal(v, value, loose_compare=loose_compare):
                return True
    return False

def _operator_wrapper(operator):
    def _is_num(obj):
        if isinstance(obj, (int, float)):
            return True
        if isinstance(obj, np.number):
            return True
        return False
    
    def wrapper(x, y):
        if (isinstance(x, str) and not isinstance(y, str)) or \
                (isinstance(y, str) and not isinstance(x, str)):
            if isinstance(x, str):
                if isinstance(y, bytes):
                    try:
                        y = y.decode()
                    except:
                        ...
                else:
                    try:
                        x = deserialize(x)
                    except:
                        ...
            if isinstance(y, str):
                if isinstance(x, bytes):
                    try:
                        x = x.decode()
                    except:
                        ...
                else:
                    try:
                        y = deserialize(y)
                    except:
                        ...
        
        if (isinstance(x, str) and _is_num(y)) or \
                (isinstance(y, str) and _is_num(x)):
            if isinstance(x, str):
                if (maybe_x:=_tidy_num(x)) is not None:
                    x = maybe_x
            elif isinstance(y, str):
                if (maybe_y:=_tidy_num(y)) is not None:
                    y = maybe_y
        
        elif isinstance(x, str) and isinstance(y, str):
            maybe_x, maybe_y = _tidy_num(x), _tidy_num(y)
            if maybe_x is not None and maybe_y is not None:
                x, y = maybe_x, maybe_y
            else:
                try:
                    x = deserialize(x)
                except:
                    ...
                try:
                    y = deserialize(y)
                except:
                    ...
        
        if isinstance(x, Iterable) and not isinstance(x, str):
            x = list(x)
        if isinstance(y, Iterable) and not isinstance(y, str):
            y = list(y)
        
        return operator(x, y)
    return wrapper

def _is_num(obj):
    return isinstance(obj, (int, float)) or isinstance(obj, np.number)

def _simply_expr_str(expr):
    s = sstr(expr)
    s = re.sub(r'Eq\((.*?), (.*?)\)', r'\1 = \2', s)
    s = re.sub(r'Ne\((.*?), (.*?)\)', r'\1 != \2', s)
    s = re.sub(r'Ge\((.*?), (.*?)\)', r'\1 >= \2', s)
    s = re.sub(r'Le\((.*?), (.*?)\)', r'\1 <= \2', s)
    s = re.sub(r'Gt\((.*?), (.*?)\)', r'\1 > \2', s)
    s = re.sub(r'Lt\((.*?), (.*?)\)', r'\1 < \2', s)
    s = re.sub(r'And\((.*?), (.*?)\)', r'(\1) and (\2)', s)
    s = re.sub(r'Or\((.*?), (.*?)\)', r'(\1) or (\2)', s)
    s = re.sub(r'Not\((.*?)\)', r'not (\1)', s)
    s = re.sub(r'\{(.*?): (.*?)\}', r'\1 = \2', s)
    return s

left_val_pattern = r'\(?\s*(?P<v>[\d\.]+)\s*(?P<op>\<\=?|\>\=?|\=|\!\=)\s*x\)?'
left_val_pattern = re.compile(left_val_pattern)
right_val_pattern = r'\(?\s*x\s*(?P<op>\<\=?|\>\=?|\=|\!\=)\s*(?P<v>[\d\.]+)\)?'
right_val_pattern = re.compile(right_val_pattern)
_inverse_compare_op = {'<':'>', '<=':'>=', '>':'<', '>=':'<=',}

def _parse_expr_str(expr: str):
    chunks: list[tuple[str, float|int]] = []    # [(operator, value), ...]
    
    def match(expr):
        m = re.search(left_val_pattern, expr)
        p = 'left'
        if not m:
            m = re.search(right_val_pattern, expr)
            p = 'right'
        if m:
            return m, p
        return None

    while (matched:= match(expr)) is not None:
        m, p = matched
        op = m.group('op')
        if p == 'left':
            op = _inverse_compare_op.get(op, op)
        v = eval(m.group('v'))
        chunks.append((op, v))  # type: ignore
        expr = expr[m.end():]
    return chunks

class _ConditionBase:
    @classmethod
    @no_type_check  
    def Not(cls, cond: Self)->Self:
        '''Return a new condition which is the inverse of the given condition.'''
        while isinstance(cond, Condition) and (cond.operator is None or cond.right is None):
            cond = cond.left
        if isinstance(cond, bool):
            return TrueCondition if not cond else FalseCondition
        elif isinstance(cond, Condition):
            return Condition(
                left=cond.left.not_(), 
                operator=_inverse_operator[cond.operator], 
                right=cond.right.not_()
            )   # case of only left condition has been treated in `while` statement
        elif isinstance(cond, BaseCondition):
            base = BaseCondition(
                attr=cond.attr, 
                operator=_inverse_base_operator[cond.operator], 
                value=cond.value
            )
            if issubclass(cls, Condition):
                return Condition(left=base)
            return base
        raise ValueError(f'Condition should be a string, Condition or BaseCondition, but got {type(cond)}')
        
    def not_(self)->Self:
        '''Return a new condition which is the inverse of the current condition.'''
        return self.Not(self)
    
    def __invert__(self)->Self:
        '''~ operator is equivalent to `not_()` method.'''
        return self.not_()
    
    def __pydantic_serialize__(self):
        # for `serializable_attrs`
        return str(self)

@attrs(eq=False)
class BaseCondition(_ConditionBase):
    '''
    Base condition class which contains 1 attribute and 1 operator.
    This is used for checking whether an object satisfies the condition,
    e.g. `obj.attr > 1`.
    
    This is for some query liked use cases, e.g. selecting nodes in
    service system.
    
    This class is serializable and can be used in pydantic models. 
    '''
    
    attr: str
    '''
    Attribute name of the object.
    Multi-layer attribute name is supported, e.g. `a.b.c`.
    '''
    operator: BaseConditionOperator
    '''
    The operator for selection. It can be one of the following:
        - `=`: equal
        - `!=`: not equal
        - `>`: greater than
        - `<`: less than
        - `>=`: greater than or equal
        - `<=`: less than or equal
        - `in`
        - `not in`
        - `contains`
        - `not contains`
    NOTE: `==` is equivalent to `=`, but it is not recommended to use it.
    '''
    
    value: Any
    '''The value for operation. It can be a list of values if the operator is `in` or `not in`.
    Note that condition value must be serializable.'''
    
    # region magic methods
    def __setattr__(self, name: str, value: Any):
        if name == 'value':
            curr_val = getattr(self, '__val_serialized__', _empty)
            if value != curr_val:
                self.__val_serialized__ = _empty
        super().__setattr__(name, value)
    
    def __str__(self)->str:
        ser_cache = getattr(self, '__val_serialized__', _empty)
        if ser_cache is _empty:
            if isinstance(self.value, str):
                val_str = self.__val_serialized__ = '"'+ _tidy_str(self.value) + '"'
            elif isinstance(self.value, (list, set, tuple)):
                lst = list(self.value)
                try:
                    lst.sort()
                except:
                    ...
                val_str = self.__val_serialized__ = serialize(lst)
            else:
                val_str = self.__val_serialized__ = serialize(self.value)
        else:
            val_str = ser_cache
        return f'({self.attr} {self.operator} {val_str})'
    
    __repr__ = __str__
    
    def __eq__(self, other)->bool:
        while isinstance(other, Condition) and (other.operator is None or other.right is None):
            other = other.left
        if isinstance(other, BaseCondition) and (self.operator == other.operator):
            if (self.attr == other.attr and self.value == other.value):
                return True # to prevent miss in serialization compare, e.g. `1 == 1.0` return False in string case  
        try:
            return str(self) == str(other)
        except:
            return False
    
    @no_type_check
    def __and__(self, other: 'str|BaseCondition|Condition|bool')->'Condition|BaseCondition':
        while isinstance(other, Condition) and (other.operator is None or other.right is None):
            other = other.left
        if isinstance(other, bool):
            if other:
                return self
            else:
                return FalseCondition
        elif other == self:
            return self
        elif isinstance(other, BaseCondition):
            if other.not_() == self:
                return FalseCondition
            
            if self.attr == other.attr:
                # check wether merge is possible
                o1, o2 = self.operator, other.operator
                def check_pair(v1, v2):
                    return ((o1 == v1 and o2 == v2) or (o1 == v2 and o2 == v1))
                
                numeric_op = ('=', '!=', '>', '<', '>=', '<=')
                if o1 in numeric_op and o2 in numeric_op and _is_num(self.value) and _is_num(other.value):
                    x = symbols('x', real=True)
                    
                    def get_expr(o, v):
                        if o =='=':
                            return Eq(x, v)
                        elif o == '!=':
                            return Ne(x, v)
                        return eval(f'x {o} {v}')
                    
                    expr1 = get_expr(o1, self.value)
                    expr2 = get_expr(o2, other.value)
                    simplified_expr = solve([expr1, expr2], x)
                    chunks = _parse_expr_str(_simply_expr_str(simplified_expr))
                    if len(chunks) == 1:
                        op, v = chunks[0]
                        return BaseCondition(attr=self.attr, operator=op, value=v)
                    else:
                        return Condition(
                            left=BaseCondition(attr=self.attr, operator=chunks[0][0], value=chunks[0][1]),
                            operator='and',
                            right=BaseCondition(attr=self.attr, operator=chunks[1][0], value=chunks[1][1])
                        )
                
                elif self.value == other.value:
                    # no need to check inverse like ('=', '!=') / ('>', '<=') / ..., since they are dont in 
                    # `if other.not_() == self` statement
                    if check_pair('>', '<') or check_pair('>', '=') or check_pair('<', '='):
                        return FalseCondition
                    elif check_pair('!=', '<') or check_pair('!=', '>') or check_pair('!=', '<=') or check_pair('!=', '>='):
                        o = o1 if o1 != '!=' else o2
                        o = o[0] if o.endswith('=') else o 
                        return BaseCondition(attr=self.attr, operator=o, value=self.value)  # type: ignore
                    elif check_pair('>', '>=') or check_pair('<', '<='):
                        o = o1 if len(o1) == 1 else o2
                        return BaseCondition(attr=self.attr, operator=o, value=self.value)  # type: ignore
                    elif check_pair('>=', '<=') or check_pair('=', '>=') or check_pair('=', '<='):
                        return BaseCondition(attr=self.attr, operator='=', value=self.value)  # type: ignore
                
                elif o1 == 'in' and o2 == 'in':
                    v1, v2 = self.value, other.value
                    if isinstance(v1, (list, set, tuple)) and isinstance(v2, (list, set, tuple)):
                        try:
                            v = set(v1) & set(v2)
                        except TypeError: # not hashable
                            if len(v1) > len(v2):
                                longer, shorter = v1, v2
                            else:
                                longer, shorter = v2, v1
                            v = []
                            for item in longer:
                                if item in shorter:
                                    v.append(item)
                        if not v:
                            return FalseCondition    # always true, since no intersection 
                        return BaseCondition(attr=self.attr, operator='in', value=v)  # type: ignore
                    
                elif check_pair('in', 'not in'):
                    in_v = self.value if o1 == 'in' else other.value
                    not_in_v = self.value if o1 == 'not in' else other.value
                    if isinstance(in_v, (list, set, tuple)) and isinstance(not_in_v, (list, set, tuple)):
                        try:
                            in_v = set(in_v)
                            not_in_v = set(not_in_v)
                            if in_v <= not_in_v:
                                return FalseCondition
                            elif in_v > not_in_v:
                                return BaseCondition(attr=self.attr, operator='in', value=in_v - not_in_v)  # type: ignore
                            else:
                                return Condition(
                                    left=BaseCondition(attr=self.attr, operator='in', value=in_v - not_in_v),
                                    operator='and',
                                    right=BaseCondition(attr=self.attr, operator='not in', value=not_in_v - in_v)
                                )
                        except TypeError: # not hashable
                            not_in_v = list(not_in_v)
                            tidied_in = []
                            for item in in_v:
                                idx = not_in_v.index(item)
                                if idx == -1:
                                    tidied_in.append(item)
                                else:
                                    not_in_v.pop(idx)
                            tidied_not_in = not_in_v
                            if not tidied_in:
                                return FalseCondition
                            if not tidied_not_in:
                                return BaseCondition(attr=self.attr, operator='in', value=tidied_in)  # type: ignore
                            return Condition(
                                left=BaseCondition(attr=self.attr, operator='in', value=tidied_in),
                                operator='and',
                                right=BaseCondition(attr=self.attr, operator='not in', value=tidied_not_in)
                            )
                
            return Condition(left=self, operator='and', right=other)  # type: ignore
        elif isinstance(other, Condition):
            # fall to `Condition.__and__()`
            return other & self  # type: ignore
        elif isinstance(other, str):
            return Condition.Parse(other) & self
        raise ValueError(f'Condition should be a string, Condition or BaseCondition, but got {type(other)}')
        
    @no_type_check
    def __or__(self, other: 'str|BaseCondition|Condition|bool')->'Condition|BaseCondition':
        while isinstance(other, Condition) and (other.operator is None or other.right is None):
            other = other.left
        if isinstance(other, bool):
            if other:
                return TrueCondition
            else:
                return self
        elif other == self:
            return self
        elif isinstance(other, BaseCondition):
            if other.not_() == self:
                return TrueCondition
            
            if self.attr == other.attr:
                # check wether merge is possible
                o1, o2 = self.operator, other.operator
                
                def check_pair(v1, v2):
                    return ((o1 == v1 and o2 == v2) or (o1 == v2 and o2 == v1))
                
                if self.value == other.value:
                    # no need to check inverse like ('=', '!=') / ('>', '<=') / ..., since they are dont in 
                    # `if other.not_() == self` statement
                    if check_pair('>=', '<='):
                        return TrueCondition
                    if check_pair('>', '<'):
                        return BaseCondition(attr=self.attr, operator='!=', value=self.value)  # type: ignore
                    elif check_pair('>', '=') or check_pair('<', '=') or check_pair('=', '>=') or check_pair('=', '<='):
                        o = o1 if o1 != '=' else o2
                        o = o+'=' if not o.endswith('=') else o
                        return BaseCondition(attr=self.attr, operator=o, value=self.value)  # type: ignore
                    elif check_pair('!=', '<') or check_pair('!=', '>') or check_pair('!=', '<=') or check_pair('!=', '>='):
                        o = o1 if o1 != '!=' else o2
                        o = o[0] if o.endswith('=') else o 
                        return BaseCondition(attr=self.attr, operator=o, value=self.value)  # type: ignore
                    elif check_pair('>', '>=') or check_pair('<', '<='):
                        o = o1 if len(o1) == 2 else o2
                        return BaseCondition(attr=self.attr, operator=o, value=self.value)
                    
                elif o1 == 'in' and o2 == 'in':
                    v1, v2 = self.value, other.value
                    if isinstance(v1, (list, set, tuple)) and isinstance(v2, (list, set, tuple)):
                        try:
                            v = set(v1) | set(v2)
                        except TypeError: # not hashable
                            v = list(v1)
                            for item in v2:
                                if item not in v:
                                    v.append(item)
                        return BaseCondition(attr=self.attr, operator='in', value=v)
                         
                elif check_pair('in', 'not in'):
                    in_v = self.value if o1 == 'in' else other.value
                    not_in_v = self.value if o1 == 'not in' else other.value
                    if isinstance(in_v, (list, set, tuple)) and isinstance(not_in_v, (list, set, tuple)):
                        try:
                            in_v = set(in_v)
                            not_in_v = set(not_in_v)
                            if inv >= not_in_v:
                                return TrueCondition
                            else:
                                return BaseCondition(attr=self.attr, operator='not in', value=not_in_v - in_v)  # type: ignore
                        except TypeError:   # not hashable
                            diff = []
                            for item in not_in_v:
                                if item not in in_v:
                                    diff.append(item)
                            return BaseCondition(attr=self.attr, operator='not in', value=diff)
            
            return Condition(left=self, operator='or', right=other)
        elif isinstance(other, Condition):
            # fall to `Condition.__or__()`
            return other | self  # type: ignore
        elif isinstance(other, str):
            return Condition.Parse(other) | self
        raise ValueError(f'Condition should be a string, Condition or BaseCondition, but got {type(other)}')
        
    @no_type_check
    def __attrs_post_init__(self):
        # check attribute name
        if not isinstance(self.attr, str):
            raise ValueError(f'Attribute name should be a string: {self.attr}')
        if not self.attr.replace('.', '').isidentifier():
            raise ValueError(f'Invalid attribute name: {self.attr}. It should be a valid identifier(\'.\' is allowed).')
        
        # check operator
        if not isinstance(self.operator, str):
            raise ValueError(f'Operator should be a string: {self.operator}')
        self.operator = self.operator.lower().strip()   # type: ignore
        if self.operator == '==':
            self.operator = '='
        elif self.operator == '!==':
            self.operator = '!='
        elif self.operator in _operators_with_not_prefix:
            self.operator = _inverse_base_operator[self.operator[4:]]
        if self.operator not in _valid_base_ops:
            raise ValueError(f'Operator should be one of {get_args(BaseConditionOperator)}, but got {self.operator}')
        
        # check value
        if not is_serializable(self.value):
            raise ValueError(f'Condition value should be serializable, but got: {self.value}')
        
    @classmethod
    def __pydantic_deserialize__(cls, data):    # for `serializable_attrs`
        if isinstance(data, str):
            return cls.Parse(data)
        elif isinstance(data, (tuple, list)):
            if len(data) == 1:
                if not isinstance(data[0], str):
                    raise ValueError(f'Invalid condition: {data[0]}. It should be a string.')
                return cls.Parse(data[0])
            elif len(data) == 2:
                if not isinstance(data[0], str):
                    raise ValueError(f'Invalid condition attribute name: {data[0]}. It should be a string.')
                return cls(attr=data[0], operator='=', value=data[1])  # type: ignore
            if len(data) != 3:
                raise ValueError(f'Condition tuple should have 3 elements, but got {len(data)}.')
            operator = data[1].lower().strip()
            return cls(attr=data[0], operator=operator, value=data[2])  # type: ignore
        return data
    # endregion
    
    @classmethod
    def Parse(cls, data: str)->Self:
        '''
        Parse condition from string,
        e.g `a.b.c > 1` -> `BaseCondition(attr='a.b.c', operator='>', value=1)`.
        '''
        cond, remain= _parse_base_condition(data)
        sim_remain = remain.replace(' ', '').replace('(', '').replace(')', '').strip()
        if sim_remain:
            raise ValueError(f'Invalid condition string: {data}. Remaining string: {remain.strip()}')
        return cond # type: ignore
        
    def validate(
        self, 
        obj: object, 
        fuzzy: bool = False,
        loose_compare: bool = True,
    )->bool:
        '''
        Validate the condition on the object.
        
        Args:
            obj: The object to validate.
            fuzzy: If True, fuzzy match will be used when getting target attribute
                 from the object.
            loose_compare: If True, loose compare will be used when comparing the value,
                    e.g. `1` == `1.0`, `True` == `1`, '' == `None`, etc.
            
        Returns:
            bool: True if the condition is satisfied, False otherwise.
        '''
        if '.' in self.attr:
            attr_layers = self.attr.split('.')
        else:
            attr_layers = [self.attr]
        
        origin_obj = curr_obj = obj
        _empty = object()
        obj_attr = _empty
        
        for attr_name in attr_layers:
            obj_attr = getattr(curr_obj, attr_name, _empty)   
            if fuzzy and (obj_attr is _empty):
                fuzzy_attr = _fuzzy_simplify_str(attr_name)
                for obj_attr_name in dir(curr_obj):
                    if _fuzzy_simplify_str(obj_attr_name) == fuzzy_attr:
                        obj_attr = getattr(curr_obj, obj_attr_name, _empty)
                        if obj_attr is not _empty:
                            break
            if obj_attr is _empty:
                return False
            else:
                curr_obj = obj_attr
        if obj_attr is _empty:
            return False

        need_not = (self.operator.startswith('not ') or self.operator == '!=') # type: ignore
        if self.operator in ('in', 'not in'):  # type: ignore
            left = self.value
            right = obj_attr
        else:
            left = obj_attr
            right = self.value
        
        # not using eval() for security reason
        if self.operator in ('=', '!='):
            operate = partial(_equal, loose_compare=loose_compare)
        elif self.operator == '>':
            operate = _operator_wrapper(lambda x, y: x > y)
        elif self.operator == '<':
            operate = _operator_wrapper(lambda x, y: x < y)
        elif self.operator == '>=':
            operate = _operator_wrapper(lambda x, y: x >= y)
        elif self.operator == '<=':
            operate = _operator_wrapper(lambda x, y: x <= y)
        elif self.operator in ('in', 'not in', 'contains', 'not contains'):
            operate = partial(_contains, loose_compare=loose_compare)
        else:
            raise ValueError(f'Operator should be one of {get_args(BaseConditionOperator)}, but got {self.operator}')
        
        try:
            result = bool(operate(left, right))
            if need_not:
                return not result
            return result
        except Exception as e:
            _logger.debug(f'Error when validating condition on `{obj}` with condition `{self}`: {type(e).__name__}: {e}')
            return False

if TYPE_CHECKING:
    _serializable_attrs = lambda x: x
else:
    _serializable_attrs = lambda x: serializable_attrs(eq=False)(x)

@_serializable_attrs
class Condition(_ConditionBase):
    
    left: "Condition|BaseCondition|bool"
    '''The left condition. It can be a Condition or BaseCondition.
    In some special case, it can be a bool constant.'''
    operator: ConditionOperator|None = None
    '''
    The operator for selection. It can be one of the following:
        - `and`
        - `or`
    NOTE: `&` / `|` is also supported(also `&&` / `||`).
    '''
    right: "Condition|BaseCondition|None" = None
    '''The right condition. It can be a Condition or BaseCondition.
    This is only used when operator is set.'''
    
    @overload
    def __init__(self, left: Self|BaseCondition|bool, operator: ConditionOperator|None = None, right: Self|BaseCondition|None = None): ...  # type: ignore
    @overload
    def __init__(self, condition_string: str, /): ...
    
    # region magic methods
    def __str__(self)->str:
        if isinstance(self.left, bool):
            return str(self.left).lower()
        else:
            left = str(self.left)
        if self.operator and self.right:
            return f'({left} {self.operator} {str(self.right)})'
        return left
    
    __repr__ = __str__
    
    def __setattr__(self, name, value):
        if name in ('left', 'operator', 'right'):
            curr_val = getattr(self, name, _empty)
            if curr_val != value:
                self.__normalized__ = False
        super().__setattr__(name, value)
    
    @override
    @no_type_check
    def __eq__(self, other)->bool:
        while isinstance(other, Condition) and (other.operator is None or other.right is None):
            other = other.left
        while isinstance(self, Condition) and (self.operator is None or self.right is None):
            self = self.left
            
        if (isinstance(self, BaseCondition) and isinstance(other, BaseCondition)) or \
                (isinstance(self, bool) and isinstance(other, bool)):
            return self == other
        
        if isinstance(self, Condition) and isinstance(other, Condition):
            self.normalize()
            other.normalize()
            if self.operator == other.operator:
                return ((self.left == other.left and self.right == other.right)
                        or
                        (self.left == other.right and self.right == other.left))
            elif self.right is not None and other.right is not None and isinstance(self.left, _ConditionBase):
                self_left_not = self.left.not_()    # type: ignore
                self_right_not = self.right.not_()
                return ((self_left_not == other.left and self_right_not == other.right)
                        or
                        (self_left_not == other.right and self_right_not == other.left))
        return False
    
    @no_type_check
    def __and__(self, other: 'str|BaseCondition|Condition|bool')->'Condition':
        while isinstance(other, Condition) and (other.operator is None or other.right is None):
            other = other.left
        if isinstance(self.left, bool) and self.right is None:
            if not self.left:
                return FalseCondition
            else:
                if isinstance(other, bool):
                    return TrueCondition if other else FalseCondition
                elif isinstance(other, BaseCondition):
                    return Condition(left=other)
                else:
                    return other
        if isinstance(other, bool):
            if other:
                return self
            else:
                return FalseCondition
        elif other == self:
            return self
        elif isinstance(other, BaseCondition):
            if (not self.operator or not self.right) and isinstance(self.left, BaseCondition):
                if other.not_() == self.left:
                    return FalseCondition
                elif other == self.left:
                    return self
            return Condition(left=self, operator='and', right=other)
        elif isinstance(other, Condition):
            return Condition(left=self, operator='and', right=other)
        elif isinstance(other, str):
            return self & Condition.Parse(other)
        raise ValueError(f'Condition should be a string, Condition or BaseCondition, but got {type(other)}')
    
    @no_type_check
    def __or__(self, other: 'str|BaseCondition|Self|bool')->'Condition':
        while isinstance(other, Condition) and (other.operator is None or other.right is None):
            other = other.left
        if isinstance(self.left, bool) and self.right is None:
            if self.left:
                return TrueCondition
            else:
                if isinstance(other, bool):
                    return TrueCondition if other else FalseCondition
                elif isinstance(other, BaseCondition):
                    return Condition(left=other)
                else:
                    return other
        if isinstance(other, bool):
            if other:
                return TrueCondition
            else:
                return self
        elif other == self:
            return self
        elif isinstance(other, BaseCondition):
            if (self.operator and self.right) and isinstance(self.left, BaseCondition):
                if other.not_() == self.left:
                    return TrueCondition
                elif other == self.left:
                    return self
            return Condition(left=self, operator='or', right=other)
        elif isinstance(other, Condition):
            return Condition(left=self, operator='or', right=other)
        elif isinstance(other, str):
            return self | Condition.Parse(other)
        raise ValueError(f'Condition should be a string, Condition or BaseCondition, but got {type(other)}')
    
    def __attrs_post_init__(self):
        if isinstance(self.left, str) and self.operator is None and self.right is None:
            # created by `Condition(condition_string)`
            cond = Condition.Parse(self.left)
            self.left = cond.left           # type: ignore
            self.operator = cond.operator   # type: ignore
            self.right = cond.right         # type: ignore
            
        if self.operator:
            if not self.right:
                _logger.warning('Condition operator is set, but right condition is not set. It will be ignored.')
                self.operator = None
                self.right = None
            else:                
                if not isinstance(self.operator, str):
                    raise ValueError(f'Operator should be a string: {self.operator}')
                operator = self.operator.lower().strip()
                if operator not in _operator_map:
                    raise ValueError(f'Operator should be one of {list(_operator_map.keys())}, but got {operator}.')
                self.operator = _operator_map[operator] # type: ignore
    
    @classmethod
    def __pydantic_deserialize__(cls, data):    # for `serializable_attrs`
        if isinstance(data, str):
            return cls.Parse(data)
        
        elif isinstance(data, (tuple, list)):
            if not data:
                return TrueCondition
            for chunk in data:
                if not isinstance(chunk, (str, BaseCondition, Condition)):
                    raise ValueError(f'Invalid condition part: {chunk}. It should be a string/Condition/BaseCondition.')
            if len(data) == 1:
                return cls(left=data[0])
            if len(data) == 2:
                second = data[1].lower().strip()
                if second in _operator_map:
                    _logger.warning(f'Condition operator is set, but right condition is not set. It will be ignored.') 
                    return cls(left=data[0])  # type: ignore
                else:
                    # default use `and` operator
                    return cls(left=data[0], operator='and', right=data[1])  # type: ignore
            if len(data) == 3:
                second = data[1].lower().strip()
                if second in _operator_map: 
                    return cls(left=data[0], operator=second, right=data[2])    # type: ignore
                else:
                    return cls(left=data[0], operator='and', right=data[1]) & cls(left=data[2])  # type: ignore
            else:
                conds = [cls.__pydantic_deserialize__(chunk) for chunk in data]
                cond = conds.pop(0)
                while conds:
                    cond = cond & conds.pop(0)
                if isinstance(cond, Condition):
                    cond.normalize()
                return cond
                
        elif isinstance(data, dict):
            if 'left' in data and 'operator' in data and 'right' in data:
                return cls(left=data['left'], operator=data['operator'], right=data['right']) # type: ignore
            elif 'left' in data and 'right' in data:
                return cls(left=data['left'], right=data['right'])    # type: ignore
            elif 'left' in data:
                return cls(left=data['left'])  # type: ignore
            raise ValueError(f'Invalid condition dict: {data}. It should be a dict with keys `left`, `operator` and `right`.')
            
        return data
    # endregion
    
    def normalize(self)->Self:
        '''Normalize the condition to a standard form.'''
        if getattr(self, '__normalized__', False):
            return self # no need to normalize again
        
        left, right = self.left, self.right
        while isinstance(left, Condition) and (left.operator is None or left.right is None):
            left = left.left
        while isinstance(right, Condition) and (right.operator is None or right.right is None):
            right = right.left
        
        if right is None and not isinstance(left, Condition):
            self.left = left
            self.__normalized__ = True
            return self # no need to normalize
        
        all_symbols = {}    # {var name: symbol}
        var_cond_map = {}    # {var name: condition}
        
        def get_var_name(cond_hash: str):
            return f'var_{cond_hash}'
        
        def get_symbol(cond_str: str, origin_cond):
            cond_hash = hash_md5(cond_str)
            var_name = get_var_name(cond_hash)
            if var_name in all_symbols:
                return all_symbols[var_name]
            else:
                symbol = all_symbols[var_name] = symbols(var_name)
                var_cond_map[var_name] = origin_cond
                return symbol
        
        def get_equation(cond: BaseCondition|Condition|bool):
            while isinstance(cond, Condition) and (cond.operator is None or cond.right is None):
                cond = cond.left
            if isinstance(cond, bool):
                origin_cond = TrueCondition if cond else FalseCondition
                return get_symbol(str(cond).lower(), origin_cond)
            elif isinstance(cond, BaseCondition):
                return get_symbol(str(cond), cond)
            else:
                left_eq, right_eq = get_equation(cond.left), get_equation(cond.right)   # type: ignore
                if cond.operator == 'and':
                    return And(left_eq, right_eq)
                else:
                    return Or(left_eq, right_eq)
        
        def get_cond_from_eq(expr: Expr):
            if expr is _False:
                return FalseCondition
            elif expr is _True:
                return TrueCondition
            expr_str = str(expr)
            return eval(expr_str, var_cond_map)
        
        eq = simplify_logic(get_equation(self))
        cond = get_cond_from_eq(eq) # type: ignore
        
        if isinstance(cond, Condition):
            self.left = cond.left   # type: ignore
            self.operator = cond.operator
            self.right = cond.right  # type: ignore
        elif isinstance(cond, (BaseCondition, bool)):
            self.left = cond
            self.operator = None
            self.right = None
        else:
            raise ValueError(f'Unknown error when normalizing condition: {self}. Got `{cond}` after simplification.')
        
        self.__normalized__ = True
        return self
    
    @classmethod 
    def Parse(cls, data: str, normalize: bool=True)->Self:
        '''
        Parse condition from string, e.g. 'a > 1 and b <2'
        
        Support base operators:
            - `=`: equal
            - `!=`: not equal
            - `>`: greater than
            - `<`: less than
            - `>=`: greater than or equal
            - `<=`: less than or equal
            - `in`
            - `not in`
            - `contains`
            - `not contains`
        
        Support logical operators:
            - `and`
            - `or`
        
        NOTE:
            1. '&' / '&&' / '|' / '||' is also supported.
            2. `==` is equivalent to `=`
            3. not(...) is also supported, but not recommended.
            4. you can define list by [1, 2, 3], items seems like number/boolean will be
                converted, otherwise, it will be treated as string.
            5. TODO: dict('{...}') is not yet supported.     
        '''
        cond, _ = _parse_condition(data)
        if normalize and not (cond is TrueCondition or cond is FalseCondition):
            cond.normalize()
        return cond # type: ignore
    
    def validate(
        self, 
        obj: object, 
        fuzzy: bool = False,
        loose_compare: bool = True,
    )->bool:
        '''
        Validate the condition on the object.
        Args:
            obj: The object to validate.
            fuzzy: If True, fuzzy match will be used when getting target attribute
                 from the object.
            loose_compare: If True, loose compare will be used when comparing the value,
                    e.g. `1` == `1.0`, '' == `None`, etc.
        '''
        self.normalize()
        if isinstance(self.left, bool):
            return self.left
        t = self.left.validate(obj, fuzzy, loose_compare)
        if self.operator and self.right:
            if self.operator == 'and':
                return t and self.right.validate(obj, fuzzy)
            elif self.operator == 'or':
                return t or self.right.validate(obj, fuzzy)
            raise ValueError(f'Operator should be one of {get_args(ConditionOperator)}, but got {self.operator}') 
        else:
            return t
    
    
TrueCondition: Final[Condition] = Condition(left=True)
'''Condition which is always True.'''
FalseCondition: Final[Condition] = Condition(left=False)
'''Condition which is always False.'''


__all__ = [
    'BaseCondition',
    'Condition',
    'TrueCondition',
    'FalseCondition',
]

if __name__ == '__main__':
    def basic_test():
        base_cond1 = BaseCondition(attr='b.c', operator='>', value=1)
        base_cond2 = BaseCondition(attr='s', operator='=', value='hi')
        cond = Condition(
            left=Condition(
                left=base_cond1,
                operator='or',
                right=base_cond2
            ), 
            operator='and', 
            right=base_cond2
        )
        print('before normalize:', cond)
        cond.normalize()
        print('after normalize:', cond) # x(x+y) = x
        
        from attr import attrib
        @serializable_attrs
        class B:
            c: int
        @serializable_attrs
        class A:
            b: B
            s: str
            lst: list[int] = attrib(factory=list)
        
        a1 = A(b=B(c=2), s='hi', lst=[1,])
        a2 = A(b=B(c=1), s='hi uncle')
        print('validate a1:', cond.validate(a1))
        print('validate a2:', cond.validate(a2))
        
        cond2 = cond & base_cond1
        print('after add base_cond1:', cond2)
        cond2.normalize()
        print('after normalize:', cond2) # should be the same
        print('validate a1:', cond2.validate(a1))
        print('validate a2:', cond2.validate(a2))
        
        cond3 = cond & (base_cond1 | BaseCondition(attr='b.c', operator='=', value=1))  # will merge into `>=1`
        print('cond3:', cond3)
        cond3.normalize()
        print('after normalize:', cond3) # should be the same
        print('validate a1:', cond3.validate(a1))
        print('validate a2:', cond3.validate(a2))
        
        query = 'b.c >= 1 and s in ["hi", "hi uncle"]'
        cond4 = Condition.Parse(query)
        print('cond4:', cond4)
        print('validate a1:', cond4.validate(a1))
        print('validate a2:', cond4.validate(a2))
        
        query = 'lst contains 1'
        cond5 = Condition.Parse(query)
        print('cond5:', cond5)
        print('validate a1:', cond5.validate(a1))
        print('validate a2:', cond5.validate(a2))
        
    basic_test()