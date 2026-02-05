import orjson
import logging
import builtins
import inspect

from attr import Attribute, attrs, NOTHING
from functools import partial
from inspect import _empty
from types import UnionType, NoneType, GenericAlias
from typing import (
    Any,
    Union,
    Literal,
    Self,
    Final,
    ClassVar,
    Annotated,
    TypeVar,
    overload,
    TypeAlias,
    Callable,
    Protocol,
    Coroutine,
    ForwardRef,
    TypeVarTuple,
    no_type_check,
    TYPE_CHECKING,
    get_args as tp_get_args,
    get_origin as tp_get_origin,
    runtime_checkable,
    _LiteralGenericAlias,  # type: ignore
    _CallableGenericAlias,  # type: ignore
    is_typeddict,
)
from typing_extensions import TypeForm, TypeAliasType
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1.fields import Undefined as PydanticV1Undefined
from pydantic import BaseModel as BaseModelV2, AliasChoices, create_model, ConfigDict
from pydantic.fields import PydanticUndefined  # type: ignore
from pydantic_core import core_schema
from pydantic_core.core_schema import JsonOrPythonSchema

_logger = logging.getLogger(__name__)

# region types
BasicType: TypeAlias = int | float | str | bool | bytes | list | tuple | dict | set | NoneType
"""Basic type of python, except for complex, range, slice, ellipsis, and types defined in typing module"""

BaseModelType: TypeAlias = BaseModelV1 | BaseModelV2
"""BaseModel type of pydantic, including BaseModelV1 and BaseModelV2"""

Number = TypeAliasType("Number", int | float)
"""Number type, including int and float"""


@runtime_checkable
class Comparable(Protocol):
    """Comparable protocol, for types that can be compared."""

    def __lt__(self, __other: Any) -> bool: ...
    def __eq__(self, __other: Any) -> bool: ...


@runtime_checkable
class StringLike(Protocol):
    """for types whom has implemented `__str__`"""

    def __str__(self) -> str: ...


__all__ = [
    "BasicType",
    "BaseModelType",
    "Number",
    "Comparable",
    "StringLike",
]
# endregion

def get_original_bases(cls: type) -> list[type]:
    """
    Get the original bases of a class, including generic types.
    Different with `cls.__bases__`, this function will return the original bases with type arguments(if any).
    e.g.
        ```
        class A[T]: ...
        class B(A[int]): ...
        get_original_bases(B) -> [A[int]]
        ```
    """
    if hasattr(cls, "__orig_bases__"):
        return list(cls.__orig_bases__)  # type: ignore
    else:
        return list(cls.__bases__)  # type: ignore

_T = TypeVar('_T')
_TT = TypeVar('_TT', bound=type)
_TO = TypeVar('_TO', bound=object)

@overload
def get_sub_clses(cls_or_ins: _TT) -> tuple[_TT, ...]: ...
@overload
def get_sub_clses(cls_or_ins: _TO) -> tuple[type[_TO], ...]: ...

def get_sub_clses(cls_or_ins):
    """
    Get all sub classes of a class, recursively.
    The class itself will also be included as the first element.
    """
    from .checking import _tidy_type

    cls_or_ins = _tidy_type(cls_or_ins)[0]  # type: ignore

    if not isinstance(cls_or_ins, type):
        cls_or_ins = type(cls_or_ins)
    if not hasattr(cls_or_ins, "__subclasses__"):
        return (cls_or_ins,)
    else:
        sub_clses = cls_or_ins.__subclasses__()
        all_subclses = [
            cls_or_ins,
        ]
        for sub_cls in sub_clses:
            sub_sub_clses = get_sub_clses(sub_cls)
            for sub_sub_cls in sub_sub_clses:
                if sub_sub_cls not in all_subclses:
                    all_subclses.append(sub_sub_cls)
        return tuple(all_subclses)

def getmro(cls: type) -> tuple[type, ...]:
    """
    Get the method resolution order of a class, recursively.
    Different with inspect.getmro, this function will return the original bases of the class(if any),
    i.e. `A[int]` instead of `A`.
    """
    from .checking import _tidy_type

    cls = _tidy_type(cls)[0]  # type: ignore
    try:
        clses = {}

        def insert(cls, depth=0, seen=0):
            nonlocal clses
            if cls in clses:
                if clses[cls][0] > depth:
                    return
                if clses[cls][1] >= seen:
                    return
            clses[cls] = (depth, seen)
            origin = tp_get_origin(cls)
            if origin:
                bases = get_original_bases(origin)
            else:
                bases = get_original_bases(cls)
            for b in bases:
                seen += 1
                insert(b, depth + 1, seen)

        insert(cls)
        return tuple(sorted(clses.keys(), key=lambda x: clses[x][0]))
    except Exception:  # some special types may fail to get mro
        return (cls,)

def get_cls_annotations(
    cls: type | object,
    no_cls_var: bool = False,
    no_final: bool = False,
) -> dict[str, type]:
    """
    Recursively get the annotations of a class, including its base classes.

    Args:
        - `cls`: the class or instance
        - `no_cls_var`: if True, will not include `ClassVar` annotations.
        - `no_final`: if True, will not include `Final` annotations.

    Some special case to note:
    1. type vars will be filled with the actual type arguments if available,
        e.g.
        ```
        class A[T]:
            x: T

        class B(A[int]):...

        get_cls_annotations(B) -> {'x': int}
        ```

    2. empty type alias type will be converted to the real type, e.g.
        ```
        type Int = int

        class A:
            x: Int

        get_cls_annotations(A) -> {'x': int}
        ```
    """
    from .checking import _tidy_type
    
    if isinstance(cls, TypeAliasType):
        cls = cls.__value__
    if cls is object:
        return {}
    
    arg_matches = {}
    origin = tp_get_origin(cls) or cls
    try:
        bases = get_original_bases(origin)  # type: ignore
    except:
        bases = []
    
    type_params = None
    if args := tp_get_args(cls):
        type_params = getattr(origin, "__type_params__", None)
    if not args and (pd_generic_meta:=getattr(origin, "__pydantic_generic_metadata__", None)):
        # special case for pydantic generic models
        args = pd_generic_meta.get("args", None)
        if not type_params and bases and (base_params:=getattr(bases[0], '__parameters__', None)):
            type_params = base_params
    if type_params and args:
        if len(type_params) == len(args):
            if len(type_params) == 1 and isinstance(type_params[0], TypeVarTuple):
                args = tuple([args])
            for t, a in zip(type_params, args):
                arg_matches[t] = a
    
    annos = {}
    for b in bases[::-1]:
        if b is object:
            continue
        annos.update(get_cls_annotations(b, no_cls_var=no_cls_var, no_final=no_final))  # type: ignore
        
    cls_annos = {}
    if not hasattr(cls, "__annotations__"):
        if hasattr(cls, "__origin__") and hasattr(cls, "__args__"):
            # this is a type alias type, e.g. `A[int]`
            cls = cls.__origin__  # type: ignore
            
    for k, v in getattr(cls, "__annotations__", {}).items():
        cls_annos[k] = _tidy_type(v, arg_matches)[0]  # type: ignore
    if not is_typeddict(cls):
        annos.update(cls_annos)  # type: ignore
    else:   # special case, as typeddict will include generic types in its annotations
        for k, v in cls_annos.items():
            if isinstance(v, TypeVar):
                if k in annos:
                    curr = annos[k]
                    if isinstance(curr, TypeVar):
                        annos[k] = v
                    else:
                        ...
                else:
                    annos[k] = v
            else:
                annos[k] = v
    
    tidied = {}
    for k, v in annos.items():
        t = _tidy_type(v, arg_matches)
        try:
            t = t[0]  # type: ignore
        except TypeError:
            ...
        t_origin = tp_get_origin(t)
        if t_origin is ClassVar and no_cls_var:
            continue
        if t_origin is Final and no_final:
            continue
        tidied[k] = t
    return tidied

def get_origin(t: Any, self=None, return_t_if_no_origin: bool = False) -> type | None:  # type: ignore
    """
    Return the origin type of the type hint.
    Different to typing.get_origin, this function will convert some special types to their real origin type,

    Args:
        `self`: if provided, for type = `Self`, it will return the `self`.
        `return_t_if_no_origin`: if True, will return the type itself if no origin is found.

    e.g.
        * int|str -> Union                  (the origin typing.get_origin will return UnionType, which is not easy to do comparison)
        * ForwardRef('A') -> ForwardRef     (the origin typing.get_origin will return None, which is not correct)
        * _empty -> Any
    """
    from .checking import _tidy_type

    tt = _tidy_type(t)
    try:
        t = tt[0]  # type: ignore
    except TypeError:
        t = tt

    if t == _empty:
        return Any  # type: ignore
    if isinstance(t, ForwardRef):
        return ForwardRef
    if t == Self:
        if self is not None:
            if isinstance(self, type):
                return type(self)
            else:
                return self
        else:
            return Self

    origin = tp_get_origin(t)
    if origin in (UnionType, Union):
        return Union  # type: ignore

    if return_t_if_no_origin and origin is None:
        return t
    return origin

def get_args(t, str_to_type: bool = True) -> tuple[Any, ...]:
    """
    Return the args of the type hint.
    Different to typing.get_args, this function will convert some special types to their real args,

    e.g.
        * ForwardRef('A') -> ('A',)     (the origin typing.get_args will return (), which is not correct)

    if `str_to_type` is True, string type args will try to be converted to their real type,
    e.g. list['int'] -> (int, ) instead of ('int', )
    """
    from .checking import _tidy_type

    tt = _tidy_type(t)
    try:
        t = tt[0]  # type: ignore
    except TypeError:
        t = tt

    if isinstance(t, ForwardRef):
        r = (t.__forward_arg__,)
    else:
        r = tp_get_args(t)
    if str_to_type:
        converted_r = []
        for arg in r:
            if isinstance(arg, str):
                converted_r.append(_tidy_type(arg)[0])  # type: ignore
            else:
                converted_r.append(arg)
    return r

def get_cls_name(
    cls: Any,
    with_module_name: bool = False,
    with_generic: bool = True,
    no_qualname: bool = False,
    ellipsis_to_dots: bool = False,
) -> str:
    """
    Return the pure class name, without module name. e.g. 'A' instead of 'utils.xxx....A
    If `__qualname__` is not available, it will use `__name__` instead.
    For generic class, it will return the class with its type arguments, e.g. `List[int]`.

    Args:
        cls: the class to be get name.
        with_module_name: if True, will return the class name with module name, e.g. 'utils.xxx....A'
                         This is only available for non-builtin classes.
        with_generic: if True, will return the class name with its type arguments, e.g. `List[int]`,
                        otherwise, will return the class name without type arguments, e.g. `List`.
        no_qualname: if True, when the class is under another class, it will return the pure
                    class name only, without the parent class name.
        ellipsis_to_dots: if False, will return 'Ellipsis' instead of '...' for `...` type.

    Note:
        1. if `cls` is a string, it will return the string itself, instead of `str`.
        2. For Literal, if `with_generic` is False, it will return `Literal` instead of `Literal[...]`.
        3. For Union, if `with_generic` is False, it will return `Union` instead of `Union[...]`.
        4. For ForwardRef, the return value will be the forward ref string itself.
    """
    from .checking import _tidy_type, _save_isinstance

    raw_cls = cls
    module = get_module_name(cls) if hasattr(cls, "__module__") else None
    cls = _tidy_type(cls)[0]  # type: ignore
    if not module and cls != raw_cls:
        module = get_module_name(cls.__module__) if hasattr(cls, "__module__") else None

    if with_generic and _save_isinstance(cls, GenericAlias):
        main_cls_name = get_cls_name(cls.__origin__, with_module_name, False)
        arg_names = []
        for arg in cls.__args__:
            arg_name = get_cls_name(arg, with_module_name, with_generic)
            if arg_name.lower() == "ellipsis":
                arg_name = "..."
            arg_names.append(arg_name)
        return f"{main_cls_name}[{', '.join(arg_names)}]"

    if _save_isinstance(cls, str):
        # seems to be a class name already
        return cls

    elif _save_isinstance(cls, TypeVar):
        n = "TypeVar" if not with_module_name else "typing.TypeVar"
        if with_generic:
            n += f"[{cls.__name__}"
            if constraints := cls.__constraints__:
                n += ": (" + ", ".join(get_cls_name(c, with_module_name, with_generic) for c in constraints) + ")]"
            else:
                n += "]"
        return n

    elif _save_isinstance(cls, UnionType):
        # Union
        if not with_generic:
            return "Union"
        else:
            name_str = ", ".join(get_cls_name(arg, with_module_name, with_generic) for arg in cls.__args__)
            name_str = f"Union[{name_str}]"
            if module and with_module_name:
                return f"{module}.{name_str}"
            return name_str

    elif _save_isinstance(cls, ForwardRef):
        # ForwardRef
        if with_module_name and cls.__forward_module__:
            if module:
                return f"{module}.{cls.__forward_arg__}"
            return f"{cls.__forward_module__}.{cls.__forward_arg__}"
        return cls.__forward_arg__

    elif _save_isinstance(cls, _LiteralGenericAlias):
        # Literal
        def get_arg_str(a):
            if type(a) == "str":
                return f"'{a}'"
            return str(a)

        full_str = "Literal[" + ", ".join(get_arg_str(a) for a in cls.__args__) + "]"
        if module and with_module_name:
            return f"{module}.{full_str}"
        return full_str

    elif _save_isinstance(cls, _CallableGenericAlias):
        # Callable
        if not with_generic:
            return "Callable"
        params, ret = get_args(cls)
        if params == Ellipsis:
            params_str = "..."
        else:
            params_str = ", ".join(get_cls_name(p, with_module_name, with_generic) for p in params)
        ret_str = get_cls_name(ret, with_module_name, with_generic, no_qualname, ellipsis_to_dots)
        full_str = f"Callable[[{params_str}], {ret_str}]"
        if module and with_module_name:
            return f"{module}.{full_str}"
        return full_str

    elif get_origin(cls) in (Final, ClassVar, Annotated):
        return get_cls_name(
            get_args(cls)[0],
            with_module_name,
            with_generic,
            no_qualname,
            ellipsis_to_dots,
        )

    elif cls == Callable:
        if module and with_module_name:
            return f"{module}.Callable"
        return "Callable"

    elif cls == Ellipsis:
        if ellipsis_to_dots:
            return "..."
        return "Ellipsis"

    elif not _save_isinstance(cls, type):
        # seems giving an object
        cls = type(cls)

    if hasattr(cls, "__qualname__"):
        n = cls.__qualname__
    elif hasattr(cls, "__name__"):
        n = cls.__name__
    elif hasattr(cls, "__repr__"):
        n = cls.__repr__().split(".")[-1].split("<")[0].split("[")[0].split("(")[0].split("{")[0]
    else:
        n = str(cls).split(".")[-1].split("<")[0].split("[")[0].split("(")[0].split("{")[0]

    if not with_generic and "[" in n:
        n = n.split("[")[0]
    if no_qualname:
        n = n.split(".")[-1]

    if with_module_name and not is_builtin(cls):
        if module:
            return f"{module}.{n}"
        try:
            module_name = get_module_name(cls)
        except:
            module_name = ""
        if module_name:
            return f"{module_name}.{n}"
    return n

@no_type_check
def get_module_name(t: Any) -> str:
    """
    Get the proper module name of the type.
    This is useful when running scripts directly for debugging,

    e.g. you define a class in `utils.xxx....`, but the module will shows '__main__' when running the script directly.
    Class will be redefined by python as '__main__.A', which is different from 'utils.xxx....A'.
    By using this function, you could get the proper module name `utils.xxx....` instead of `__main__`
    """
    from .checking import _get_module_name
    return _get_module_name(t)

MAX_MRO_DISTANCE = 999

def get_mro_distance(cls: Any, super_cls: type | str | None) -> int:
    """
    Return the distance of cls to super_cls in the mro.
    If cls is not a subclass of super_cls, return 999.

    Args:
        cls: the class to be checked.
        super_cls: the super class to be checked. It could also be special types like Union, Optional, ForwardRef, etc.
    """
    from .checking import _tidy_type, check_type_is

    cls = _tidy_type(cls)
    super_cls = _tidy_type(super_cls)[0]  # type: ignore

    if cls is None and super_cls is None:
        return 0
    elif cls is None or super_cls is None:
        return MAX_MRO_DISTANCE

    if cls == Any:
        cls = object
    if super_cls == Any:
        super_cls = object

    if not check_type_is(cls, super_cls):
        return MAX_MRO_DISTANCE

    origin = get_origin(super_cls)
    type_args = get_args(super_cls)

    if origin == Union and type_args:
        return min(get_mro_distance(cls, t) for t in type_args)

    elif origin == Literal and type_args:
        try:
            return (
                type_args == get_args(cls) and get_origin(cls) == Literal
            )  # e.g. Literal[1, 2, 3] == Literal[1, 2, 3] -> True
        except:
            return MAX_MRO_DISTANCE
    elif (origin == ForwardRef and type_args) or isinstance(super_cls, str):
        cls_mro_names = [get_cls_name(c) for c in getmro(cls)]
        try:
            return cls_mro_names.index(super_cls if isinstance(super_cls, str) else type_args[0])
        except ValueError:  # not found
            return MAX_MRO_DISTANCE
    else:
        try:
            return getmro(cls).index(super_cls)
        except ValueError:  # not found
            return MAX_MRO_DISTANCE

def is_builtin(obj: Any) -> bool:
    """check if an object is a builtin function or type."""
    from .checking import _tidy_type
    obj = _tidy_type(obj)[0]  # type: ignore
    if not (r := inspect.isbuiltin(obj)):
        cls_name = get_cls_name(obj)
        r = hasattr(builtins, cls_name)
    return r

@overload
def getattr_raw(obj: Any, attr_name: str) -> Any: ...
@overload
def getattr_raw(obj: Any, attr_name: str, raise_err: Literal[True]) -> Any: ...
@overload
def getattr_raw(obj: Any, attr_name: str, raise_err: Literal[False]) -> Any | inspect.Parameter.empty: ...

def getattr_raw(obj, attr_name: str, raise_err=True):
    """
    Get the attr object with the given name.
    Different from `getattr`, this method will avoid triggering magic methods, e.g. `__getattr__`,
    `__getattribute__`, `__get__`, etc.
    """
    if not isinstance(obj, type) and hasattr(obj, "__dict__") and attr_name in obj.__dict__:
        return obj.__dict__[attr_name]

    object_type = type(obj) if not isinstance(obj, type) else obj
    all_clses = (*object_type.__bases__, object_type)
    for cls in all_clses[::-1]:
        if hasattr(cls, "__dict__") and attr_name in cls.__dict__:
            return cls.__dict__[attr_name]
    if not raise_err:
        return inspect.Parameter.empty
    raise AttributeError(f"{obj} has no attribute {attr_name}.")


if TYPE_CHECKING:
    Empty = TypeForm["Empty"]
    """
    Marker object for an empty parameter, for cases that you don't want to use `None` as default value.
    This is actually inspect.Parameter.empty, but some special treatment is done to make it 
    available for easy type hints like `x: T|Empty = Empty`.
    
    Example:
    ```python
    def f(x: int|Empty = Empty):
        ...
    ```
    """
else:
    Empty = inspect.Parameter.empty  # type: ignore

    # make `inspect.Parameter.empty` serializable in pydantic
    if not hasattr(inspect.Parameter.empty, "__get_pydantic_core_schema__"):

        @classmethod
        def __get_pydantic_core_schema__(cls, source, handler):
            from pydantic_core import core_schema

            def validator(value):
                if isinstance(value, dict) and value.get("type") == "empty" and len(value) == 1:
                    return inspect.Parameter.empty
                elif value == inspect.Parameter.empty:
                    return inspect.Parameter.empty
                raise ValueError(f"Cannot deserialize value `{value}` to empty.")

            def serializer(value):
                if value != Empty:
                    return value
                return {
                    "type": "empty",
                }

            validate_schema = core_schema.no_info_after_validator_function(validator, core_schema.any_schema())
            serialize_schema = core_schema.plain_serializer_function_ser_schema(serializer, when_used="unless-none")
            return core_schema.json_or_python_schema(
                json_schema=validate_schema,
                python_schema=validate_schema,
                serialization=serialize_schema,
            )

        setattr(
            inspect.Parameter.empty,
            "__get_pydantic_core_schema__",
            __get_pydantic_core_schema__,
        )
        
    inspect.Parameter.empty.__repr__ = lambda *args: "Empty"  # type: ignore


def is_attrs_cls(cls: Any) -> bool:
    """
    Check if a class is an attrs class, i.e. decorated by `@attrs`.
    You can also pass the object of the class, it will automatically convert to the class.
    """
    if not isinstance(cls, type):
        cls = type(cls)
    return hasattr(cls, "__attrs_attrs__")


def is_dataclass(cls: Any) -> bool:
    """
    check if a class is a dataclass, i.e. decorated by `@dataclass`.
    You can also pass the object of the class, it will automatically convert to the class.
    """
    if not isinstance(cls, type):
        cls = type(cls)
    return hasattr(cls, "__dataclass_fields__")


def get_attr_cls_default_pydantic_validator(cls: type[_T]) -> Callable[[Any], _T]:
    """
    Create or get a default validator(deserializer) for a class decorated by `@attrs`.
    This validator is valid for using in pydantic module.
    The default validator will loop through all the fields of the class and
    validate them one by one.

    NOTE: `Self` is also supported in the validator.
    """
    from ..data_structs.advanced_builtins import FuzzyDict
    from .convertors import get_pydantic_type_adapter

    if "__DefaultAttrPydanticValidator__" in cls.__dict__:
        return cls.__DefaultAttrPydanticValidator__  # type: ignore

    attrs: tuple[Attribute, ...] = cls.__attrs_attrs__  # type: ignore
    if (
        "__DefaultPydanticTypeAdapters__" not in cls.__dict__
        or "__DefaultFuzzyFieldMatchDict__" not in cls.__dict__
        or "__DefaultSelfValidators__" not in cls.__dict__
    ):
        fuzzy_match_dict = FuzzyDict()
        type_adapters = {}
        self_validators = {}

        def self_validator(val, origin_validator):
            return origin_validator(data=val)

        def self_list_validator(val, origin_validator, is_tuple=False):
            lst_adapter = get_pydantic_type_adapter(list)
            lst_val = lst_adapter.validate_python(val)
            for i, v in enumerate(tuple(lst_val)):
                lst_val[i] = origin_validator(data=v)
            if is_tuple:
                return tuple(lst_val)
            return lst_val

        def self_tuple_validator(val, origin_validator, self_pos: list[int]):
            lst_adapter = get_pydantic_type_adapter(list)
            lst_val = lst_adapter.validate_python(val)
            if len(lst_val) != len(self_pos):
                raise ValueError(f"Tuple length {len(lst_val)} does not match self position {self_pos}.")
            for i in self_pos:
                lst_val[i] = origin_validator(data=lst_val[i])
            return tuple(lst_val)

        def self_dict_validator(val, origin_validator):
            dict_adapter = get_pydantic_type_adapter(dict)
            dict_val = dict_adapter.validate_python(val)
            for k, v in dict_val.items():
                dict_val[k] = origin_validator(data=v)
            return dict_val

        def self_union_validator(val, origin_validator, other_types: list):
            for t in other_types:
                adapter = get_pydantic_type_adapter(t)
                try:
                    return adapter.validate_python(val)
                except:
                    continue
            try:
                return origin_validator(data=val)
            except Exception as e:
                raise ValueError(f"Cannot validate value {val} to {other_types} or `{cls}`. Error: {type(e)}-{e}")

        def self_str_union_validator(val, origin_validator, other_types: list[str]):
            union_adapter = get_pydantic_type_adapter("|".join(other_types))
            try:
                return union_adapter.validate_python(val)
            except:
                try:
                    return origin_validator(data=val)
                except Exception as e:
                    raise ValueError(f"Cannot validate value {val} to {other_types} or `{cls}`. Error: {type(e)}-{e}")

        cls_name = cls.__name__.split(".")[-1]

        def is_self_t(t):
            if isinstance(t, str) and t == cls_name:
                return True
            return t == Self

        def any_self_t(ts):
            if not ts:
                return False
            for t in ts:
                if is_self_t(t):
                    return True
            return False

        def is_union_self_t(t):
            if isinstance(t, str):
                if "|" in t:
                    t_str_chunks = t.split("|")
                    for t_str in t_str_chunks:
                        if t_str in (cls_name, "Self"):
                            return True
                elif t.startswith("Union["):
                    t_str_chunks = t[6:-1].split(",")
                    for t_str in t_str_chunks:
                        if t_str.strip() in (cls_name, "Self"):
                            return True
            return False

        @no_type_check
        def recursive_get_self_validator(t):
            t_origin, t_args = get_origin(t), get_args(t)
            if is_union_self_t(t):
                if "|" in t:  # type: ignore
                    t_str_chunks = t.split("|")  # type: ignore
                    other_types = [t for t in t_str_chunks if not is_self_t(t)]
                    return partial(self_str_union_validator, other_types=other_types)
                elif t.startswith("Union["):  # type: ignore
                    t_str_chunks = t[6:-1].split(",")  # type: ignore
                    other_types = [t.strip() for t in t_str_chunks if not is_self_t(t.strip())]
                    return partial(self_str_union_validator, other_types=other_types)
                return None

            elif t_origin in (Union, UnionType) and t_args:
                if any_self_t(t_args):
                    other_types = [t for t in t_args if not is_self_t(t)]
                    return partial(self_union_validator, other_types=other_types)
                else:
                    validators = [recursive_get_self_validator(t) for t in t_args]
                    if any(validators):
                        other_types = []
                        other_validators = []
                        for a, v in zip(t_args, validators):
                            if v:
                                other_validators.append(v)
                            else:
                                other_types.append(a)

                        @no_type_check
                        def joint_union_validator(val, origin_validator, other_types, other_validators):
                            if other_types:
                                union_adapter = get_pydantic_type_adapter(Union[*other_types])
                                try:
                                    return union_adapter.validate_python(val)
                                except:
                                    ...
                            for v in other_validators:
                                try:
                                    return v(val, origin_validator)
                                except:
                                    ...
                            raise ValueError(f"Cannot validate value {val} to {t}.")

                        return partial(
                            joint_union_validator,
                            other_types=other_types,
                            other_validators=other_validators,
                        )
                    return None

            elif t_origin == list and t_args:
                if is_self_t(t_args[0]):
                    return partial(self_list_validator, is_tuple=False)
                else:
                    inner_validator = recursive_get_self_validator(t_args[0])
                    if inner_validator:

                        def list_validator(val, origin_validator, inner_validator):
                            lst_adapter = get_pydantic_type_adapter(list)
                            lst_val = lst_adapter.validate_python(val)
                            for i, v in enumerate(tuple(lst_val)):
                                lst_val[i] = inner_validator(v, origin_validator)
                            return lst_val

                        return partial(list_validator, inner_validator=inner_validator)
                    return None

            elif t_origin == set and t_args:
                if is_self_t(t_args[0]):
                    return partial(self_list_validator, is_tuple=False)
                else:
                    inner_validator = recursive_get_self_validator(t_args[0])
                    if inner_validator:

                        def set_validator(val, origin_validator, inner_validator):
                            set_adapter = get_pydantic_type_adapter(set)
                            set_val = set_adapter.validate_python(val)
                            tidied = set()
                            for v in set_val:
                                tidied.add(inner_validator(v, origin_validator))
                            return tidied

                        return partial(set_validator, inner_validator=inner_validator)
                    return None

            elif t_origin == tuple and t_args:
                if any_self_t(t_args):
                    if len(t_args) == 2 and t_args[1] == Ellipsis:
                        self_validators[attr.alias] = partial(self_list_validator, is_tuple=True)
                    else:
                        self_pos = [i for i, t in enumerate(t_args) if is_self_t(t)]
                        self_validators[attr.alias] = partial(self_tuple_validator, self_pos=self_pos)
                else:
                    inner_validators = [recursive_get_self_validator(t) for t in t_args]
                    if any(inner_validators):
                        fake_ts = []
                        for i, v in enumerate(inner_validators):
                            if v:
                                fake_ts.append(Any)
                            else:
                                fake_ts.append(t_args[i])
                        fake_tuple_adapter = get_pydantic_type_adapter(tuple[*fake_ts])

                        def tuple_validator(val, origin_validator, fake_tuple_adapter, inner_validators):
                            lst_val = list(fake_tuple_adapter.validate_python(val))
                            for i, v in enumerate(tuple(lst_val)):
                                if inner_validators[i]:
                                    lst_val[i] = inner_validators[i](v, origin_validator)
                            return tuple(lst_val)

                        return partial(
                            tuple_validator,
                            fake_tuple_adapter=fake_tuple_adapter,
                            inner_validators=inner_validators,
                        )
                    return None

            elif t_origin == dict and t_args and len(t_args) == 2:
                if is_self_t(t_args[1]):
                    return self_dict_validator
                else:
                    inner_validator = recursive_get_self_validator(t_args[1])
                    if inner_validator:
                        fake_adapter = get_pydantic_type_adapter(dict[t_args[0], Any])

                        def dict_validator(val, origin_validator, fake_adapter, inner_validator):
                            dict_val = fake_adapter.validate_python(val)
                            for k, v in dict_val.items():
                                dict_val[k] = inner_validator(v, origin_validator)
                            return dict_val

                        return partial(
                            dict_validator,
                            fake_adapter=fake_adapter,
                            inner_validator=inner_validator,
                        )
                    return None

            elif is_self_t(t):
                return self_validator

            return None

        cls_annos = get_cls_annotations(cls)  # type: ignore
        for attr in attrs:  # type: ignore
            t = cls_annos.get(attr.name, attr.type)  # type: ignore
            _self_validator = recursive_get_self_validator(t)  # type: ignore
            if _self_validator:
                self_validators[attr.alias] = _self_validator
            else:
                try:
                    type_adapters[attr.alias] = get_pydantic_type_adapter(t)  # type: ignore
                except Exception as e:
                    raise ValueError(
                        f"Type {t} is not serializable. Cannot be used in pydantic model. Error: {type(e)}-{e}"
                    )

            fuzzy_match_dict[attr.name] = attr.alias
            fuzzy_match_dict[attr.alias] = attr.alias

        setattr(cls, "__DefaultPydanticTypeAdapters__", type_adapters)
        setattr(cls, "__DefaultFuzzyFieldMatchDict__", fuzzy_match_dict)
        setattr(cls, "__DefaultSelfValidators__", self_validators)
    else:
        self_validators = cls.__DefaultSelfValidators__  # type: ignore
        type_adapters = cls.__DefaultPydanticTypeAdapters__  # type: ignore
        fuzzy_match_dict = cls.__DefaultFuzzyFieldMatchDict__  # type: ignore

    def default_validator(cls, data):
        if isinstance(data, cls):
            return data
        if isinstance(data, (str, bytes)):
            try:
                data = orjson.loads(data)
            except:
                ...
        if isinstance(data, dict):
            tidied = {}
            for k, v in data.items():
                proper_k = fuzzy_match_dict.get(k, k)
                if proper_k in type_adapters:
                    tidied[proper_k] = type_adapters[proper_k].validate_python(v)
                elif proper_k in cls.__dict__:
                    tidied[proper_k] = self_validators[proper_k](v, partial(default_validator, cls=cls))
            return cls(**tidied)
        else:
            return cls(data)  # type: ignore

    default_validator = partial(default_validator, cls)  # type: ignore
    cls.__DefaultAttrPydanticValidator__ = default_validator  # type: ignore
    return default_validator  # type: ignore


def get_attr_cls_pydantic_validator(cls: type[_T]) -> Callable[[Any], _T]:
    """
    Create or get a validator(deserializer) for a class decorated by `@attrs`.
    This validator is valid for using in pydantic module.

    if `__pydantic_deserialize__` is defined in the class, it will be
    used as the validator. This function can be in form of:
        - (data) -> T
        - (data, default_validator) -> T

    e.g:
    ```
    @attrs
    class A:
        @classmethod
        def __pydantic_deserialize__(cls, data, default_validator):
            return default_validator(data)
    ```
    """
    if not isinstance(cls, type) or not is_attrs_cls(cls):
        raise ValueError(f"`cls` must be a class with attrs decorator, got {cls}")

    custom_validator = None
    if hasattr(cls, "__pydantic_deserialize__"):
        custom_deserialize = cls.__pydantic_deserialize__  # type: ignore
        param_count = len(inspect.signature(custom_deserialize).parameters)
        if param_count == 1:
            return lambda data, handler: custom_deserialize(data)  # type: ignore
        elif param_count > 2:
            raise ValueError(f"`__pydantic_deserialize__` must have 1 or 2 parameters, got {param_count}.")
        custom_validator = custom_deserialize

    if "__attr_pydantic_validator__" in cls.__dict__:
        return cls.__attr_pydantic_validator__  # type: ignore

    default_validator = get_attr_cls_default_pydantic_validator(cls)  # type: ignore

    if custom_validator:

        def validator(cls, data, handler=None):  # type: ignore
            if handler:
                data = handler(data)
            return custom_validator(data, default_validator)

    else:

        def validator(cls, data, handler=None):
            if handler:
                data = handler(data)
            return default_validator(data)

    validator = partial(validator, cls)  # type: ignore
    cls.__attr_pydantic_validator__ = validator  # type: ignore
    return validator  # type: ignore


def get_attrs_cls_pydantic_schema(cls: type) -> JsonOrPythonSchema:
    """
    Get attrs cls's pydantic schema. Attrs class is those classes decorated by `@attrs`.
    This helps the class to be a valid field inside pydantic model.

    You can define:
        - `__pydantic_serialize__`: it will become the serialization schema,
            otherwise, `recursive_dump_to_basic_types` will be used as serialization schema.
        - `__pydantic_deserialize__`: it will become the deserialization schema,
            otherwise, the default implementation is to deserialize data by field annotations.
            This function can accept 1 or 2 parameters(data, handler), and must be callable
            by class directly(i.e. classmethod/staticmethod).

    Note:
        All fields inside this attrs class must be serializable, otherwise, it will raise error
        during serialization.
    """
    from .convertors import recursive_dump_to_basic_types
    from .func_helpers import func_arg_count
    from ..concurrent_utils import wait_coroutine

    if not isinstance(cls, type) or not is_attrs_cls(cls):
        raise ValueError(f"`cls` must be a class with attrs decorator, got {cls}")
    if "__attrs_pydantic_schema__" in cls.__dict__:
        return cls.__attrs_pydantic_schema__  # type: ignore

    validator = get_attr_cls_pydantic_validator(cls)
    # `__pydantic_deserialize__` is gotten from this function

    if hasattr(cls, "__pydantic_serialize__"):
        custom_ser_func = getattr_raw(cls, "__pydantic_serialize__", raise_err=True)
        if isinstance(custom_ser_func, (classmethod, staticmethod)):
            custom_ser_func = custom_ser_func.__get__(None, cls)
        if not callable(custom_ser_func):
            raise ValueError(f"`__pydantic_serialize__` must be callable, got {type(custom_ser_func)}.")
        if (ac := func_arg_count(custom_ser_func)) != 1:  # type: ignore
            raise ValueError(f"`__pydantic_serialize__` must have 1 parameter, got {ac}.")

        def serializer_wrapper(custom_ser_func, data):
            data = custom_ser_func(data)
            if isinstance(data, Coroutine):
                data = wait_coroutine(data)
            return data

        serializer = partial(serializer_wrapper, custom_ser_func)
    else:

        def serializer(data):
            return recursive_dump_to_basic_types(data)

    validate_schema = core_schema.no_info_wrap_validator_function(validator, core_schema.any_schema())  # type: ignore
    serialize_schema = core_schema.plain_serializer_function_ser_schema(serializer)
    schema = core_schema.json_or_python_schema(
        json_schema=validate_schema,
        python_schema=validate_schema,
        serialization=serialize_schema,
    )
    cls.__attrs_pydantic_schema__ = schema
    return schema


if TYPE_CHECKING:
    serializable_attrs = attrs
    """
    Advance decorator for attrs class, which makes it supports pydantic serialization/deserialization.
    For fields explanation, please refer to `attrs` documentation.

    Note: 
        `auto_attribs` is default be True.

        You can defined:
            - `__pydantic_serialize__`: it will become the serialization schema,
                otherwise, `recursive_dump_to_basic_types` will be used as serialization schema.
            - `__pydantic_deserialize__`: it will become the deserialization schema,
                otherwise, the default implementation is to deserialize data by field annotations.
                
        All fields inside this attrs class must be serializable, otherwise, it will raise error
        during serialization.
    """
else:

    def serializable_attrs(
        maybe_cls=None,
        these=None,
        repr_ns=None,
        repr=None,
        cmp=None,
        hash=None,
        init=None,
        slots=False,
        frozen=False,
        weakref_slot=True,
        str=False,
        auto_attribs=True,
        kw_only=False,
        cache_hash=False,
        auto_exc=False,
        eq=None,
        order=None,
        auto_detect=False,
        collect_by_mro=False,
        getstate_setstate=None,
        on_setattr=None,
        field_transformer=None,
        match_args=True,
        unsafe_hash=None,
    ):
        def wrapper(cls):
            decorated_cls = attrs(
                maybe_cls=None,
                these=these,
                repr_ns=repr_ns,
                repr=repr,  # type: ignore
                cmp=cmp,
                hash=hash,
                init=init,  # type: ignore
                slots=slots,
                frozen=frozen,
                weakref_slot=weakref_slot,
                str=str,
                auto_attribs=auto_attribs,
                kw_only=kw_only,
                cache_hash=cache_hash,
                auto_exc=auto_exc,
                eq=eq,
                order=order,
                auto_detect=auto_detect,
                collect_by_mro=collect_by_mro,
                getstate_setstate=getstate_setstate,
                on_setattr=on_setattr,
                field_transformer=field_transformer,
                match_args=match_args,
                unsafe_hash=unsafe_hash,
            )(
                cls
            )  # type: ignore

            @classmethod
            def __get_pydantic_core_schema__(cls, source, handler):
                return get_attrs_cls_pydantic_schema(cls)

            @classmethod
            def __get_pydantic_json_schema__(cls, _, handler):
                if "__json_schema_cache__" in cls.__dict__:
                    return cls.__json_schema_cache__
                from .convertors import get_pydantic_type_adapter, recursive_dump_to_basic_types
                from .checking import _save_isinstance

                fields: tuple[Attribute, ...] = cls.__attrs_attrs__  # type: ignore
                cls_annos = get_cls_annotations(cls, no_cls_var=True, no_final=True)
                cls_docs = get_doc(cls)
                field_docs = cls_docs.field_docs
                _null = object()

                def get_ref_name(t):
                    return get_cls_name(t, with_module_name=False, no_qualname=True)

                cls_name = get_ref_name(cls)
                cls_ref = core_schema.definition_reference_schema(schema_ref=cls_name)
                refs = {}

                def is_ref_t(t):
                    if getattr(t, "__serializable_attrs_decorated__", False):
                        return True
                    if isinstance(t, (BaseModelV1, BaseModelV2)):
                        return True
                    return False

                def get_core_schema(t, cls):
                    cls_name = get_cls_name(cls, with_module_name=False, no_qualname=True)
                    t_origin = get_origin(t)
                    while t_origin == Annotated:
                        t = get_args(t)[0]
                        t_origin = get_origin(t)

                    def get_single_t_core_schema(t):
                        if (t == cls_name) or (t == cls) or (t == Self):
                            _logger.warning(
                                f"Class {cls_name} has Self reference type, which is not supported to generate json schema."
                            )
                            return _null
                        if is_ref_t(t):
                            t_ref = get_ref_name(t)
                            if t_ref not in refs:
                                refs[t_ref] = get_pydantic_type_adapter(t).core_schema.copy()
                                refs[t_ref]["ref"] = t_ref
                            return core_schema.definition_reference_schema(t_ref)
                        else:
                            return get_pydantic_type_adapter(t).core_schema

                    if t_origin in (Union, UnionType):
                        arg_schemas = [get_single_t_core_schema(arg) for arg in get_args(t)]
                        tidied_arg_schemas = []
                        for a in arg_schemas:
                            if a == _null:
                                tidied_arg_schemas.append(get_single_t_core_schema(Any))
                            else:
                                tidied_arg_schemas.append(a)
                        arg_schemas = tidied_arg_schemas
                        if not arg_schemas:
                            return get_single_t_core_schema(Any)
                        elif len(arg_schemas) == 1:
                            return arg_schemas[0]
                        return core_schema.union_schema(arg_schemas)
                    else:
                        return get_single_t_core_schema(t)

                def get_field_schema(attr: Attribute):
                    name = attr.alias or attr.name
                    anno = cls_annos.get(attr.name, attr.type)
                    metadata = {}
                    if attr.default != NOTHING:
                        metadata["default"] = recursive_dump_to_basic_types(attr.default)
                    if doc := field_docs.get(attr.name, None):
                        metadata["description"] = doc
                    field_cs = get_core_schema(anno, cls)
                    if field_cs == _null:
                        return None, None, metadata
                    schema = core_schema.typed_dict_field(
                        field_cs,  # type: ignore
                        required=(attr.default == NOTHING),
                    )
                    return name, schema, metadata

                sm = {}
                field_meta_datas = {}
                for attr in fields:
                    name, schema, metadata = get_field_schema(attr)
                    if not name:
                        continue
                    sm[name] = schema
                    field_meta_datas[name] = metadata

                json_schema = core_schema.definitions_schema(
                    schema=core_schema.typed_dict_schema(sm, ref=cls_name, cls=cls),
                    definitions=list(refs.values()),
                )
                json_schema = handler(json_schema)
                json_schema = handler.resolve_ref_schema(json_schema)
                json_schema["title"] = cls_name
                if properties := json_schema.get("properties", None):
                    for k, v in field_meta_datas.items():
                        if k in properties:
                            properties[k].update(v)
                cls.__json_schema_cache__ = json_schema
                return json_schema

            setattr(decorated_cls, "__get_pydantic_core_schema__", __get_pydantic_core_schema__)
            setattr(decorated_cls, "__get_pydantic_json_schema__", __get_pydantic_json_schema__)
            setattr(decorated_cls, "__serializable_attrs_decorated__", True)
            return decorated_cls

        if isinstance(maybe_cls, type):
            return wrapper(maybe_cls)
        return wrapper


def attrs_cls_has_field(cls, field_name: str, fuzzy=True):
    """
    Check if the attrs class has the field with the given name.
    Return False directly if the class is not a attrs class.

    Args:
        cls: the class to be checked.
        field_name: the field name to be checked.
        fuzzy: if True, will do fuzzy match for the field name, i.e.
                `HelloWorld`= `hello_world`
    """
    if not hasattr(cls, "__attrs_attrs__"):
        return False
    if fuzzy:
        def _fuzzy_compare(a: str, b: str) -> bool:
            a = a.lower().replace(' ', '').replace('_', '').replace('-', '')
            b = b.lower().replace(' ', '').replace('_', '').replace('-', '')
            return a == b

        match = lambda attr: _fuzzy_compare(attr.alias, field_name)
    else:
        match = lambda attr: attr.alias == field_name
    for attr in cls.__attrs_attrs__:
        if match(attr):
            return True
    return False


__base_model_fields_aliases__: dict[type, dict[str, tuple[str, ...]]] = {}


def get_pydantic_model_field_aliases(
    cls: type[BaseModelV2] | BaseModelV2 | type[BaseModelV1] | BaseModelV1, field: str
) -> tuple[str, ...]:
    """
    Get all possible values of a field with AliasChoices.
    The first value is the original field name.
    Note: all `AliasPath` object will be ignored.
    """
    if not isinstance(cls, type):
        cls = type(cls)
    if cls not in __base_model_fields_aliases__:
        cache = {}
        __base_model_fields_aliases__[cls] = cache
    else:
        cache = __base_model_fields_aliases__[cls]

    if issubclass(cls, BaseModelV1):
        if field not in cls.__fields__:  # type: ignore
            raise ValueError(f"{field} not found in {cls}")
        if field not in cache:
            model_fields = cls.__fields__
            field_info = model_fields[field]  # type: ignore
            name = field_info.field_info.alias or field
            cache[field] = (name,)

    elif issubclass(cls, BaseModelV2):
        if field not in cls.model_fields:
            raise ValueError(f"{field} not found in {cls}")
        if field not in cache:
            model_fields = cls.model_fields
            field_info = model_fields[field]
            aliases = field_info.validation_alias
            if aliases:
                if isinstance(aliases, str):
                    aliases = (aliases,)
                if isinstance(aliases, AliasChoices):
                    tidied = []
                    for c in aliases.choices:
                        if isinstance(c, str):
                            tidied.append(c)
                    aliases = tuple(tidied)
                else:
                    aliases = (field,)
            else:
                aliases = (field,)
            cache[field] = aliases

    return cache[field]


def pydantic_field_has_default(
    model: type[BaseModelV2] | BaseModelV2 | type[BaseModelV1] | BaseModelV1, field: str
) -> bool:
    """check whether a field in pydantic model has default value or not."""
    from pydantic.v1.fields import Undefined as PydanticUndefinedV1

    if not isinstance(model, type):
        model = type(model)
    if issubclass(model, BaseModelV1):
        field_info = model.__fields__[field]  # will raise error if not found
        if field_info.default == PydanticUndefinedV1 and field_info.default_factory == PydanticUndefinedV1:
            return False
    else:
        field_info = model.model_fields[field]  # will raise error if not found
        if field_info.default == PydanticUndefined and field_info.default_factory == PydanticUndefined:
            return False
    return True

_T = TypeVar("_T")

def create_pydantic_core_schema(
    validator: Callable[[Any], _T], 
    serializer: Callable[[_T], BasicType] | None = None,
    schema_model: type[BaseModelV1]|type[BaseModelV2]|None = None,
):
    """
    Alias of `core_schema.json_or_python_schema` for pydantic v2.
    NOTE: this function should only be used under a `__get_pydantic_core_schema__` classmethod.
    
    Example:
    ```python
    from pydantic import BaseModel
    from utils.common.type_utils.type_helpers import create_pydantic_core_schema
    
    class A:
        x: int
        y: str
    
        @classmethod
        def __get_pydantic_core_schema__(cls, source, handler):
            class ASchema(BaseModel):
                x: int
                y: str
                
            return create_pydantic_core_schema(
                validator=lambda data: (A(x=data["x"], y=data["y"]) if isinstance(data, dict) else data),
                serializer=lambda a: {"x": a.x, "y": a.y},
                schema_model=ASchema,
            )
    ```
    """
    validate_schema = core_schema.no_info_after_validator_function(validator, core_schema.any_schema())  # type: ignore
    if serializer is not None:
        serialize_schema = core_schema.plain_serializer_function_ser_schema(serializer)  # type: ignore
    else:
        serialize_schema = None
    if schema_model:
        if issubclass(schema_model, BaseModelV1):
            fields = {}
            for f in schema_model.__fields__.values():
                if f.default != PydanticV1Undefined:
                    fields[f.name] = (f.type_, f.default)
                else:
                    fields[f.name] = f.type_
            v2_config_keys = set(ConfigDict.__annotations__.keys())
            origin_configs = {k:v for k,v in schema_model.__config__.__dict__.items() if not k.startswith("_")}
            configs = {k: v for k, v in origin_configs.items() if k in v2_config_keys}
            configs = ConfigDict(**configs)
            v2_model = create_model(schema_model.__name__, __config__=configs, **fields)    
            json_schema = v2_model.__pydantic_core_schema__
        elif issubclass(schema_model, BaseModelV2):
            json_schema = schema_model.__pydantic_core_schema__
        else:
            raise ValueError(f"`schema_model` must be a pydantic BaseModel class, got {schema_model}.")
    else:
        json_schema = validate_schema
    return core_schema.json_or_python_schema(
        json_schema=json_schema,  # type: ignore
        python_schema=validate_schema,  # type: ignore
        serialization=serialize_schema,  # type: ignore
    )


def get_json_schema(t: type | str) -> dict[str, Any]:
    """Return the json schema dict of the given type"""
    if isinstance(t, type) and issubclass(t, BaseModelV1):
        return t.schema()
    else:
        from .convertors import get_pydantic_type_adapter

        return get_pydantic_type_adapter(t).json_schema()


__all__.extend(
    [
        "get_original_bases",
        "get_origin",
        "get_args",
        "get_cls_name",
        "get_mro_distance",
        "get_module_name",
        "get_sub_clses",
        "get_cls_annotations",
        "is_builtin",
        "getattr_raw",
        "Empty",
        "getmro",
        "is_attrs_cls",
        "is_dataclass",
        "get_attr_cls_default_pydantic_validator",
        "get_attr_cls_pydantic_validator",
        "get_attrs_cls_pydantic_schema",
        "serializable_attrs",
        "attrs_cls_has_field",
        "get_pydantic_model_field_aliases",
        "pydantic_field_has_default",
        "create_pydantic_core_schema",
        "get_json_schema",
    ]
)