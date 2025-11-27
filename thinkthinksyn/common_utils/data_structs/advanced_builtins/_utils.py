import typing
from typing import (TypeVar, ClassVar, TypeAliasType, get_origin as tp_get_origin, get_args as tp_get_args, TypeVarTuple,
                    is_typeddict, Final)
from functools import cache

def _save_isinstance(v, t):
    try:
        return isinstance(v, t)
    except:
        return False

_TypingGlobalDict = typing.__dict__.copy()
_TypingGlobalDict.update(types.__dict__)
_TypingGlobalDict.update(typing_extensions.__dict__)
_TypingGlobalDict.update(annotated_types.__dict__)

@cache
def _get_type_from_str(t: str):
    t = t.strip()
    try:
        if t == "...":
            return Ellipsis
        if t == "None":
            return None
        if real_t := locate(t):
            return real_t
        global_type_dict = _TypingGlobalDict.copy()
        global_types, local_types = _globals_locals_types()
        global_type_dict.update(global_types)
        return eval(t, global_type_dict, local_types)
    except:
        ...
    return _FAIL_TO_GET_TYPE_FROM_STR

def get_type_from_str(t: str, raise_err: bool = False):  # type: ignore
    """
    Try to get type from given string,
    e.g. "list[int]" -> list[int].
    If no type can be gotten, return the original string when `raise_err` is False,
    else raise TypeError.
    """
    if not _save_isinstance(t, str):
        return t
    try_t = _get_type_from_str(t)
    if try_t is _FAIL_TO_GET_TYPE_FROM_STR:
        if raise_err:
            raise TypeError(f"Cannot get type from string: {t}.")
        else:
            return t
    return try_t

@no_type_check
def _tidy_type(t, arg_matches: dict | None = None, _rec=False):
    arg_matches = arg_matches or {}
    if _save_isinstance(t, str):
        t = get_type_from_str(t)
        if not _save_isinstance(t, str):
            new_t = _tidy_type(t, arg_matches, _rec=True)
            if not _rec and len(new_t) > 1:
                return (t,)
            return new_t
        return (t,)  # type: ignore
    
    try:
        if t in arg_matches:
            v = arg_matches[t]
            if isinstance(v, tuple):
                if len(v) > 1 and not _rec:
                    return (t,)
                return v
            return (v,)  # type: ignore
    except:
        pass
    
    if isinstance(t, TypeAliasType):
        if not t.__type_params__:
            new_t = _tidy_type(t.__value__, arg_matches, _rec=True)  # type: ignore
            if not _rec and len(new_t) > 1:
                return (t.__value__,)
            return new_t  # type: ignore
        else:
            tidied = []
            for a in t.__type_params__:
                tidied.extend(_tidy_type(a, arg_matches, _rec=True))  # type: ignore
            if tidied:
                o = t.__value__.__origin__ or t.__value__
                return (o[*tidied],)
            return (t,)

    if isinstance(t, _UnionGenericAlias):
        tidied = []
        for a in tp_get_args(t):
            tidied.extend(_tidy_type(a, arg_matches, _rec=True))  # type: ignore
        if tidied:
            return (Union[*tidied],)  # type: ignore
        return (t,)

    if isinstance(t, _UnpackGenericAlias):
        tidied = []
        for a in tp_get_args(t):
            tidied.extend(_tidy_type(a, arg_matches, _rec=True))  # type: ignore
        if _rec:
            return tuple(tidied)
        else:
            if len(tidied) == 1:
                return (tidied[0],)
            else:
                return (t,)

    if isinstance(t, _AnnotatedAlias):
        args = tp_get_args(t)
        first_arg, remaining = args[0], args[1:]
        tidied_first_arg = _tidy_type(first_arg, arg_matches, _rec=True)[0]
        return (Annotated[tidied_first_arg, *remaining],)  # type: ignore
    
    if isinstance(t, _CallableGenericAlias):
        t_args = tp_get_args(t)
        params, ret = t_args
        tidied_ret = _tidy_type(ret, arg_matches, _rec=True)[0]
        if isinstance(params, ParamSpec):
            params = _tidy_type(params, arg_matches, _rec=True)[0]
        if params != Ellipsis and isinstance(params, Sequence):
            tidied_params = []
            for p in params:
                tidied_params.extend(_tidy_type(p, arg_matches, _rec=True))  # type: ignore
            return (Callable[tuple(tidied_params), tidied_ret],)  # type: ignore
        else:
            return (Callable[..., tidied_ret],)  # type: ignore

    if isinstance(t, _LiteralGenericAlias):
        return (t,)  # type: ignore
    
    if isinstance(t, (GenericAlias, _GenericAlias)):
        t_origin, t_args = tp_get_origin(t), tp_get_args(t)
        if t_args:
            tidied = []
            for a in t_args:
                tidied.extend(_tidy_type(a, arg_matches, _rec=True))  # type: ignore
            t_args = tuple(tidied)

        if isinstance(t_origin, TypeAliasType):
            params = t_origin.__type_params__
            if len(params) != len(t_args):
                if len(params) == 1 and isinstance(params[0], TypeVarTuple):
                    t_args = [t_args]
                else:
                    raise ValueError(
                        f"Type parameters {params} do not match arguments {t_args} for {t_origin}"
                    )
            arg_matches.update({p: a for p, a in zip(params, t_args)})  # type: ignore
            new_t = _tidy_type(t_origin.__value__, arg_matches, _rec=True)  # type: ignore
            if not _rec and len(new_t) > 1:
                if not params:
                    return (t_origin.__value__,)
                return (t,)
            return new_t  # type: ignore
        else:
            t_origin = _tidy_type(t_origin, arg_matches, _rec=True)[0]
        if t_args:
            if t_origin in (Final, ClassVar):
                return (t_origin[t_args[0]],)   # type: ignore
            return (t_origin[*t_args],)  # type: ignore
        return (t_origin,)  # type: ignore
    
    return (t,)

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
    
    if args := tp_get_args(cls):
        type_params = getattr(origin, "__type_params__", None)
        if type_params:
            if len(type_params) != len(args):
                if len(type_params) == 1 and isinstance(type_params[0], TypeVarTuple):
                    args = tuple([args])
                else:
                    raise ValueError(f"Type parameters {type_params} do not match arguments {args} for {origin}")
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