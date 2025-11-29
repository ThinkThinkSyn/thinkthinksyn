import types
import typing
import logging
import inspect
import annotated_types
import typing_extensions

from pydoc import locate
from functools import cache
from types import UnionType, GenericAlias
from typeguard import check_type as tg_check_type, TypeCheckError
from typing import (
    Any,
    Sequence,
    Union,
    ForwardRef,
    Iterable,
    Mapping,
    Literal,
    ClassVar,
    ParamSpec,
    TypeVar,
    overload,
    Annotated,
    Final,
    Protocol,
    Callable,
    no_type_check,
    get_args as tp_get_args,
    get_origin as tp_get_origin,
    TypeVarTuple,
    _GenericAlias, # type: ignore
    _LiteralGenericAlias,  # type: ignore
    _AnnotatedAlias, # type: ignore
    _UnionGenericAlias, # type: ignore
    _UnpackGenericAlias,  # type: ignore
    _CallableGenericAlias,  # type: ignore
)  # type: ignore
from typing_extensions import TypeIs, TypeForm, TypeAliasType
from collections.abc import (Callable as CallableType, Mapping as ABCMapping, MutableMapping as ABCMutableMapping)

_logger = logging.getLogger(__name__)
_empty = inspect.Parameter.empty
_check_sub_cls_cache: dict[tuple[str, str], bool] = dict()  # type: ignore
_T = TypeVar('_T')

def _save_isinstance(v, t):
    try:
        return isinstance(v, t)
    except:
        return False

def _save_issubclass(sub_cls, super_cls):
    try:
        return issubclass(sub_cls, super_cls)
    except:
        return False

@overload
def get_type_from_str(t: str) -> type | str: ...
@overload
def get_type_from_str(t: str, raise_err: Literal[True] = True) -> type | str: ...
@overload
def get_type_from_str(t: str, raise_err: Literal[False] = False) -> type: ...

_FAIL_TO_GET_TYPE_FROM_STR = object()  # type: ignore
_TypingGlobalDict = None

def _get_typing_global_dict():
    global _TypingGlobalDict
    if _TypingGlobalDict is None:
        _TypingGlobalDict = typing.__dict__.copy()
        _TypingGlobalDict.update(types.__dict__)
        _TypingGlobalDict.update(typing_extensions.__dict__)
        _TypingGlobalDict.update(annotated_types.__dict__)
    return _TypingGlobalDict

def _get_module_name(t):
    if not isinstance(t, str):
        if hasattr(t, "__module__"):
            module = t.__module__
        elif not isinstance(t, type) and hasattr(type(t), "__module__"):
            module = type(t).__module__
        else:
            raise ValueError(f"Cannot get module name of {t}.")
    else:
        module = t

    if module == "__main__":
        from pathlib import Path
        import __main__ as _main
        main_path = Path(_main.__file__).resolve()
        source_path = main_path.resolve()   # assume `src` dir is the source dir
        try:
            module = (
                main_path.relative_to(source_path)
                .with_suffix("")
                .as_posix()
                .replace("/", ".")
            )
        except ValueError:
            module = "__main__"  # not in source dir, use __main__ instead
    return module

def _globals_locals_types():
    def is_type_form(t):
        return _save_isinstance(t, (type, TypeAliasType, _GenericAlias, GenericAlias))
    def only_types(d: dict):
        tidied = {}
        for k, v in d.items():
            if is_type_form(v):
                tidied[k] = v
        return tidied
    merged_globals = {}
    merged_locals = {}
    stack = inspect.stack()
    if len(stack) <= 3:
        return merged_globals, merged_locals
    merged_modules = set()
    for frame_info in stack[3:][::-1]:
        frame = frame_info.frame
        module_name = frame.f_globals.get('__name__', None)
        if module_name:
            module_name = _get_module_name(module_name)
        if module_name and module_name not in merged_modules:
            merged_modules.add(module_name)
            merged_globals.update(only_types(frame.f_globals))
        merged_locals.update(only_types(frame.f_locals))
    return merged_globals, merged_locals

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
        global_type_dict = _get_typing_global_dict().copy()
        global_types, local_types = _globals_locals_types()
        global_type_dict.update(global_types)
        return eval(t, global_type_dict, local_types)
    except:
        ...
    return _FAIL_TO_GET_TYPE_FROM_STR

@no_type_check
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


def _is_protocol_type(t):
    if isinstance(t, str):
        return False
    return _save_issubclass(t, Protocol)

def _check_qualname_eq_without_main(a: str, b: str):
    if (not a.startswith('__main__') and not b.startswith('__main__')) or \
        (a.startswith('__main__') and b.startswith('__main__')):
        return a == b
    have_main = a if a.startswith('__main__') else b
    no_main = b if a.startswith('__main__') else a
    have_main_remaining = have_main[len('__main__.'):]
    no_main_remaining = no_main[-len(have_main_remaining):]
    if have_main_remaining == no_main_remaining:
        return True
    return False

@no_type_check
def _direct_check_sub_cls(sub_cls: type | str, super_cls: type | str):
    from .type_helpers import get_cls_name, get_sub_clses, get_origin, get_args
    
    sub_cls, super_cls = _tidy_type(sub_cls)[0], _tidy_type(super_cls)[0]  # type: ignore
    should_cache = _save_isinstance(sub_cls, type) and _save_isinstance(super_cls, type)

    # find cache
    if should_cache:
        sub_cls_name_for_cache = get_cls_name(sub_cls, with_module_name=True, with_generic=True)
        super_cls_name_for_cache = get_cls_name(super_cls, with_module_name=True, with_generic=True)
        if _save_isinstance(sub_cls, str):
            sub_cls_name_for_cache += "_str"
        if _save_isinstance(super_cls, str):
            super_cls_name_for_cache += "_str"
        if (sub_cls_name_for_cache, super_cls_name_for_cache) in _check_sub_cls_cache:
            return _check_sub_cls_cache[(sub_cls_name_for_cache, super_cls_name_for_cache)]

    result: bool | None = None

    if super_cls in (Any, any):
        result = True
    
    sub_cls_origin = get_origin(sub_cls)
    super_cls_origin = get_origin(super_cls)
        
    wrap_types = (ClassVar, Final, Annotated)
    if result is None and (sub_cls_origin in wrap_types or super_cls_origin in wrap_types):
        sub_cls = get_args(sub_cls)[0] if sub_cls_origin in wrap_types else sub_cls
        super_cls = get_args(super_cls)[0] if super_cls_origin in wrap_types else super_cls
        result = _direct_check_sub_cls(sub_cls, super_cls)

    if result is None and (sub_cls_origin == Union or super_cls_origin == Union):
        if sub_cls_origin == Union and super_cls_origin != Union:
            result = all([_direct_check_sub_cls(arg, super_cls) for arg in tp_get_args(sub_cls)])  # type: ignore
    
        elif (super_cls_origin == Union and sub_cls_origin != Union):
            result = any([_direct_check_sub_cls(sub_cls, arg) for arg in tp_get_args(super_cls)])  # type: ignore
        
        else:   # both Union
            sub_cls_args = tp_get_args(sub_cls)
            super_cls_args = tp_get_args(super_cls)
            if not sub_cls_args and not super_cls_args:
                result = True
            elif not sub_cls_args or not super_cls_args:
                result = False
            else:
                for sa in sub_cls_args:
                    if any(_direct_check_sub_cls(sa, ua) for ua in super_cls_args):
                        result = True
                        break

        # special treatment for Protocol
    if result is None and _is_protocol_type(super_cls) and super_cls != Protocol:
        if _save_isinstance(sub_cls, str):
            result = False
        elif _save_issubclass(sub_cls, super_cls) and sub_cls != Protocol:
            result = True
        else:
            checked = set()
            # check annotations
            if not super_cls.__callable_proto_members_only__:  # type: ignore
                from .type_helpers import get_cls_annotations

                super_cls_anno = get_cls_annotations(super_cls)
                sub_cls_anno = get_cls_annotations(sub_cls)

                for attr_name, anno in super_cls_anno.items():
                    checked.add(attr_name)
                    if attr_name in sub_cls_anno:
                        if not _direct_check_sub_cls(sub_cls_anno[attr_name], anno):
                            result = False
                            break
                    else:
                        attr = getattr(sub_cls, attr_name, _empty)
                        if attr == _empty:
                            result = False
                            break
                        if not check_value_is(attr, anno):
                            result = False
                            break

            # check methods
            if result is None:
                for attr_name in super_cls.__protocol_attrs__:  # type: ignore
                    protocol_attr = getattr(super_cls, attr_name)
                    if attr_name in checked:
                        continue
                    if not callable(protocol_attr):
                        continue
                    sub_cls_attr = getattr(sub_cls, attr_name, _empty)
                    if not callable(sub_cls_attr):
                        result = False
                        break

                    try:
                        protocol_attr_sig = inspect.signature(protocol_attr)
                    except:
                        continue
                    try:
                        sub_cls_attr_sig = inspect.signature(sub_cls_attr)
                    except:
                        result = False
                        break

                    if len(protocol_attr_sig.parameters) != len(sub_cls_attr_sig.parameters):
                        result = False
                        break
                    for p, s in zip(protocol_attr_sig.parameters.values(), sub_cls_attr_sig.parameters.values()):
                        p_anno, s_anno = p.annotation, s.annotation
                        if p_anno is _empty or s_anno is _empty:
                            continue
                        if not _direct_check_sub_cls(s.annotation, p.annotation):
                            result = False
                            break

                    if result is None:
                        p_ret_anno, s_ret_anno = protocol_attr_sig.return_annotation, sub_cls_attr_sig.return_annotation
                        if p_ret_anno is not _empty and s_ret_anno is not _empty:
                            if not _direct_check_sub_cls(s_ret_anno, p_ret_anno):
                                result = False
                                break
                if result is None:
                    result = True

    # special treatment for TypeVar
    if result is None and (type(sub_cls) == TypeVar or type(super_cls) == TypeVar):
        if type(sub_cls) == TypeVar and type(super_cls) != TypeVar:
            sub_type_constraints = sub_cls.__constraints__
            for sub_constraint in sub_type_constraints:  # type: ignore
                if _direct_check_sub_cls(sub_constraint, super_cls):
                    result = True
                    break
            else:
                result = False
        elif type(sub_cls) != TypeVar and type(super_cls) == TypeVar:
            super_type_constraints = super_cls.__constraints__  # type: ignore
            for super_constraint in super_type_constraints:  # type: ignore
                if _direct_check_sub_cls(sub_cls, super_constraint):
                    result = True
                    break
            else:
                result = False
        else:
            if sub_cls.__bound__:
                sub_type_constraints = [sub_cls.__bound__]  # type: ignore
            else:
                sub_type_constraints = sub_cls.__constraints__
            if super_cls.__bound__:
                super_type_constraints = [super_cls.__bound__]
            else:
                super_type_constraints = super_cls.__constraints__
            if (not sub_type_constraints) and super_type_constraints:
                if len(super_type_constraints) == 1 and super_type_constraints[0] in (
                    Any,
                    "Any",
                    "typing.Any",
                ):
                    result = True
                else:
                    result = False
            elif sub_type_constraints and (not super_type_constraints):
                result = True  # i.e. `T: some_type` to `T: Any`
            else:
                for sub_constraint in sub_type_constraints:  # type: ignore
                    for super_constraint in super_type_constraints:
                        if _direct_check_sub_cls(sub_constraint, super_constraint):
                            break
                    else:
                        result = False
                        break
                else:
                    result = True  # sub constraints are all in super constraints

    if result is None and (sub_cls_origin == CallableType and super_cls_origin == CallableType):
        sub_cls_types = get_args(sub_cls)
        super_cls_types = get_args(super_cls)
        if not super_cls_types:
            result = True  # no type hint for super cls
        elif not sub_cls_types:
            result = False  # no type hint for sub cls
        else:
            sub_cls_params, sub_cls_ret = sub_cls_types
            super_cls_params, super_cls_ret = super_cls_types
            if sub_cls_params == Ellipsis:
                result = False
            elif sub_cls_params == Ellipsis:
                result = False
            elif len(sub_cls_params) != len(super_cls_params):
                result = False
            else:
                for sub_param, super_param in zip(sub_cls_params, super_cls_params):
                    if not _direct_check_sub_cls(sub_param, super_param):
                        result = False
                        break
                else:
                    if not _direct_check_sub_cls(sub_cls_ret, super_cls_ret):
                        result = False
                    else:
                        result = True

    # check Literal case, e.g. check_type_is(Literal[1, 2], Literal[1, 2, 3]) -> True
    if result is None and (sub_cls_origin == Literal or super_cls_origin == Literal):
        # Literal to Literal
        if super_cls_origin == Literal and sub_cls_origin == Literal:
            super_args = get_args(super_cls)
            for arg in get_args(sub_cls):
                if arg not in super_args:
                    result = False
                    break
            else:
                result = True
        # Literal to other types
        elif sub_cls_origin == Literal and super_cls_origin != Literal:
            lit_arg_types = [type(arg) for arg in tp_get_args(sub_cls)]
            if not lit_arg_types:
                result = False
            else:
                for t in lit_arg_types:
                    if not _direct_check_sub_cls(t, super_cls):  # type: ignore
                        result = False
                        break
                else:
                    result = True
        else:   # super_cls_origin == Literal and sub_cls_origin != Literal
            result = False
        
    if result is None and _save_isinstance(sub_cls, ForwardRef):
        result = _direct_check_sub_cls(sub_cls.__forward_arg__, super_cls)  # type: ignore

    if result is None and _save_isinstance(super_cls, ForwardRef):
        result = _direct_check_sub_cls(sub_cls, super_cls.__forward_arg__)  # type: ignore

    if result is None and _save_isinstance(sub_cls, str) and not _save_isinstance(super_cls, str):  
        # check str to type
        result = sub_cls in [get_cls_name(cls) for cls in get_sub_clses(super_cls)]

    if result is None:
        if not _save_isinstance(sub_cls, str) and _save_isinstance(super_cls, str):
            from .type_helpers import getmro

            result = super_cls in [get_cls_name(cls) for cls in getmro(sub_cls)]  # type: ignore

        elif _save_isinstance(sub_cls, str) and _save_isinstance(super_cls, str):
            if sub_cls == super_cls:
                result = True
            elif "[" in sub_cls and "[" in super_cls:  # type: ignore
                # e.g. `list[int]` to `list`
                result = sub_cls.split("[")[0] == super_cls  # type: ignore
            raise TypeError(f"Sub cls and super cls cannot both be str: sub_cls: {sub_cls}, super_cls: {super_cls}. There should be at least one type.")

    # check directly
    if result is None:
        sub_cls_for_compare = sub_cls_origin or sub_cls
        super_cls_for_compare = super_cls_origin or super_cls
        try:
            result = issubclass(sub_cls_for_compare, super_cls_for_compare)  # type: ignore
            if result:
                sub_cls_args, super_cls_args = get_args(sub_cls), get_args(super_cls)
                if sub_cls_args == super_cls_args:
                    result = True
                elif len(super_cls_args) == 2 and super_cls_args[-1] == Ellipsis:  # e.g. tuple[int, ...]
                    for sub_arg in sub_cls_args:
                        if not _save_issubclass(sub_arg, super_cls_args[0]):
                            result = False
                            break
                elif len(sub_cls_args) != len(super_cls_args):
                    if len(super_cls_args) != 0:  # e.g. `list[int]` to `list`
                        result = False
                elif sub_cls_args and super_cls_args:
                    for sub_arg, super_arg in zip(sub_cls_args, super_cls_args):
                        if not _direct_check_sub_cls(sub_arg, super_arg):
                            result = False
                            break
            else:
                if _save_isinstance(sub_cls, type) and _save_isinstance(super_cls, type):
                    if sub_cls.__name__ == super_cls.__name__:  # for fixing problem in non-main-mode
                        result = _check_qualname_eq_without_main(sub_cls.__qualname__, super_cls.__qualname__)

        except TypeError as e:
            if str(e).startswith("issubclass() arg 1 must be a class"):
                raise TypeError(f"Invalid sub class type: `{sub_cls}`(type of sub cls input={type(sub_cls)}).") from e
            elif str(e).startswith("issubclass() arg 2 must be"):
                raise TypeError(f"Invalid super class type: `{super_cls}`(type of super cls input={type(super_cls)}).") from e
            else:
                raise TypeError(f"(_direct_check_sub_cls) Error occurs on comparing `{sub_cls}` & `{super_cls}`: {e}")
        except Exception as e:
            raise e

    if result is None:
        result = False
    if should_cache:
        _check_sub_cls_cache[(sub_cls_name_for_cache, super_cls_name_for_cache)] = result
    return result

@overload
def check_value_is(value: Any, types: TypeForm[_T]) -> TypeIs[_T]: ...
@overload
def check_value_is(value: Any, types: type[_T]) -> TypeIs[_T]: ...
@overload
def check_value_is(value: Any, types: str | Sequence[type | str] | UnionType | TypeAliasType) -> bool: ...
@overload
def check_value_is(value: Any, types: Any) -> bool: ...

@no_type_check
def check_value_is(value: Any, types):
    """
    Check value with given types. Advance version of `isinstance`.
    support passing special types in `typing` module, e.g. Union, Literal, TypedDict, etc.

    Example 1:
    ```
    check_value_is([1,2,'abc',1.23], list[int|str|float])
    check_value_is([1,2], Any)
    check_value_is(1, Literal[1, 2])
    check_value_is(1, Union[int, str])
    check_value_is(1, int | str)
    check_value_is(1, Annotated[int, 123])
    ```

    Example 2:
    ```
    class A:
        pass
    a = A()
    check_value_is(a, 'A') # True, accept class name
    ```
    """
    from .type_helpers import get_origin, get_args, getattr_raw
    _empty = inspect.Parameter.empty

    if _save_isinstance(
        types,
        (
            str,
            TypeAliasType,
            GenericAlias,
            _UnionGenericAlias,
            _UnpackGenericAlias,
            _CallableGenericAlias,
        ),
    ):
        types = _tidy_type(types)[0]  # type: ignore

    if not _save_isinstance(types, str) and _save_isinstance(types, Sequence):
        return any(check_value_is(value, t) for t in types)

    types = _tidy_type(types)[0]  # type: ignore

    if _save_isinstance(types, str):
        return _direct_check_sub_cls(type(value), types)

    elif (origin := get_origin(types)) and origin not in (
        None,
        Union,
        Iterable,
        Literal,
    ):
        # None: means no origin, e.g. list
        # Union/UnionType: means union, e.g.: Union[int, str], int | str
        # Iterable: checking inner type of Iterable is meaningless, since it will destroy the structure of Iterable

        # list, tuple, ...
        if _save_issubclass(origin, Sequence):
            if not _save_issubclass(origin, tuple):
                args = get_args(types)
                if len(args) == 0:  # no args, e.g. check_value_is([1,2], list)
                    return _save_isinstance(value, origin)
                else:
                    return _save_isinstance(value, origin) and all(
                        check_value_is(v, args[0]) for v in value
                    )
            else:
                args = get_args(types)
                if len(args) == 0:
                    return _save_isinstance(value, origin)
                elif len(args) == 2 and args[-1] == Ellipsis:
                    return _save_isinstance(value, origin) and all(
                        check_value_is(v, args[0]) for v in value
                    )
                else:
                    return (
                        _save_isinstance(value, origin)
                        and len(value) == len(get_args(types))
                        and all(
                            check_value_is(v, t) for v, t in zip(value, get_args(types))
                        )
                    )  # type: ignore

        # dict
        elif _save_issubclass(origin, Mapping):
            mapping_args = get_args(types)
            if not (len(mapping_args) == 2 and origin in (dict, ABCMapping, ABCMutableMapping)):
                return _save_isinstance(value, origin)
            else:
                return (
                    _save_isinstance(value, origin)
                    and all(
                        check_value_is(v, get_args(types)[1]) for v in value.values()
                    )
                    and all(check_value_is(k, get_args(types)[0]) for k in value.keys())
                )

        # check Callable
        elif _save_issubclass(origin, Callable):
            if not callable(value):
                return False
            params, return_t = get_args(types)
            val_sig = inspect.signature(value)
            if params != Ellipsis:  # no need to check params for `Callable[..., Any]`
                if len(val_sig.parameters) != len(params):
                    return False
                for p, param in zip(val_sig.parameters.values(), params):
                    if not check_type_is(p.annotation, param):
                        return False
            if return_t is not _empty and val_sig.return_annotation is not _empty:
                if not check_type_is(val_sig.return_annotation, return_t):
                    return False
            return True

        else:
            try:
                tg_check_type(value, types)
                return True
            except TypeCheckError:
                return False

    # check Protocol
    elif _is_protocol_type(types):
        checked = set()
        # check annotations
        if not types.__callable_proto_members_only__:  # type: ignore
            from .type_helpers import get_cls_annotations

            type_anno = get_cls_annotations(types)
            for attr_name, anno in type_anno.items():
                checked.add(attr_name)
                attr = getattr(value, attr_name, _empty)
                if attr == _empty:
                    return False
                if not check_value_is(attr, anno):
                    return False

        # check methods
        for attr_name in types.__protocol_attrs__:  # type: ignore
            protocol_attr = getattr(types, attr_name)
            if attr_name in checked:
                continue
            if not callable(protocol_attr):
                continue
            val_attr = getattr_raw(value, attr_name, _empty)
            if not callable(val_attr):
                return False

            protocol_attr_sig = inspect.signature(protocol_attr)
            val_attr_sig = inspect.signature(val_attr)
            protocol_attr_params = protocol_attr_sig.parameters.copy()
            val_attr_params = val_attr_sig.parameters.copy()

            if len(protocol_attr_params) != len(val_attr_params):
                return False
            for protocol_param, val_param in zip(
                protocol_attr_params.values(), val_attr_params.values()
            ):
                p_anno, s_anno = protocol_param.annotation, val_param.annotation
                if p_anno is _empty or s_anno is _empty:
                    continue
                if not _direct_check_sub_cls(
                    val_param.annotation, protocol_param.annotation
                ):
                    return False

            p_ret_anno, val_ret_anno = (
                protocol_attr_sig.return_annotation,
                val_attr_sig.return_annotation,
            )
            if p_ret_anno is not _empty and val_ret_anno is not _empty:
                if not _direct_check_sub_cls(val_ret_anno, p_ret_anno):
                    return False
        return True

    else:
        try:
            tg_check_type(value, types)
            return True
        except TypeCheckError:
            vt = type(value)
            if _save_isinstance(vt, type) and _save_isinstance(types, type):
                if vt.__name__ == types.__name__:   # for fixing problem in non-main-mode
                    return _check_qualname_eq_without_main(vt.__qualname__, types.__qualname__)
            return False

_TT = TypeVar("_TT", bound=type)

@overload
def check_type_is(sub_cls: Any, super_cls: _TT) -> TypeIs[_TT]: ...
@overload
def check_type_is(sub_cls: Any, super_cls: Any | Sequence[Any]) -> bool: ...
@no_type_check
def check_type_is(sub_cls, super_cls):
    """
    Check if sub_cls is a subclass of super_cls.
    You could use `|` to represent Union, e.g.: `check_type_is(sub_cls, int | str)`
    You could also use list to represent Union, e.g.: `check_type_is(sub_cls, [int, str])`
    Class name is also supported, e.g.: `check_type_is(sub_cls, 'A')`
    """
    from .type_helpers import get_cls_name, get_sub_clses

    def tidy(t):
        tt = _tidy_type(t)
        try:
            return tt[0]  # type: ignore
        except TypeError:
            return tt
    
    try:
        sub_cls = _tidy_type(sub_cls)[0]  # type: ignore
    except Exception as e:
        _logger.warning(f'Error when tidying sub_cls `{sub_cls}`: {type(e).__name__}: {str(e)}')
    
    if isinstance(super_cls, Sequence) and not _save_isinstance(super_cls, str):
        super_cls = [tidy(t) for t in super_cls]  # type: ignore
    else:
        super_cls = tidy(super_cls)  # type: ignore

    if _save_isinstance(sub_cls, str):
        if _save_isinstance(super_cls, str):
            return sub_cls.split(".")[-1] == super_cls.split(".")[-1]
        else:
            all_super_cls_names = []
            if _save_isinstance(super_cls, Sequence):
                for c in super_cls:
                    if hasattr(c, "__subclasses__"):
                        all_super_cls_names.extend([get_cls_name(cls) for cls in get_sub_clses(c)])  # type: ignore
                    else:
                        all_super_cls_names.append(get_cls_name(c))
            else:
                if hasattr(super_cls, "__subclasses__"):
                    all_super_cls_names = [get_cls_name(cls) for cls in get_sub_clses(super_cls)]  # type: ignore
                else:
                    all_super_cls_names = [get_cls_name(super_cls)]

            return sub_cls.split(".")[-1] in all_super_cls_names

    if not _save_isinstance(super_cls, str) and _save_isinstance(super_cls, Sequence):
        return any(_direct_check_sub_cls(sub_cls, t) for t in super_cls)
    try:
        return _direct_check_sub_cls(sub_cls, super_cls)
    except TypeError as e:
        _logger.debug(
            f"Error when checking sub class `{sub_cls}` and super class `{super_cls}`. {type(e).__name__}: {str(e)}"
        )
        return False


__all__ = ["check_value_is", "check_type_is", "get_type_from_str"]