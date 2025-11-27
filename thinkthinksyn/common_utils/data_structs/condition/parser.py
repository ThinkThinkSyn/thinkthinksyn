import logging

from json import loads
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING, get_args

from ._utils import get_num_from_text

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .condition import BaseCondition as _BaseCond, Condition as _Cond

_should_log = (__name__ == '__main__')


type BaseConditionOperator = Literal['=', '!=', '>', '<', '>=', '<=', 'in', 'not in', 'contains', 'not contains']
'''
Operators for attribute conditions.
- '=': equal
- '!=': not equal
- '>': greater than
- '<': less than
- '>=': greater than or equal to
- '<=': less than or equal to
- 'in': e.g. in a list
- 'not in'
- 'contains': contains (e.g. contains a substring)
- 'not contains': not contains

NOTE: `==` is will be seen as `=` in the parser, and `not ...` will also be seen as an inverse operation
    to the next operator, e.g. `not !=` equals `=`.
'''

_valid_base_ops = set(get_args(BaseConditionOperator)) | {'==',}
_inverse_base_operator = {
    '=': '!=',
    '!=': '=',
    '>': '<=',
    '<': '>=',
    '>=': '<',
    '<=': '>',
    '==': '!=',
    'in': 'not in',
    'not in': 'in',
    'contains': 'not contains',
    'not contains': 'contains',
}

@dataclass
class _StateMachine:
    states: list[str]
    curr_idx: int = 0
    loop: bool = False
    
    @property
    def curr(self):
        return self.states[self.curr_idx]
    
    def next(self):
        self.curr_idx += 1
        if self.curr_idx >= len(self.states):
            if self.loop:
                self.curr_idx = 0
            else:
                raise StopIteration
        elif _should_log:
            _logger.debug(f'Switching state from `{self.states[self.curr_idx-1]}` to `{self.states[self.curr_idx]}`')
        return self.curr
    
    def prev(self):
        self.curr_idx -= 1
        if self.curr_idx < 0:
            if self.loop:
                self.curr_idx = len(self.states) - 1
            else:
                raise StopIteration
        elif _should_log:
            _logger.debug(f'Switching state back from `{self.states[self.curr_idx+1]}` to `{self.states[self.curr_idx]}`')
        return self.curr

def _find_first_lst(s: str, quote):
    stack = [quote,]
    start_str = None
    buf = ''
    while s:
        c, s = s[0], s[1:]
        buf += c
        if c in ('"', "'"):
            if not start_str:
                start_str = c
            elif start_str == c:
                start_str = None
        elif c in ('[', '('):
            if not start_str:
                stack.append(c)
        elif c in (']', ')'):
            if (stack[-1] == '[' and c != ']') or (stack[-1] == '(' and c != ')'):
                raise ValueError(f"Invalid list format: {s}")
            stack.pop()
            if not stack:
                return buf, s
    raise ValueError(f"Invalid list format: {s}.")

def _load_list(s: str, _inner=False)->tuple[list, str]:
    '''
    load list from string,
    e.g:
        - [a, b, c] -> ['a', 'b', 'c']
        - (1, 2, 3) -> (1, 2, 3)
        - [[a, b], [1, 2]] -> [['a', 'b'], [1, 2]]
    '''
    s = s.strip()
    if not _inner:
        if not (s.startswith("[") and s.endswith("]")) and not (s.startswith("(") and s.endswith(")")):
            raise ValueError(f"Invalid list format: {s}")
    
    origin_s = s    
    s = s[1:-1]
    lst = []
    buf, str_start_char = '', None
    state_machine = _StateMachine(['item', 'comma'], loop=True)
    
    while s:
        c, s= s[0], s[1:]
        if state_machine.curr == 'item':
            match c:
                case ' ':
                    if str_start_char:
                        buf += c
                    else:
                        if buf:
                            lst.append(buf)
                            buf = ''
                            state_machine.next()
                case ',':
                    if not str_start_char:
                        if buf:
                            lst.append(buf)
                            buf = ''
                            s = c + s   # to keep the comma for the next state
                            state_machine.next()
                        else:
                            raise ValueError(f"Invalid list format: {origin_s}")
                    else:
                        buf += c
                case "'" | '"':
                    if str_start_char:
                        if c == str_start_char:
                            str_start_char = None
                            lst.append(buf)
                            buf = ''
                            state_machine.next()
                        else:
                            buf += c
                    else:
                        str_start_char = c
                case '[' | '(':
                    if str_start_char:
                        buf += c
                    else:
                        if buf:
                            raise ValueError(f"Invalid list format: {origin_s}")
                        f, s = _find_first_lst(s, c)
                        inner_list, _ = _load_list(c+f, _inner=True)
                        if _should_log:
                            _logger.debug(f'loaded inner list: {inner_list}')
                        lst.append(inner_list)
                        state_machine.next()
                case ']' | ')':
                    if str_start_char:
                        buf += c
                    else:
                        if _inner:
                            if buf:
                                lst.append(buf)
                                buf = ''
                            break
                        else:
                            raise ValueError(f"Invalid list format: {origin_s}")
                case _:
                    buf += c        
        else:
            match c:
                case ' ':
                    continue
                case ',':
                    state_machine.next()
                case ']' | ')':
                    if _inner:
                        break
                    raise ValueError(f"Invalid list format: {origin_s}")
                case _:
                    raise ValueError(f"Invalid list format: {origin_s}")
    
    if str_start_char:
        raise ValueError(f"Invalid list format: {origin_s}. Unmatched quote: {str_start_char}")
    if buf:
        if str_start_char:
            buf += str_start_char
        lst.append(buf)
        buf = ''
    if not _inner and s.strip():
        raise ValueError(f"Invalid list format: {origin_s}")
    
    for i, item in enumerate(tuple(lst)):
        if isinstance(item, str):
            item_lower = item.lower().strip()
            if item_lower in ('true', 'false'):
                lst[i] = (item_lower == 'true')  
            else:
                maybe_num = get_num_from_text(item)
                if maybe_num is not None:
                    lst[i] = maybe_num
                else:
                    try:
                        item = loads(item)  # not using `deserialize` here, since we dont wanna remove things like `""`
                        lst[i] = item
                    except:
                        ...
    if not _inner and _should_log:
        _logger.debug(f'loaded list: {lst}')
    return lst, s
    
def _parse_base_condition(query: str, calling_by_parse_cond: bool=False) -> tuple["_BaseCond", str]:
    #TODO support parse dict value
    from .condition import BaseCondition as _BaseCond
    
    err_msg = f'Fail to parse base condition from query: `{query}`.'
    invalid_err = lambda msg: ValueError(f"{err_msg} Reason: {msg}")
    
    attr, op, value = '', '', ''
    open_bracket_count = 0
    escape_next, inv_op, is_list = False, False, False
    string_start_char: str|None = None  # '"' or "'"
    
    state_machine = _StateMachine(['attr', 'op', 'value',])
    
    def append(var: Literal['attr', 'op', 'value'], char: str):
        nonlocal attr, op, value
        if _should_log:
            _logger.debug(f'append `{char}` to {var}')
        if var == 'attr':
            attr += char
        elif var == 'op':
            op += char
        elif var == 'value':
            value += char   
        else:
            raise ValueError(f"Invalid variable name: `{var}`")
    
    while query:
        s, query = query[0], query[1:]
        if state_machine.curr == 'attr':
            if not attr:
                if s == '(':
                    open_bracket_count += 1
                elif s == ')':
                    open_bracket_count -= 1
                    if open_bracket_count < 0:
                        raise invalid_err(f"Unmatched closing bracket.")
                elif s == ' ':
                    continue
                elif not (s.isalpha() or s == '_'):   # first char cannot be digit or `.`
                    raise invalid_err(f"Invalid first char in attribute name: `{s}`")
                else:
                    append('attr', s)
            else:
                if s == ' ':
                    state_machine.next()
                elif not (s.isalpha() or s.isdigit() or s in ('_', '.')):
                    if s in ('=', '!', '>', '<'):
                        state_machine.next()
                        query = s + query
                        continue
                    raise invalid_err(f"Invalid char in attribute name: `{attr+s}`")
                else:
                    append('attr', s)
        elif state_machine.curr == 'op':
            # valid op: ['=', '!=', '>', '<', '>=', '<=', 'in', 'not in', 'contains', 'not contains']
            s = s.lower()
            match s:
                case ' ':
                    if op == 'not':
                        inv_op = not inv_op
                        op = '' # reset op
                        continue
                    elif not op:
                        # just ignore
                        continue
                    elif op in _valid_base_ops:
                        if op == '==':
                            op = '='
                        if inv_op:
                            op = _inverse_base_operator[op]
                        state_machine.next()    # go next state
                        continue
                    else:
                        raise invalid_err(f"Invalid space char in operator. Current operator: `{op}`")
                case '=':
                    if op in ('', '=', '!', '>', '<'):
                        # note: '==' also seen as '='
                        append('op', s)
                        continue
                    else:
                        raise invalid_err(f'Invalid operator: `{op+s}`')
                case '!' | '>' | '<':
                    if not op:
                        append('op', s)
                        continue
                    else:
                        raise invalid_err(f"Invalid operator: `{op+s}`")
                case 'n':
                    if op in ('', 'i', 'co', 'contai'):
                        append('op', s)
                        continue
                    else:
                        raise invalid_err(f"Invalid operator: `{op+s}`")
                case 'i':
                    if op in ('', 'conta'):
                        append('op', s)
                        continue
                    else:
                        raise invalid_err(f"Invalid operator: `{op+s}`")
                case 'c':
                    if not op:
                        append('op', s)
                        continue
                    else:
                        raise invalid_err(f"Invalid operator: `{op+s}`")
                case 'o':
                    if op in ('c', 'n'):
                        append('op', s)
                        continue
                    else:
                        raise invalid_err(f"Invalid operator: `{op+s}`")
                case 't':
                    if op in ('no', 'con'):
                        append('op', s)
                        continue
                    else:
                        raise invalid_err(f"Invalid operator: `{op+s}`")
                case 'a':
                    if op == 'cont':
                        append('op', s)
                        continue
                    else:
                        raise invalid_err(f"Invalid operator: `{op+s}`")
                case 's':
                    if op == 'contain':
                        append('op', s)
                        continue
                    else:
                        raise invalid_err(f"Invalid operator: `{op+s}`")
                case _:
                    if op in ('=', '!=', '>', '<', '>=', '<=', '=='):
                        if op == '==':
                            op = '='
                        if inv_op:
                            op = _inverse_base_operator[op]
                        query = s + query   # add back for next state
                        state_machine.next()
                    else:
                        raise invalid_err(f"Invalid operator: `{op+s}`")
        elif state_machine.curr == 'value':
            if escape_next:
                escape_next = False
                match s:
                    case 's':
                        append('value', ' ')
                    case '\\' | '"' | "'":
                        append('value', s)
                    case 'n':
                        append('value', '\n')
                    case 't':
                        append('value', '\t')
                    case 'r':
                        append('value', '\r')
                    case ' ':
                        append('value', '\\')
                        if string_start_char:
                            append('value', ' ')
                        else:   # seems end of value
                            break
                    case _:
                        append('value', '\\' + s)   # not a valid escape char
            else:
                match s:
                    case '\\':
                        escape_next = True
                    case ' ':
                        if string_start_char:
                            append('value', s)
                        else:   
                            if not value:
                                continue
                            else:
                                # seems end of value
                                break
                    case '"' | "'":
                        if s == string_start_char:
                            break # end of value
                        else:
                            if not value:
                                string_start_char = s
                            else:
                                append('value', s)
                    case '(' | '[':
                        if string_start_char:
                            append('value', s)
                        else:
                            try:
                                lst_string, query = _find_first_lst(query, s)
                                lst_string = s + lst_string
                                if _should_log:
                                    _logger.debug(f'list pattern matched: `{lst_string}`. Remaining query: `{query}`')
                                is_list = True
                                value = lst_string.strip()
                                break
                            except:
                                append('value', s)
                    case ')':
                        if string_start_char:
                            append('value', s)
                        elif open_bracket_count > 0:
                            open_bracket_count -= 1
                            break   # end of value
                        else:
                            if calling_by_parse_cond:
                                query = s + query
                                break
                            raise invalid_err(f"Unmatched closing bracket.")
                    case _:
                        append('value', s)
    
    if open_bracket_count > 0:
        while query:
            s, query = query[0], query[1:]
            if s == ' ':
                continue
            elif s == ')':
                open_bracket_count -= 1
                if open_bracket_count < 0:
                    raise invalid_err(f"Unmatched closing bracket.")
                elif open_bracket_count == 0:
                    break
                continue
            else:
                raise invalid_err(f"Unmatched opening bracket. Got: `{s}`")
    if open_bracket_count != 0:
        raise invalid_err(f"Unmatched opening bracket.")
    
    query = query.lstrip()
        
    if not attr:
        raise invalid_err(f'Empty attribute name.')
    if not op:
        raise invalid_err(f'Empty operator.')
    if value:
        if is_list:
            try:
                value, _ = _load_list(value)
                if _should_log:
                    _logger.debug(f'got list value in base condition: {value}')
            except Exception as e:
                raise invalid_err(f"Invalid list format: `{value}`. Reason: {e}")
        else:
            try: 
                value = deserialize(value)  
                if _should_log:
                    _logger.debug(f'Deserialized value: {value}({type(value)})')
            except:
                ...

    return _BaseCond(attr, op, value), query    


type ConditionOperator = Literal['and', 'or']
'''
Operator for joining base conditions,
e.g. `a.b.c = 1 and d.e.f > 2`

NOTE: `&`/`&&` and `|`/`||` are also seen as `and` and `or` respectively.
'''

_operator_map = {
    'and': 'and',
    'or': 'or',
    '&&': 'and',
    '||': 'or',
    '&': 'and',
    '|': 'or',
}

def _parse_condition(query: str, inner: bool=False)->tuple["_Cond", str]:
    from .condition import BaseCondition as _BaseCond, Condition as _Cond
    from .condition import TrueCondition, FalseCondition
    
    err_msg = f'Fail to parse condition from query: `{query}`.'
    invalid_err = lambda msg: ValueError(f"{err_msg} Reason: {msg}")
    buf, op, inv_next_cond = '', None, False
    cond: '_BaseCond|_Cond|bool|None' = None
    state_machine = _StateMachine(['cond', 'op',], loop=True)
    
    def join_cond(op: ConditionOperator|None, right: '_BaseCond|_Cond|bool'):
        nonlocal cond
        if _should_log:
            _logger.debug(f'joining new condition: `{right}` to `{cond}` with operator `{op}`')
        if op:
            op = _operator_map.get(op, None)        # type: ignore
            if not op:
                raise invalid_err(f"Invalid operator: `{op}`.")
            
            if cond is None:
                raise invalid_err(f"Invalid operator: `{op}` before condition.")
            if isinstance(cond, bool):
                if isinstance(right, bool):
                    cond = (cond and right) if op == 'and' else (cond or right)
                else:   # _BaseCond/_Cond
                    if cond:
                        if op == 'and':
                            cond = right    # 1 & x = x
                        else:
                            ... # 1 | x = 1
                    else:
                        if op == 'and':
                            ... # 0 & x = 0
                        else:
                            cond = right    # 0 | x = x
            elif isinstance(cond, _BaseCond):
                if isinstance(right, bool):
                    if right:
                        if op == 'and':
                            ... # x & 1 = x
                        else:
                            cond = right    # x | 1 = 1
                    else:
                        if op == 'and':
                            cond = right    # x & 0 = 0
                        else:
                            ... # x | 0 = x
                else:
                    if op == 'and':
                        cond = cond & right
                    else:
                        cond = cond | right
            elif isinstance(cond, _Cond):
                if isinstance(right, bool):
                    if right:
                        if op == 'and': # x & 1 = x
                            ...
                        else:
                            cond = right
                    else:
                        if op == 'and':
                            cond = right
                        else:
                            ...
                else:
                    if op == 'and':
                        cond = cond & right
                    else:
                        cond = cond | right
            else:
                raise invalid_err(f"Invalid condition: `{cond}`.")
        else:
            if not cond is None:
                raise invalid_err(f"No operator between conditions.")
            cond = right

    while query:
        s, query = query[0], query[1:]
        if state_machine.curr == 'cond':
            match s:
                case " ":
                    buf_lower = buf.lower()
                    if not buf_lower:
                        continue
                    elif buf_lower == 'not':
                        inv_next_cond = not inv_next_cond
                        buf = '' 
                    elif buf_lower in ('true', 'false', '1', '0'):
                        new_cond = deserialize(buf, target_type=bool)  
                        buf = ''
                        if inv_next_cond:
                            new_cond = not new_cond
                            inv_next_cond = False
                        join_cond(op, new_cond)     # type: ignore
                        state_machine.next()
                    else:
                        # must be base condition
                        base_cond, query = _parse_base_condition(buf + ' ' + query, calling_by_parse_cond=True)
                        if _should_log:
                            _logger.debug(f'got base condition: `{base_cond}`. Remaining query: `{query}`')
                        buf = ''
                        if inv_next_cond:
                            base_cond = base_cond.not_()
                            inv_next_cond = False
                        join_cond(op, base_cond)   # type: ignore 
                        state_machine.next()
                case "(":
                    inv = inv_next_cond
                    inv_next_cond = False
                    if buf:
                        if buf.lower() == 'not':
                            inv = not inv
                            buf = ''
                        else:
                            raise invalid_err(f"Invalid pattern: `{buf}`.")
                    new_cond, query = _parse_condition(query, inner=True)
                    if _should_log:
                        _logger.debug(f'got inner condition: `{new_cond}`. Remaining query: `{query}`')
                    if not query or query[0] != ")":
                        raise invalid_err(f"Missing closing bracket.")
                    else:
                        query = query[1:]
                    if inv:
                        new_cond = new_cond.not_()
                    join_cond(op, new_cond)     # type: ignore
                    state_machine.next()
                case ")":
                    if inner:
                        query = s + query # add back for outer parser
                        break
                    else:
                        raise invalid_err(f"Unmatched closing bracket.")
                case _:
                    buf += s
                    
        else:   # op
            # NOTE: op does not support `not` or `!`
            match s:
                case " ":
                    buf_lower = buf.lower()
                    if not buf_lower:
                        continue
                    elif buf_lower == 'not':
                        raise invalid_err(f"`not` is not allowed before an operator.")
                    elif buf_lower in _operator_map:
                        op = _operator_map[buf_lower]   
                        buf = ''
                        state_machine.next()
                    else:
                        raise invalid_err(f"Invalid operator: `{buf}`.")
                case "&" | "|":
                    if (buf and buf[-1] != s) or len(buf) >=2:
                        raise invalid_err(f"Invalid operator: `{buf+s}`.")
                    buf += s
                case "a" | "o":
                    if buf:
                        raise invalid_err(f"Invalid operator: `{buf+s}`.")
                    buf += s
                case "n":
                    if buf.lower() != 'a':
                        raise invalid_err(f"Invalid operator: `{buf+s}`.")
                    buf += s
                case "d":
                    if buf.lower() != 'an':
                        raise invalid_err(f"Invalid operator: `{buf+s}`.")
                    buf += s
                case "r":
                    if buf.lower() != 'o':
                        raise invalid_err(f"Invalid operator: `{buf+s}`.")
                    buf += s
                case _:
                    if not buf and s == ')' and inner:
                        query = s + query
                        break
                    raise invalid_err(f"Invalid operator: `{buf+s}`.")
    if buf:
        base_cond, rest = _parse_base_condition(buf, calling_by_parse_cond=True)
        if rest.strip():
            raise invalid_err(f"Invalid condition: `{buf}`.")
        if _should_log:
            _logger.debug(f'got base condition: `{base_cond}`. Remaining query: `{rest}`')
        buf = ''
        if inv_next_cond:
            base_cond = base_cond.not_()
            inv_next_cond = False
        join_cond(op, base_cond)        # type: ignore
    
    if cond is None:
        raise invalid_err(f"Empty condition.")
    if inv_next_cond:
        if isinstance(cond, bool):
            cond = not cond
        else:
            cond = cond.not_()
    
    if not inner and query.strip():
        raise invalid_err(f"Invalid condition: `{query}`.")
    if isinstance(cond, bool):
        return TrueCondition if cond else FalseCondition, query
    elif isinstance(cond, _BaseCond):
        return _Cond(left=cond), query
    return cond, query


__all__ = [
    'BaseConditionOperator',
    'ConditionOperator',
]

if __name__ == '__main__':
    # test
    query = 'a.b.c = 1 and (d.e.f > 2 or g.h.i != 3)'
    cond, rest = _parse_base_condition(query)
    print('base cond:', cond, '| rest:', rest)
    
    print('----' * 5)
    
    query = 'a.b.c = 1 and (d.e.f > 2 or g.h.i != 3) and (j.k.l in [1, 2, True] or m.n.o not in [4, 5, 6])'
    cond, rest = _parse_condition(query)
    print('cond:', cond)