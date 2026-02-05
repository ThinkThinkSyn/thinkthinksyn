if __name__ == "__main__": # for debugging
    import os, sys
    _proj_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
    sys.path.append(_proj_path)
    __package__ = 'thinkthinksyn.db_utils.milvus'

import numpy as np

from typing import Self, Union, TypeAlias, TYPE_CHECKING, Sequence, ClassVar

from ...common_utils.type_utils import get_cls_name, check_value_is

if TYPE_CHECKING:
    from .fields import MilvusField
    from .base_model import MilvusModel

ExpressionT: TypeAlias = Union['QueryExpression', 'MilvusField', int, float, str, Sequence, bool, None, np.number]
'''types supported for query expression'''

def _format_exp(exp: ExpressionT)->str:
    from .fields import MilvusField
    if isinstance(exp, str) and not isinstance(exp, QueryExpression):
        if not exp.startswith('"') and not exp.endswith('"'):
            return f'"{exp}"'
        else:
            return exp
    elif isinstance(exp, (int, float)):
        return str(exp)
    elif isinstance(exp, QueryExpression):
        return exp
    elif isinstance(exp, MilvusField):
        return exp.name
    else:
        raise TypeError(f'Unsupported type {type(exp)} for expression: {exp}')

class QueryExpression(str):
    '''
    Expression proxy for milvus searching.
    This object will be returned when operate a model field (equal, compare, ...) directly,
    e.g. `A.x == 1`
    '''
    
    ModelClass: ClassVar['type[MilvusModel]|None'] = None
    '''The model class binding to this expression.'''
    __QueryExpressionClses__: ClassVar[dict[str, 'type[QueryExpression]']] = {}
    
    def __class_getitem__(cls, item: 'type[MilvusModel]|None')->'type[QueryExpression]':
        '''Get a QueryExpression subclass binding to a model class.'''
        if not item:
            return QueryExpression
        cls_name = get_cls_name(cls, with_module_name=True)
        if cls_name not in QueryExpression.__QueryExpressionClses__:
            QueryExpression.__QueryExpressionClses__[cls_name] = type('QueryExpression', (QueryExpression,), {'ModelClass': item})
        return QueryExpression.__QueryExpressionClses__[cls_name]
    
    @classmethod
    def Build(cls, a: ExpressionT, operator: str, b: ExpressionT)->Self:
        '''build an expression string.'''
        left, right = _format_exp(a), _format_exp(b)
        if cls.ModelClass:
            from .fields import MilvusField
            if isinstance(a, MilvusField):
                real_field_name = cls.ModelClass.ContainsField(a.name)
                if real_field_name:
                    if real_field_name in cls.ModelClass.__non_standard_type_adapters__:
                        # non-standard type field
                        if b is None:
                            right = _format_exp('null')
                        elif check_value_is(b, a.annotation):
                            adapter = cls.ModelClass.__non_standard_type_adapters__[real_field_name]
                            right = _format_exp(adapter.dump_json(b).decode())
                    elif real_field_name in cls.ModelClass.__optional_str_fields__:
                        # optional string field
                        if b is None:
                            right = _format_exp('')
        return cls(f'({left} {operator} {right})')
    
    def __eq__(self, other: ExpressionT)->Self:
        return self.Build(self, '==', other) # type: ignore
    
    def __ne__(self, other: ExpressionT)->Self:
        return self.Build(self, '!=', other) # type: ignore
    
    def __lt__(self, other: ExpressionT)->Self:
        return self.Build(self, '<', other)  # type: ignore
    
    def __le__(self, other: ExpressionT)->Self:
        return self.Build(self, '<=', other) # type: ignore
    
    def __gt__(self, other: ExpressionT)->Self:
        return self.Build(self, '>', other)  # type: ignore
    
    def __ge__(self, other: ExpressionT)->Self:
        return self.Build(self, '>=', other) # type: ignore
    
    def __and__(self, other: ExpressionT)->Self:
        return self.Build(self, 'and', other)    # type: ignore
    
    def __or__(self, other: ExpressionT)->Self:
        return self.Build(self, 'or', other) # type: ignore
    
    def __xor__(self, other: ExpressionT)->Self:
        return self.Build(self, '^', other)  # type: ignore
    
    def __invert__(self):
        return self.__class__(f'not ({self})')


__all__ = ['QueryExpression', 'ExpressionT']


if __name__ == '__main__':
    print(~QueryExpression('A.x == 1'))
    print(QueryExpression('A.x == 1') & QueryExpression('A.x == 2') & QueryExpression('A.x == 3'))