__all__ = ["MatrixError", 
           "InvalidDimensionsError",
           "NotSquareError",
           "SingularMatrixError",
           "IndexOutOfBoundsError",
           "InvalidDataError",
           "InvalidShapeError",
           "MatrixValueError",
           ]

class MatrixError(Exception):
    """Base class for all matrix-related exceptions."""
    pass

class InvalidDimensionsError(ValueError, MatrixError):
    """Raised when two matrices do not have compatible dimensions for an operation."""
    def __init__(self, first=None, second=None, operation='<unspecified>', reason='Matrices do not have compatible dimensions'):
        self.first, self.second  = first, second 
        self.operation = operation
        self.reason = reason
        self.first_shape = first.shape if first is not None else None
        self.second_shape = second.shape if second is not None else None

        parts = [f'Cannot perform: "{operation}":']
        parts.append(f'  Reason: {reason}')
        if first is not None or second is not None:
            parts.append(f'  First shape: {self.first_shape}')
            parts.append(f'  Second shape: {self.second_shape}')
        message = '\n'.join(parts)
        super().__init__(message)

class NotSquareError(ValueError, MatrixError):
    """Raised when an operation requires a square matrix but a non-square one is provided."""
    def __init__(self, matrix=None, operation='<unspecified>', reason='Matrix is not square'):
        self.matrix = matrix
        self.operation = operation
        self.reason = reason
        self.shape = matrix.shape if matrix is not None else None
        
        parts = [f'Cannot perform: "{operation}":']
        parts.append(f'  Reason: {reason}')
        if matrix is not None:
            parts.append(f'  Matrix shape: {matrix.shape}')
        message = '\n'.join(parts)
        super().__init__(message)

class SingularMatrixError(ArithmeticError, MatrixError):
    """Raised when attempting to invert or solve a system with a singular (non-invertible) matrix."""
    def __init__(self, matrix=None, operation='<unspecified>', reason='The matrix is singular (determinant = 0) and therefore non-invertible'):
        self.matrix = matrix
        self.operation = operation
        self.reason = reason

        parts = [f'Cannot perform: "{operation}":']
        parts.append(f'  Reason: {reason}')
        message = '\n'.join(parts)
        super().__init__(message)

class IndexOutOfBoundsError(IndexError, MatrixError):
    """Raised when indexing outside the valid row/column range."""
    def __init__(self, matrix=None, index=None, axis='row', operation='<unspecified>', reason='Index is out of bounds'):
        self.matrix = matrix
        self.index = index
        self.axis = axis
        self.max_valid = (matrix.rows if axis == 'row' else matrix.cols) if matrix is not None else None
        self.operation = operation
        self.reason = reason

        parts = [f'Cannot perform: "{operation}":']
        parts.append(f'  Reason: {reason}')
        parts.append(f'  Axis: {axis}')
        if index is not None:
            if isinstance(index, (list, tuple)):
                parts.append(f'  Valid indices:  [1 .. {self.max_valid}]')
                parts.append('  Got Indices:  ' + '[' + ', '.join(map(str, index)) + ']')
            else:
                parts.append(f'  Valid index: [1 .. {self.max_valid}]')
                parts.append(f'  Got index: {index}')
        if matrix is not None:
            parts.append(f'  Matrix shape: {matrix.shape}')
        message = '\n'.join(parts)
        super().__init__(message)

class InvalidDataError(TypeError, MatrixError):
    """Raised when input has the wrong type."""
    def __init__(self, obj=None, expected_type=None, operation='<unspecified>', reason='Input has an invalid type'):
        # infer type of nested sequences
        def infer_type(o):
            # Base case: not a container we handle
            if not isinstance(o, (list, tuple)):
                return type(o).__name__

            # List
            if isinstance(o, list):
                if not o:
                    return 'list'
                inner = {infer_type(el) for el in o}
                inner_str = inner.pop() if len(inner) == 1 else 'Any'
                return f'list[{inner_str}]'

            # Tuple
            if isinstance(o, tuple):
                if not o:
                    return 'tuple'
                inner = {infer_type(el) for el in o}
                inner_str = inner.pop() if len(inner) == 1 else 'Any'
                return f'tuple[{inner_str}]'
        
        self.obj = obj
        self.expected_type = expected_type
        self.actual_type = infer_type(obj)
        self.operation = operation
        self.reason = reason

        parts = [f'Cannot perform: "{operation}":']
        parts.append(f'  Reason: {reason}')
        if expected_type is not None:
            parts.append(f'  Expected type: {expected_type}')
        if obj is not None:
            parts.append(f'  Got type:      {self.actual_type}')
        message = '\n'.join(parts)
        super().__init__(message)
    
class InvalidShapeError(ValueError, MatrixError):
    """Raised when input has the wrong shape."""
    def __init__(self, obj=None, expected_shape=None, operation='<unspecified>', reason='Input has an invalid shape'):
        # infer shape of nested sequences
        def infer_shape(o):
            if isinstance(o, (list, tuple)):
                if not o:
                    return (0,)
                child = infer_shape(o[0])
                if all(infer_shape(el) == child for el in o):
                    return (len(o),) + child
                return (len(o),)
            return ()

        self.obj = obj
        self.expected_shape = expected_shape
        self.actual_shape = infer_shape(obj) if obj is not None else None
        self.operation = operation
        self.reason = reason

        parts = [f'Cannot perform: "{operation}":']
        parts.append(f'  Reason: {reason}')
        if expected_shape is not None:
            parts.append(f'  Expected shape:   {expected_shape}')
        if obj is not None:
            parts.append(f'  Got shape:        {self.actual_shape}')
        message = '\n'.join(parts)
        super().__init__(message)


class MatrixValueError(ValueError, MatrixError):
    """Raised when a value is semantically invalid for a matrix operation."""
    def __init__(self, matrix=None, value=None, operation='<unspecified>', reason='Invalid value',):
        self.operation = operation
        self.reason = reason
        self.value = value
        self.matrix = matrix
        self.shape = matrix.shape if matrix is not None else None

        parts = [f'Cannot perform: "{operation}":']
        parts.append(f'  Reason: {reason}')
        if value is not None:
            if hasattr(value, '__repr__'):
                parts.append(f'  Got value: {repr(value)}')
            else:
                parts.append(f'  Got value: {type(value).__name__}')
        if matrix is not None:
            parts.append(f'  Matrix shape: {matrix.shape}')
        message = '\n'.join(parts)
        super().__init__(message)
