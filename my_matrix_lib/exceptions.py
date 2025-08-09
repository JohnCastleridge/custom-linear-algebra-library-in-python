

class MatrixError(Exception):
    """Base class for all matrix-related exceptions."""
    pass

class InvalidDimensionsError(ValueError, MatrixError):
    """Raised when two matrices do not have compatible dimensions for an operation."""
    def __init__(self, first, second, *, operation='<unspecified>', reason='Matrices do not have compatible dimensions'):
        self.first, self.second  = first, second 
        self.operation = operation
        self.reason = reason
        self.first_shape, self.second_shape = first.shape, second.shape

        message = (
            f'Cannot perform: "{operation}":' + '\n'
            f'  Reason: {reason}' + ': ' + '\n'
            f'  First is {first.rows}×{first.cols}' + '\n'
            f'  Second is {second .rows}×{second .cols}'
        )
        super().__init__(message)

class NotSquareError(ValueError, MatrixError):
    """Raised when an operation requires a square matrix but a non-square one is provided."""
    def __init__(self, matrix, *, operation='<unspecified>', reason='Matrix is not square'
                 ):
        self.matrix = matrix
        self.operation = operation
        self.reason = reason
        self.shape = matrix.shape
        
        message = (
            f'Cannot perform: "{operation}":' + '\n'
            f'  Reason: {reason}' + ': ' + '\n'
            f'  The matrix is {matrix.rows}×{matrix.cols}'
        )
        super().__init__(message)

class SingularMatrixError(ArithmeticError, MatrixError):
    """Raised when attempting to invert or solve a system with a singular (non-invertible) matrix."""
    def __init__(self, matrix, *, operation='<unspecified>', reason='The matrix is singular (determinant = 0) and therefore non‑invertible'):
        self.matrix = matrix
        self.operation = operation
        self.reason = reason

        message = (
            f'Cannot perform: "{operation}"\n'
            f'  Reason: {reason}\n'
        )
        super().__init__(message)

class IndexOutOfBoundsError(IndexError, MatrixError):
    """Raised when indexing outside the valid row/column range."""
    def __init__(self, matrix, index, *, axis = 'row', operation='<unspecified>', reason='Index is out of bounds'):
        self.matrix = matrix
        self.index = index
        self.axis = axis
        self.max_valid = matrix.rows if axis == 'row' else matrix.cols
        self.operation = operation
        self.reason = reason


        parts = [f'Cannot perform: "{operation}":']
        parts.append(f'  Reason: {reason}')
        parts.append(f'  Axis: {axis}')
        if isinstance(index, (list, tuple)):
            parts.append(f'  Valid indices:  [1 .. {self.max_valid}]')
            parts.append('  Got Indices:  ' + '[' + ', '.join(map(str, index)) + ']')
        else:
            parts.append(f'  Valid index: [1 .. {self.max_valid}]')
            parts.append(f'  Got index: {index}')
    
        parts.append(f'  Matrix shape: {matrix.shape}')
        message = '\n'.join(parts)
        super().__init__(message)

class InvalidDataError(TypeError, MatrixError):
    """Raised when input has the wrong type."""
    def __init__(self, obj, expected_type: str, *, operation='<unspecified>', reason='Input has an invalid type'):
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

        message = (
            f'Cannot perform "{operation}":\n'
            f'  Reason: {reason}\n'
            f'  Expected type: {expected_type}\n'
            f'  Got type:      {self.actual_type}\n'
        )
        super().__init__(message)
    
class InvalidShapeError(ValueError, MatrixError):
    """Raised when input has the wrong shape."""
    def __init__(self, obj, expected_shape, *, operation='<unspecified>', reason='Input has an invalid shape'):
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
        self.actual_shape = infer_shape(obj)
        self.operation = operation
        self.reason = reason

        message = (
            f'Cannot perform "{operation}":\n'
            f'  Reason: {reason}\n'
            f'  Expected shape:   {expected_shape}\n'
            f'  Got shape:        {self.actual_shape}\n'
        )
        super().__init__(message)

class MatrixValueError(ValueError, MatrixError):
    """Raised when a value is semantically invalid for a matrix operation."""
    def __init__(self, matrix, value, *, operation = '<unspecified>', reason = 'Invalid value',):
        self.operation = operation
        self.reason = reason
        self.value = value
        self.matrix = matrix

        message = (
            f'Cannot perform "{operation}":\n'
            f'  Reason: {reason}\n'
            f'  Got value: {value}\n'
            f'  Matrix shape: {matrix.shape}\n'
        )
        super().__init__(message)
