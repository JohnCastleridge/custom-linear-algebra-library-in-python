from .exceptions import (
    InvalidDataError,
    InvalidShapeError,
)

from .mixins import (
        BinaryMatrixOperationsMixin, 
        UnaryMatrixOperationsMixin,
        MatrixRowOperationsMixin,
        BooleanLogicMixin,
        ElementwiseComparisonMixin,
        MatrixFactoryMixin,
        DunderMixin,
        MatrixMathMixin,
)

__all__ = ["Matrix"]

class Matrix(
        BinaryMatrixOperationsMixin, 
        UnaryMatrixOperationsMixin,
        MatrixRowOperationsMixin,
        BooleanLogicMixin,
        ElementwiseComparisonMixin,
        MatrixFactoryMixin,
        DunderMixin,
        MatrixMathMixin,
    ):
    
    # === Initialization ===
    def __init__(self, data: list[list[any]]):
        # Validate input matrix structure
        if not isinstance(data, list) or not data or not all(isinstance(row, list) for row in data):
            raise InvalidDataError(obj=data, expected_type='list[list]', operation='Matrix.__init__', reason='Data must be a non‐empty list of lists')
        if any(len(row) != len(data[0]) for row in data):
            raise InvalidShapeError(obj=data, expected_shape=(len(data), len(data[0])), operation='Matrix.__init__', reason='All rows must have the same number of columns')

        self.rows = len(data)
        self.cols = len(data[0])
        self.shape = (self.rows, self.cols)
        self.data = [row[:] for row in data]
    

    # === Representation ===
    def __str__(self):
        decimal_places = 2
        column_padding = 2
        
        if self._is_integer_matrix():
            data = [[str(entry) for entry in row] for row in self.data]
        else:
            data = [[f'{entry:.{decimal_places}f}' for entry in row] for row in self.data]
        
        
        num_len = max(len(string) if '-' not in string else len(string)-1 
                 for row in data for string in row)
        
        data = [[' '*(num_len-len(string)+column_padding) + string 
                 for string in row] for row in data]
        
        return '\n'.join(
            ["┌" + " "*(num_len+2)*self.cols + "  " + "┐"] +
            ["|" + "".join(string for string in row) + "  |" for row in data] +
            ["└" + " "*(num_len+2)*self.cols + "  " + "┘"])

    # find where to place
    def map(self, func):
        return self.__class__([
             [func(self.data[row][col])
              for col in range(self.cols)] 
              for row in range(self.rows)
        ])

    # === Validation functions ===
    def _have_same_size(self, other):
        return self.rows != other.rows or self.cols != other.cols
    
    def _is_square(self):
        return self.rows == self.cols
    
    def _is_boolean_matrix(self):
        return all([isinstance(value, bool) for row in self.data for value in row])
    
    def _is_integer_matrix(self):
        return all([value-round(value) == 0 for row in self.data for value in row])

