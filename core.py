from .exceptions import (
    InvalidDataError,
    InvalidShapeError,
)

from .mixins import (
        BinaryMatrixOperationsMixin, 
        UnaryMatrixOperationsMixin,
        ElementaryOperationsMixin,
        BooleanLogicMixin,
        ElementwiseComparisonMixin,
        MatrixFactoryMixin,
        DunderMixin,
        MatrixMathMixin,
        EpsMixin,
        HelperMixin,
)

__all__ = ["Matrix"]

class Matrix(
        BinaryMatrixOperationsMixin, 
        UnaryMatrixOperationsMixin,
        ElementaryOperationsMixin,
        BooleanLogicMixin,
        ElementwiseComparisonMixin,
        MatrixFactoryMixin,
        DunderMixin,
        MatrixMathMixin,
        EpsMixin,
        HelperMixin,
    ):
    # === Initialization ===
    def __init__(self, data: list[list[any]]):
        # Validate input matrix structure
        if not isinstance(data, list) or not data or not all(isinstance(row, list) for row in data):
            raise InvalidDataError(obj=data, expected_type='list[list]', operation='Matrix.__init__', reason='Data must be a non‐empty list of lists')
        if any(len(row) != len(data[0]) for row in data):
            raise InvalidShapeError(obj=data, expected_shape=(len(data), len(data[0])), operation='Matrix.__init__', reason='All rows must have the same number of columns')
        
        self._rows = len(data)
        self._cols = len(data[0])
        self._shape = (self.rows, self.cols)
        self._data = [row[:] for row in data]

    # === NoName ===
    @property
    def rows(self):
        return self._rows
    @property
    def cols(self):
        return self._cols
    @property
    def shape(self):
        return (self.rows, self.cols)