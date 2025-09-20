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
        EpsMixin,
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
        EpsMixin,
    ):

    # === NoName ===
    def map(self, func: function):
        return self.__class__([
             [func(self.data[row][col])
              for col in range(self.cols)] 
              for row in range(self.rows)
        ])

