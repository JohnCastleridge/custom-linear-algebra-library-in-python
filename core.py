from typing import Callable, Self

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
        HelperMixin,
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
        HelperMixin,
    ):

    # === NoName ===
    def map(self, func: Callable) -> Self:
        return self.__class__([
             [func(self.data[row][col])
              for col in range(self.cols)] 
              for row in range(self.rows)
        ])

