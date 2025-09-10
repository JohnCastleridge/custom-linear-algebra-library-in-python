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

    @classmethod
    def eps(cls):
        return 1e-8

    def new_eps(cls, tol):
        pass

    @property
    def eps(self):
        return self.__name__.eps()


    def map(self, func: function):
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

