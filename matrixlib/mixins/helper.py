from typing import Self

from ..exceptions import (
    MatrixError, 
    InvalidDimensionsError,
    NotSquareError,
    SingularMatrixError,
    IndexOutOfBoundsError,
    InvalidDataError,
    InvalidShapeError,
    MatrixValueError,
)

class HelperMixin:
    # === NoName ===
    def _have_same_size(self, other: Self) -> bool:
        return self.rows == other.rows and self.cols == other.cols
    
    def _round_off(self) -> None:
        rows, cols = self.rows, self.cols
        if self._is_floats_matrix():
            eps = type(self).eps()
            for i in range(1, rows+1):
                for j in range(1, cols+1):
                    if abs(round(self[i,j]) - self[i,j]) <= eps:
                        self[i,j] = int(round(self[i,j]))

    def _triple_equal(self, other: Self) -> bool:
        if not self._have_same_size(other):
            return False
        
        eps = type(self).eps()
        rows, cols = self.rows, self.cols

        if self._is_floats_matrix() and other._is_floats_matrix():
            return all(abs(self[i,j]-other[i,j]) <= eps for i in range(1, rows+1) for j in range( 1, cols+1))
            
        return all(self[i,j] == other[i,j] for i in range(1, rows+1) for j in range( 1, cols+1))
    
    # === NoName ===
    def _is_square(self) -> bool:
        return self.rows == self.cols
    
    def _is_boolean_matrix(self) -> bool:
        return all([isinstance(entry, bool) for entry in self])
    
    def _is_integer_matrix(self) -> bool:
        if all(isinstance(entry, int) for entry in self):
            return True
        if all(isinstance(entry, float | int) for entry in self):
            return all(isinstance(entry, int) or (entry-round(entry) == 0) for entry in self)
        else:
            return False
    
    def _is_floats_matrix(self) -> bool:
        return all(isinstance(entry, float | int) for entry in self)
    
    # === Helpers ===
    def _validate_other_type(self, other, *, operation: str = "<unspecified>", reason: str = 'Operand must be an "Matrix"') -> None:
        if not isinstance(other, type(self)):
            raise InvalidDataError(
                obj=other,
                expected_type=type(self).__name__,
                operation=operation,
                reason=reason,
            )
        
    def _validate_same_size(self, other: Self, *, operation: str = "<unspecified>", reason: str = "Matrices have different dimensions") -> None:
        if not self._have_same_size(other):
            raise InvalidDimensionsError(
                first=self,
                second=other,
                operation=operation,
                reason=reason,
            )

    def _validate_boolean_matrix(self, *, operation: str = "<unspecified>", reason: str = "Operand is not a boolean matrix") -> None:
        if not self._is_boolean_matrix():
            raise MatrixValueError(
                value=self,
                operation=operation,
                reason=reason,
            )