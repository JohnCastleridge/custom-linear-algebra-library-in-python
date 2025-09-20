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
    def _have_same_size(self, other):
        return self.rows == other.rows and self.cols == other.cols
    
    def _is_square(self):
        return self.rows == self.cols
    
    def _is_boolean_matrix(self):
        return all([isinstance(value, bool) for row in self.data for value in row])
    
    def _is_integer_matrix(self):
        return all([value-round(value) == 0 for row in self.data for value in row])
    
    # === Helpers ===
    def _validate_other_type(self, other, *, operation: str) -> None:
        """Validate that ``other`` is the same (or subclass-compatible) type.

        Args:
            other: Right-hand-side operand to validate.
            operation: Operation name used in error messages.

        Raises:
            InvalidDataError: If ``other`` is not an instance of ``type(self)``.
        """

        if not isinstance(other, type(self)):
            raise InvalidDataError(
                obj=other,
                expected_type=type(self).__name__,
                operation=operation,
                reason="Operand must be the same matrix type as self",
            )
        
    def _validate_same_size(self, other, *, operation: str) -> None:
        """Validate that ``self`` and ``other`` have identical shapes.

        Args:
            other: Right-hand-side operand whose size is compared to ``self``.
            operation: Operation name used in error messages.

        Raises:
            InvalidDimensionsError: If the operands have different dimensions.
        """
        if not self._have_same_size(other):
            raise InvalidDimensionsError(
                first=self,
                second=other,
                operation=operation,
                reason="Matrices have different dimensions",
            )

    def _validate_boolean_matrix(self, *, operation: str) -> None:
        """Validate that ``self`` is a boolean matrix.

        Relies on the implementation-provided ``_is_boolean_matrix()``.

        Args:
            operation: Operation name used in error messages.

        Raises:
            MatrixValueError: If ``self`` is not a boolean matrix.
        """
        if not self._is_boolean_matrix():
            raise MatrixValueError(
                value=self,
                operation=operation,
                reason="Operand is not a boolean matrix",
            )