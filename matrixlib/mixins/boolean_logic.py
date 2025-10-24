from ..exceptions import (
    InvalidDimensionsError,
    InvalidDataError,
)

class BooleanLogicMixin:
    # === NoName ===
    def elementwise_OR(self, other):
        """
        Returns a new boolean Matrix where each entry
        is the logical disjunction (also known as Logical OR or logical addition) 
        of the corresponding entry in self and other.
        """
        if self._have_same_size(other):
            raise InvalidDimensionsError(self, other, 
                operation="elementwise OR",
                reason="Matrices have different dimensions"
            )
        if not self._is_boolean_matrix() or not other._is_boolean_matrix():
            raise InvalidDataError(
                "Cannot perform logical disjunction (Logical OR) on non-boolean matrices"
            )
        return self.__class__([
            [not self.data[row][col] or other.data[row][col]
             for col in range(self.cols)]
            for row in range(self.rows)
        ])
    
    def elementwise_AND(self, other):
        """
        Returns a new boolean Matrix where each entry
        is the logical conjunction (also known as Logical AND or logical multiplication) 
        of the corresponding entry in self and other.
        """
        if self._have_same_size(other):
            raise InvalidDimensionsError(self, other, 
                operation="elementwise AND",
                reason="Matrices have different dimensions"
            )
        if not self._is_boolean_matrix() or not other._is_boolean_matrix():
            raise InvalidDataError(
                "Cannot perform logical conjunction (Logical AND) on non-boolean matrices"
            )
        return self.__class__([
            [not self.data[row][col] and other.data[row][col]
             for col in range(self.cols)]
            for row in range(self.rows)
        ])
    
    def elementwise_NOT(self):
        """
        Returns a new boolean Matrix where each entry
        is the negation of the corresponding entry in self.
        """
        if not all([isinstance(value, bool) for row in self.data for value in row]):
            raise InvalidDataError(
                "Cannot perform logical NOT on non-boolean matrix"
            )
        return self.__class__([
            [not self(i, j)
             for j in range(self.cols)]
            for i in range(self.rows)
        ])

    # === Elementwise comparisons ===
    def elementwise_equal(self, other, *, tol=None):
        """Test elementwise equality with tolerance.

        Elements ``a`` and ``b`` are considered equal if ``|a - b| <= tol``.

        Args:
            other: Matrix-like object with the same type and shape as ``self``.
            tol: Absolute tolerance for equality. If ``None``, uses
                ``type(self).eps()``.

        Returns:
            Matrix: Boolean matrix where each entry is ``True`` if elements are
            equal within tolerance.

        Raises:
            InvalidDataError: If ``other`` is not the same matrix type as ``self``.
            InvalidDimensionsError: If the matrices have different dimensions.
        """
        op = "elementwise equality"
        self._validate_other_type(other, operation=op)
        self._validate_same_size(other, operation=op)

        tol = type(self).eps() if tol is None else tol
        rows, cols = self.rows, self.cols
        return self.__class__([
            [abs(self[i, j] - other[i, j]) <= tol 
             for j in range(1, cols+1)]
             for i in range(1, rows+1)
        ])

    def elementwise_not_equal(self, other, *, tol=None):
        """Test elementwise inequality with tolerance.

        Elements ``a`` and ``b`` are considered different if ``|a - b| > tol``.

        Args:
            other: Matrix-like object with the same type and shape as ``self``.
            tol: Absolute tolerance used to negate equality. If ``None``, uses
                ``type(self).eps()``.

        Returns:
            Matrix: Boolean matrix where each entry is ``True`` if elements
            differ beyond tolerance.

        Raises:
            InvalidDataError: If ``other`` is not the same matrix type as ``self``.
            InvalidDimensionsError: If the matrices have different dimensions.
        """
        op = "elementwise inequality"
        self._validate_other_type(other, operation=op)
        self._validate_same_size(other, operation=op)

        tol = type(self).eps() if tol is None else tol
        rows, cols = self.rows, self.cols
        return self.__class__([
            [abs(self[i, j] - other[i, j]) > tol 
             for j in range(1, cols+1)]
             for i in range(1, rows+1)
        ])

    def elementwise_less_than(self, other, *, tol=None):
        """Test elementwise strict less-than with tolerance.

        Compares using ``self - other < -tol``.

        Args:
            other: Matrix-like object with the same type and shape as ``self``.
            tol: Non-negative slack for comparison. If ``None``, uses
                ``type(self).eps()``.

        Returns:
            Matrix: Boolean matrix where each entry is ``True`` if
            ``self(i, j) < other(i, j)`` within tolerance.

        Raises:
            InvalidDataError: If ``other`` is not the same matrix type as ``self``.
            InvalidDimensionsError: If the matrices have different dimensions.
        """
        op = "elementwise less than"
        self._validate_other_type(other, operation=op)
        self._validate_same_size(other, operation=op)

        tol = type(self).eps() if tol is None else tol
        rows, cols = self.rows, self.cols
        return self.__class__([
            [self[i, j] - other[i, j] < -tol 
             for j in range(1, cols+1)]
             for i in range(1, rows+1)
        ])

    def elementwise_greater_than(self, other, *, tol=None):
        """Test elementwise strict greater-than with tolerance.

        Compares using ``self - other > tol``.

        Args:
            other: Matrix-like object with the same type and shape as ``self``.
            tol: Non-negative slack for comparison. If ``None``, uses
                ``type(self).eps()``.

        Returns:
            Matrix: Boolean matrix where each entry is ``True`` if
            ``self(i, j) > other(i, j)`` within tolerance.

        Raises:
            InvalidDataError: If ``other`` is not the same matrix type as ``self``.
            InvalidDimensionsError: If the matrices have different dimensions.
        """
        op = "elementwise greater than"
        self._validate_other_type(other, operation=op)
        self._validate_same_size(other, operation=op)

        tol = type(self).eps() if tol is None else tol
        rows, cols = self.rows, self.cols
        return self.__class__([
            [self[i, j] - other[i, j] > tol 
             for j in range(1, cols+1)]
             for i in range(1, rows+1)
        ])

    def elementwise_less_than_or_equal(self, other, *, tol=None):
        """Test elementwise less-than-or-equal with tolerance.

        Uses ``self - other <= tol`` so values within ``tol`` are treated as equal.

        Args:
            other: Matrix-like object with the same type and shape as ``self``.
            tol: Absolute tolerance. If ``None``, uses ``type(self).eps()``.

        Returns:
            Matrix: Boolean matrix where each entry is ``True`` if
            ``self(i, j) <= other(i, j)`` within tolerance.

        Raises:
            InvalidDataError: If ``other`` is not the same matrix type as ``self``.
            InvalidDimensionsError: If the matrices have different dimensions.
        """
        op = "elementwise less than or equal"
        self._validate_other_type(other, operation=op)
        self._validate_same_size(other, operation=op)

        tol = type(self).eps() if tol is None else tol
        rows, cols = self.rows, self.cols
        return self.__class__([
            [self[i, j] - other[i, j] <= tol 
             for j in range(1, cols+1)]
             for i in range(1, rows+1)
        ])

    def elementwise_greater_than_or_equal(self, other, *, tol=None):
        """Test elementwise greater-than-or-equal with tolerance.

        Uses ``self - other >= -tol`` so values within ``tol`` are treated as equal.

        Args:
            other: Matrix-like object with the same type and shape as ``self``.
            tol: Absolute tolerance. If ``None``, uses ``type(self).eps()``.

        Returns:
            Matrix: Boolean matrix where each entry is ``True`` if
            ``self(i, j) >= other(i, j)`` within tolerance.

        Raises:
            InvalidDataError: If ``other`` is not the same matrix type as ``self``.
            InvalidDimensionsError: If the matrices have different dimensions.
        """
        op = "elementwise greater than or equal"
        self._validate_other_type(other, operation=op)
        self._validate_same_size(other, operation=op)

        tol = type(self).eps() if tol is None else tol
        rows, cols = self.rows, self.cols
        return self.__class__([
            [self[i, j] - other[i, j] >= -tol 
             for j in range(1, cols+1)]
             for i in range(1, rows+1)
        ])
    
