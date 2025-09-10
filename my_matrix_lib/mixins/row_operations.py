from ..exceptions import (
    InvalidDataError,
    IndexOutOfBoundsError,
)

class MatrixRowOperationsMixin:
    """Elementary row operations (EROs) for matrices.

    Notes:
        - Row indices are **1-based** in the public API (i.e., the top row is ``1``).
        - All operations return a **new** matrix; the original is not modified.
    """

    # === Elementary Row Operations ===
    def row_switching(self, i: int, j: int):
        """
        Swap two rows (``Rᵢ ↔ Rⱼ``).

        Replaces row ``i`` with row ``j`` and row ``j`` with row ``i``.

        Args:
            i (int): 1-based index of the first row to swap.
            j (int): 1-based index of the second row to swap.

        Returns:
            Matrix: A new matrix with rows ``i`` and ``j`` exchanged.

        Raises:
            InvalidDataError: If ``i`` or ``j`` is not an integer.
            IndexOutOfBoundsError: If ``i`` or ``j`` is outside ``[1 .. rows]``.
        """
        if not isinstance(i, int):
            raise InvalidDataError(obj=i, expected_type='int', operation='row switching', reason='The first input has not type: "int"')
        if not isinstance(j, int):
            raise InvalidDataError(obj=j, expected_type='int', operation='row switching', reason='The second input has not type: "int"')
        if (i-1) not in range(self.rows) or (j-1) not in range(self.rows):
            raise IndexOutOfBoundsError(matrix=self, index=(i, j), operation='row switching', reason='At least one of the indices is out of bounds')

        return self.__class__([
            self.data[i-1] if idx == j-1 else 
            self.data[j-1] if idx == i-1 else 
            row[:] 
            for idx, row in enumerate(self.data)
        ])

    def row_multiplication(self, i: int, k):
        """
        Scale a row by a nonzero scalar (``k⋅Rᵢ → Rᵢ``).

        Args:
            i (int): 1-based index of the row to scale.
            k: Scalar multiplier.

        Returns:
            Matrix: A new matrix with row ``i`` multiplied by ``k``.

        Raises:
            InvalidDataError: If ``i`` is not an integer.
            IndexOutOfBoundsError: If ``i`` is outside ``[1 .. rows]``.
            ValueError: If ``|k| < 1e-8`` (treated as zero).
        """
        if not isinstance(i, int):
            raise InvalidDataError(obj=i, expected_type='int', operation='row multiplication', reason='The index input has an invalid type')
        if (i-1) not in range(self.rows):
            raise IndexOutOfBoundsError(matrix=self, index=i, operation='row multiplication')
        if abs(k) < 1e-8:
            raise ValueError('Can not multiply row with 0')

        return self.__class__([
            [k*a for a in self.data[i-1]] if idx == i-1 else 
            row[:] 
            for idx, row in enumerate(self.data)
        ])

    def row_addition(self, i: int, j: int, k):
        """
        Add a multiple of one row to another (``Rᵢ + k⋅Rⱼ → Rᵢ``).

        Args:
            i (int): 1-based index of the destination row (modified in-place conceptually).
            j (int): 1-based index of the source row.
            k: Scalar multiplier applied to row ``j`` before addition.

        Returns:
            Matrix: A new matrix where row ``i`` is replaced by ``Rᵢ + k⋅Rⱼ``.

        Raises:
            InvalidDataError: If ``i`` or ``j`` is not an integer.
            IndexOutOfBoundsError: If ``i`` or ``j`` is outside ``[1 .. rows]``.
        """
        if not isinstance(i, int):
            raise InvalidDataError(obj=i, expected_type='int', operation='row addition', reason='The first input has an invalid type')
        if not isinstance(j, int):
            raise InvalidDataError(obj=j, expected_type='int', operation='row addition', reason='The second input has an invalid type')
        if (i-1) not in range(self.rows) or (j-1) not in range(self.rows):
            raise IndexOutOfBoundsError(matrix=self, index=(i, j), operation='row addition')
        
        return self.__class__([
            [self.data[i-1][idx]+k*self.data[j-1][idx] for idx in range(self.cols)] 
            if row == self.data[i-1] else row[:] 
            for row in self.data
        ])
    
    def row_division(self, i: int, k):
        """
        Divide a row by a nonzero scalar (``Rᵢ / k → Rᵢ``).

        Equivalent to scaling row ``i`` by ``1/k``.

        Args:
            i (int): 1-based index of the row to divide.
            k: Nonzero scalar divisor.

        Returns:
            Matrix: A new matrix with row ``i`` divided by ``k``.

        Raises:
            ZeroDivisionError: If ``k == 0``.
            InvalidDataError: If ``i`` is not an integer (raised by the underlying call).
            IndexOutOfBoundsError: If ``i`` is outside ``[1 .. rows]`` (raised by the underlying call).
            ValueError: If the implied multiplier ``1/k`` is treated as zero by tolerance (``|1/k| < 1e-8``).
        """
        return self.row_multiplication(i, 1/k)

    def row_subtraction(self, i: int, j: int, k):
        """
        Subtract a multiple of one row from another (``Rᵢ - k⋅Rⱼ → Rᵢ``).

        Equivalent to ``row_addition(i, j, -k)``.

        Args:
            i (int): 1-based index of the destination row (modified in-place conceptually).
            j (int): 1-based index of the source row.
            k: Scalar multiplier applied to row ``j`` before subtraction.

        Returns:
            Matrix: A new matrix where row ``i`` is replaced by ``Rᵢ - k⋅Rⱼ``.
        """
        return self.row_addition(i, j, -k)
