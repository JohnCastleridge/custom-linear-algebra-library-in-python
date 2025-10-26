from typing import Self, Any

from ..exceptions import (
    InvalidDataError,
    IndexOutOfBoundsError,
)

class ElementaryOperationsMixin:
    # === Elementary Row Operations ===
    def row_switching(self, i: int, j: int) -> Self:
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
            self._data[i-1] if idx == j-1 else 
            self._data[j-1] if idx == i-1 else 
            row[:] 
            for idx, row in enumerate(self._data)
        ])

    def row_multiplication(self, i: int, k: Any=1) -> Self:
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
            [k*a for a in self._data[i-1]] if idx == i-1 else 
            row[:] 
            for idx, row in enumerate(self._data)
        ])

    def row_addition(self, i: int, j: int, k: Any=1) -> Self:
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
            [self._data[i-1][idx]+k*self._data[j-1][idx] for idx in range(self.cols)] 
            if row == self._data[i-1] else row[:] 
            for row in self._data
        ])
    
    # === Elementary Column Operations ===
    def column_switching(self, i: int, j: int) -> Self:
        return self.T.row_switching(i, j).T

    def column_multiplication(self, i: int, k: Any=1) -> Self:
        return self.T.row_multiplication(i, k).T

    def column_addition(self, i: int, j: int, k: Any=1) -> Self:
        return self.T.row_addition(i, j, k).T

    # === NoName ===
    def reduced_row_echelon_form(self) -> Self:
        M = self
        pivot = 1
        for j in range(1, self.cols+1):
            find_pivot = False
            for i in range(pivot, self.rows+1):
                if abs(M[i,j]) >= self.eps(): # chek if the elemnt we are tryng to make to an piviot elemnt is zero
                    M = M.row_switching(i, pivot)
                    M = M.row_multiplication(pivot,M[pivot,j]**-1)
                    find_pivot = True
                    break

            if find_pivot:
                for i in range(1, self.rows+1):
                    if i == pivot:
                        continue
                    M = M.row_addition(i, pivot, -M[i,j])
                pivot += 1
        
        return M

    # === NoName ===
    def rank(self) -> int:
        pass
    
    def nullity(self) -> int:
        pass

    # 
    RREF = property(reduced_row_echelon_form)
    