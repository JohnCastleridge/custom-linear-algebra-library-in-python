from typing import Self, Any, Callable
from math import log

from ..exceptions import (
    InvalidDimensionsError,
    NotSquareError,
    MatrixValueError,
    InvalidDataError,
)

class BinaryMatrixOperationsMixin:
    def matrix_addition(self, other: Self) -> Self:
        """Matrix-matrix addition.
        
        Computes ``C = A + B`` with ``Cᵢⱼ = Aᵢⱼ + Bᵢⱼ``.

        Args:
            other (Self): Matrix of the same type and shape as ``self``.
        
        Returns:
            Self: A new matrix containing the elementwise sum.
        
        Raises:
            InvalidDimensionsError: If the matrices have different dimensions.
            InvalidDataError: If ``other`` is not the same matrix type as ``self``.
        
        See Also:
            ``__add__``: Operator overload for ``A + B``.
        """
        operation="matrix_addition"
        self._validate_other_type(other, operation=operation)
        self._validate_same_size(other, operation=operation, reason="Matrices have different dimensions")

        rows, cols = self.rows, self.cols
        return self.__class__([
             [self[row,col] + other[row,col] 
              for col in range(1, cols+1)]
              for row in range(1, rows+1)
        ])

    def scalar_addition(self, scalar: Any) -> Self:
        """Matrix-scalar addition.

        Computes ``C = s + A`` with ``Cᵢⱼ = s + Aᵢⱼ``.

        Args:
            scalar (Any): Scalar value to add to each entry.

        Returns:
            Self: A new matrix where each element equals ``scalar + Aᵢⱼ``.

        See Also:
            ``__add__``: Operator overload for ``A + s``.
            ``__radd__``: Supports ``s + A`` using Python's addition.
        """
        # chek scalar
        rows, cols = self.rows, self.cols
        return self.__class__([
             [scalar + self[row,col]
              for col in range(1, cols+1)] 
              for row in range(1, rows+1)
        ])

    def matrix_multiplication(self, other: Self) -> Self:
        """Matrix-matrix multiplication.

        Computes the product ``C = A·B`` with ``Cᵢⱼ = ∑ᵣ₌₁ⁿ Aᵢᵣ ⋅ Bᵣⱼ``.

        Only defined when the number of columns of ``A`` equals the number of rows of ``B``.

        Args:
            other (Self): Right-hand-side matrix.

        Returns:
            Self: The matrix product ``A·B``.

        Raises:
            InvalidDimensionsError: If ``self.cols != other.rows``.
            InvalidDataError: If ``other`` is not the same matrix type as ``self``.

        See Also:
            ``__mul__``: Operator overload for ``A * B``.
        """
        operation="matrix_multiplication"
        self._validate_other_type(other, operation=operation)
        if self.cols != other.rows:
            raise InvalidDimensionsError(self, other, 
                operation="matrix_multiplication",
                reason="column count of first ≠ row count of last"
            )
        
        self_rows, other_cols = self.rows, other.cols
        k = self.cols
        return self.__class__([
            [sum(self[i,r]*other[r,j] for r in range(1, k+1))
             for j in range(1, other_cols+1)] 
             for i in range(1, self_rows+1)
        ])

    def scalar_multiplication(self, scalar: Any) -> Self:
        """Multiply every element by a scalar (elementwise scaling).

        Computes ``C = s·A`` with ``Cᵢⱼ = s·Aᵢⱼ``.

        Args:
            scalar (Any): Scalar value to multiply each entry by.

        Returns:
            Self: A new matrix where each element equals ``scalar * Aᵢⱼ``.

        See Also:
            ``__mul__``: Operator overload for ``A * s``.
            ``__rmul__``: Supports ``s * A`` via Python's multiplication.

        """
        # chek scalar
        rows, cols = self.rows, self.cols
        return self.__class__([
             [scalar * self[row,col]
              for col in range(1, cols+1)] 
              for row in range(1, rows+1)
        ])

    def matrix_exponentiation(self, exponent: int) -> Self:
        """Integer power of a square matrix.

        Implements ``A**k`` for integer ``k`` using recursion:

        - ``k < 0`` uses ``A^{-k} = (A^{-1})**k`` (requires ``A`` invertible).
        - ``k = 0`` returns the identity matrix of matching size.
        - ``k > 0`` computes ``A**(k-1) * A``.

        Args:
            exponent (int): Integer exponent ``k``.

        Returns:
            Self: ``A`` raised to the integer power ``k``.

        Raises:
            NotSquareError: If ``A`` is not square.
            InvalidDataError: If ``exponent`` is not an ``int``.
            SingularMatrixError: May be raised indirectly when ``exponent < 0`` and
            ``A`` is non-invertible (via ``self.inv``).

        See Also:
            ``__pow__``: Operator overload for ``A ** k``.
        """
        operation="matrix_exponentiation"
        if not self._is_square():
            raise NotSquareError(self, operation=operation)
        if not isinstance(exponent, int):
            raise InvalidDataError(obj=exponent, expected_type='int', operation=operation, reason='"exponent" must be an integer')
        
        if exponent < 0:
            return self.inv.matrix_exponentiation(-exponent)
        if exponent == 0:
            return type(self).I(self.rows)
        return self.matrix_exponentiation(exponent-1) * self

    def scalar_exponentiation(self, base: Any, *, ln: Callable | None = None) -> Self:
        """Scalar-to-matrix power ``base**A`` via ``exp(log(base) * A)``.

        Interprets the scalar-matrix power using the identity
        ``b**A = exp((ln b)·A)``. This requires ``A`` to be square.

        Args:
            base (Any): Real/complex base ``b``.

        Returns:
            Self: The matrix ``exp((ln base)·A)``.

        Raises:
            MatrixValueError: If ``base <= 0`` according to the current check.
            NotSquareError: If ``A`` is not square.

        See Also:
            ``__rpow__``: Supports ``base ** A`` via Python's power function.``
        """
        operation="scalar_exponentiation"
        if base <= 0 and ln is None:
            raise MatrixValueError(value=base, operation=operation, reason='"base" must be strictly positive')
        if not self._is_square():
            raise NotSquareError(self, operation=operation)
        
        if ln is None:
            ln = lambda base: log(base)

        if abs(base-1) < type(self).eps():
            return type(self).I(self.rows)
        return (ln(base)*self).math.exp()

    def hadamard_product(self, other: Self) -> Self:
        """Elementwise (Hadamard) product ``A ∘ B``.

        Multiplies matrices element by element. Intended to require shapes to match exactly.

        Args:
            other (Self): Matrix with the same shape as ``self``.

        Returns:
            Self: A new matrix where ``Cᵢⱼ = Aᵢⱼ · Bᵢⱼ``.

        Raises:
            InvalidDimensionsError: If the matrices have different dimensions. 
        """
        operation="hadamard_product"
        self._validate_other_type(other, operation=operation)
        self._validate_same_size(other, operation=operation, reason="Matrices have different dimensions")
        
        rows, cols = self.rows, self.cols
        return self.__class__([
             [self[row,col] * other[row,col]
              for col in range(1, cols+1)] 
              for row in range(1, rows+1)
        ])

    def kronecker_product(self, other: Self) -> Self:
        """Kronecker (tensor) product ``A ⊗ B``.

        Constructs the block matrix defined by
        ``[A ⊗ B]_{pr+v, qs+w} = a_{r,s} · b_{v,w}``.

        Args:
            other (Self): Right-hand-side matrix ``B``.

        Returns:
            Self: The Kronecker product ``A ⊗ B``.
        """
        operation="kronecker_product"
        self._validate_other_type(other, operation=operation)

        self_rows, self_cols = self.rows, self.cols
        other_rows, other_cols = other.rows, other.cols
        return self.__class__([
            [self[r, s] * other[v, w]
             for s in range(1, self_cols+1)
             for w in range(1, other_cols+1)]
             for r in range(1, self_rows+1)
             for v in range(1, other_rows+1)
            ])

    def map(self, func: Callable[[Any], Any]) -> Self:
        """Apply a callable to each element and return the result.

        Args:
            func (Callable): Function ``f(x)`` applied to every element.

        Returns:
            Self: New matrix where each entry is ``func(self[i, j])``.
        """
        rows, cols = self.rows, self.cols
        return self.__class__([
             [func(self[row,col])
              for col in range(1, cols+1)] 
              for row in range(1, rows+1)
        ])

    def augment(self, other: Self) -> Self:

        op = "augment"
        self._validate_other_type(other, operation = op, reason = 'Operand must be an "Matrix"')
        if self.rows != other.rows:
            raise InvalidDimensionsError(self, other, operation=op, reason="Matrices do not have the same number of rows")

        return self.__class__([
             [self[i,j] if j<=self.cols else other[i,j-self.cols] 
              for j in range(1, self.cols+other.cols+1)]
              for i in range(1, self.rows+1)
        ])