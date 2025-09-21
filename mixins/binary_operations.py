from typing import Callable, Self
from math import log

from ..exceptions import (
    InvalidDimensionsError,
    NotSquareError,
    MatrixValueError,
)

class BinaryMatrixOperationsMixin:
    def matrix_addition(self, other: Self) -> Self:
        """
        element wise addition of two matrices with the same size
        
        [A+B]ᵢⱼ = Aᵢⱼ + Bᵢⱼ
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

    def scalar_addition(self, scaler: any) -> Self:
        # chek scaler
        rows, cols = self.rows, self.cols
        return self.__class__([
             [scaler + self[row,col]
              for col in range(1, cols+1)] 
              for row in range(1, rows+1)
        ])

    def matrix_multiplication(self, other: Self) -> Self:
        """
        matrix multiplication
        
        [AB]ᵢⱼ = ∑ᵣ₌₁ⁿ Aᵢᵣ⋅ Bᵣⱼ
        
        Only defined for matrices with where the number of columns in the first is the same as the number of rows in the last
        """
        operation="matrix_multiplication"
        self._validate_other_type(other, operation=operation)
        if self.cols != other.rows:
            raise InvalidDimensionsError(self, other, 
                operation="matrix_multiplication",
                reason="column count of first ≠ row count of last"
            )
        
        rows, cols = self.rows, self.cols
        return self.__class__([
            [sum(self[i,r]*other[r,j] for r in range(1, cols+1))
             for j in range(1, cols+1)] 
             for i in range(1, rows+1)
        ])

    def scalar_multiplication(self, scaler: any) -> Self:
        # chek scaler
        rows, cols = self.rows, self.cols
        return self.__class__([
             [scaler * self[row,col]
              for col in range(1, cols+1)] 
              for row in range(1, rows+1)
        ])

    def matrix_exponentiation(self, exponent: int) -> Self:
        operation="matrix_exponentiation"
        if not self._is_square():
            raise NotSquareError(self, operation=operation)
        if not isinstance(exponent, int):
            raise InvalidDataError(obj=exponent, expected_type='int', operation=operation, reason='"exponent" must be an integer')
        
        if exponent < 0:
            return self.inv.matrix_exponentiation(-exponent)
        if exponent == 0:
            return self.I(self.rows)
        return self.matrix_exponentiation(exponent-1) * self

    def scalar_exponentiation(self, base: any) -> Self:
        """
        """
        operation="scalar_exponentiation"
        if base <= 0:
            raise MatrixValueError(value=base, operation=operation, reason='"base" must be an non-negetive')
        if not self._is_square():
            raise NotSquareError(self, operation=operation)
        
        return (log(base)*self).math.exp()

    def hadamard_product(self, other: Self) -> Self:
        operation="hadamard_product"
        self._validate_same_size(other, operation=operation, reason="Matrices have different dimensions")
        if self._have_same_size(other):
            raise InvalidDimensionsError(self, other, 
                operation="hadamard product",
                reason="Matrices have different dimensions"
            )
            
        return self.__class__([
             [self(row,col) * other(row,col)
              for col in range(self.cols)] 
              for row in range(self.rows)
        ])

    def kronecker_product(self, other: Self) -> Self:
        """
        kronecker product
        
        [A ⊗ B]_{pr+v,qs+w} = a_{r,s} ⋅ b_{v,w}
        """
        return self.__class__([
            [self(r, s) * other(v, w)
             for s in range(self.cols)
             for w in range(other.cols)]
             for r in range(self.rows)
             for v in range(other.rows)
            ])

    def map(self, func: Callable) -> Self:
        return self.__class__([
             [func(self.data[row][col])
              for col in range(self.cols)] 
              for row in range(self.rows)
        ])
