from ..exceptions import (
    InvalidDimensionsError,
    NotSquareError
)

class BinaryMatrixOperationsMixin:
    def matrix_addition(self, other):
        """
        element wise addition of two matrices with the same size
        
        [A+B]ᵢⱼ = Aᵢⱼ + Bᵢⱼ
        """
        if self._have_same_size(other):
            raise InvalidDimensionsError(self, other, 
                operation="matrix addition",
                reason="Matrices have different dimensions"
            )
        return self.__class__([
             [self(row,col) + other(row,col) 
              for col in range(self.cols)] 
              for row in range(self.rows)
        ])

    def scalar_addition(self, scaler):
        return self.__class__([
             [scaler + self(row,col)
              for col in range(self.cols)] 
              for row in range(self.rows)
        ])

    def matrix_multiplication(self, other):
        """
        matrix multiplication
        
        [AB]ᵢⱼ = ∑ᵣ₌₁ⁿ Aᵢᵣ⋅ Bᵣⱼ
        
        Only defined for matrices with where the number of columns in the first is the same as the number of rows in the last
        """
        if self.cols != other.rows:
            raise InvalidDimensionsError(self, other, 
                operation="matrix multiplication",
                reason="column count of first ≠ row count of last"
            )
        return self.__class__([
            [sum(self(i,r)*other(r,j) for r in range(self.cols))
             for j in range(other.cols)] 
             for i in range(self.rows)
        ])

    def scalar_multiplication(self, scaler):
        return self.__class__([
             [scaler * self(row,col)
              for col in range(self.cols)] 
              for row in range(self.rows)
        ])

    def matrix_exponentiation(self, n: int):
        if not self._is_square():
            raise NotSquareError(self, 
                operation="matrix exponentiation"
            )
        if not isinstance(n, int):
            pass
        
        if n < 0:
            return self.inv ** -n
        if n == 0:
            return self.I(self.rows)
        return self.matrix_exponentiation(n-1) * self

    def scalar_exponentiation(self, base):
        if not self._is_square():
            raise NotSquareError(self, 
                operation="scalar exponentiation of matrix"
            )
        
        if base >= 0:
            pass

        return (_ln(base)*self).math.exp()

    def hadamard_product(self, other):
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

    def kronecker_product(self, other):
        """
        kronecker product
        
        [A ⊗ B]_{pr+v,qs+w} = a_{r,s}⋅ {b_v,w}
        """
        return self.__class__([
            [self(r, s) * other(v, w)
             for s in range(self.cols)
             for w in range(other.cols)]
             for r in range(self.rows)
             for v in range(other.rows)
            ])

# === utils ===
def _ln(x, tol=1e-9, max_iterations=1000):
    """
    Compute ln(x) by Newton’s method solving
    f(y) = e^y - x = 0
    """
    e = 2.71828182846
    y = x-1 # y_0 guess
    for _ in range(max_iterations):
        delta = x*e**(-y)-1
        y += delta
        if abs(delta) < tol:
            return y
    raise RuntimeError(f'"ln" failed to converge for x={x} with a tolerance of {tol} after {max_iterations} iterations')



