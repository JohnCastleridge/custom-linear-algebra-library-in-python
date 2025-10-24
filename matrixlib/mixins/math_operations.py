from ..exceptions import (
    InvalidDimensionsError,
    NotSquareError
)

class MathNamespace:
    def __init__(self, parent):
        self._M = parent
    
    # === Exponential and Natural logarithm ===
    def exp(self, terms=100):
        """
        Compute the matrix exponential eᴹ using its Taylor series expansion.

        eᴹ = ∑ Mⁿ⁄n!
        """
        if not self._M._is_square():
            raise NotSquareError(self._M, 
                operation="exp"
            )
        return sum(self._M**n / _fact(n) for n in range(terms))
    
    def log(self, terms=100):
        """
        Compute the matrix natural logarithm ln(M) using the Mercator series / Taylor series expansion.

        ln(M) = ∑ (-1)^(n+1)⁄n Mⁿ
        
        Converges for matrices whose eigenvalues lie in the interval [-1, 1]
        """
        if not self._M._is_square():
            raise NotSquareError(self._M, 
                operation="log"
            )
        #  chek Spectral radius of A - I < 1
        return sum((-1)**(n+1) * self._M**n / n for n in range(terms))
    
    # === Trigonometric functions
    def sin(self, terms=50):
        """
        Compute the matrix sine sin(M) using its Taylor series expansion.

        sin(M) = ∑ (-1)ⁿ M^(2n+1)⁄(2n+1)!
        """
        if not self._M._is_square():
            raise NotSquareError(self._M, 
                operation="sin"
            )
        return sum(self._M**(2*n+1) * (-1)**n / _fact(2*n+1) for n in range(terms))

    def cos(self, terms=50):
        """
        Compute the matrix cosine cos(M) using its Taylor series expansion.

        cos(M) = ∑ (-1)ⁿ M^(2n)⁄(2n)!
        """
        if not self._M._is_square():
            raise NotSquareError(self._M, 
                operation="cos"
            )
        return sum(self._M**(2*n) * (-1)**n / _fact(2*n) for n in range(terms))

    def tan(self, terms=50):
        """
        Compute the matrix tangent tan(M) using the Taylor series expansion of sine and cosine.
        
        tan(M) = sin(M) / cos(M)
        """
        if not self._M._is_square():
            raise NotSquareError(self._M, 
                operation="tan"
            )
        return self._M.sin(terms=terms) / self._M.cos(terms=terms)
    
    
    def sec(self, terms=50):
        """
        
        """
        if not self._M._is_square():
            raise NotSquareError(self._M, 
                operation="sec"
            )
        return self._M.cos(terms=terms).inv
    
    def arcsin(self, terms=50):
        """
        Compute the matrix arcsine arcsin(M) using its Taylor series expansion.

        arcsin(M) = ∑ (2n)! / (4^n (n!)^2 (2n+1)) M^(2n+1)
        
        Converges for matrices whose eigenvalues lie in the interval [-1, 1]
        """
        #  chek Spectral radius of A < 1
        if not self._M._is_square():
            raise NotSquareError(self._M, 
                operation="arcsin"
            )
        return sum(_fact(2*n) * self._M**(2*n+1) / (4**n * (_fact(n))**2 * (2*n+1)) for n in range(terms))
    
    def arccos(self, terms=50):
        """
        Compute the matrix arccosine arccos(M) using the Taylor series expansion of arcsine.
        
        arccos(M) = π/2 - arcsin(M)
        
        Converges for matrices whose eigenvalues lie in the interval [-1, 1]
        """
        if not self._M._is_square():
            raise NotSquareError(self._M, 
                operation="arccos"
            )
        pi = 3.14159265359
        return pi/2 - self._M.arcsin(terms=terms)
    
    def arctan(self, terms=100):
        """
        Compute the matrix arctangent arctan(M) using its Taylor series expansion.

        arctan(M) = ∑ (-1)ⁿ⁄(2n+1) M^(2n+1)
        """
        if not self._M._is_square():
            raise NotSquareError(self._M, 
                operation="arctan"
            )
        return sum(self._M**(2*n+1) * (-1)**n / (2*n+1) for n in range(terms))
        
    # === Hyperbolic functions ===
    def sinh(self, terms=50):
        """
        Compute the matrix hyperbolic sine sinh(M) using its Taylor series expansion.

        sinh(M) = ∑ M^(2n+1)⁄(2n+1)!
        """
        if not self._M._is_square():
            raise NotSquareError(self._M, 
                operation="sinh"
            )
        return sum(self._M**(2*n+1) / _fact(2*n+1) for n in range(terms))

    def cosh(self, terms=50):
        """
        Compute the matrix hyperbolic sine sinh(M) using its Taylor series expansion.

        sinh(M) = ∑ M^(2n+1)⁄(2n+1)!
        """
        if not self._M._is_square():
            raise NotSquareError(self._M, 
                operation="cosh"
            )
        return sum(self._M**(2*n) / _fact(2*n) for n in range(terms))
    
    def tanh(self, terms=50):
        """
        Compute the matrix hyperbolic tangent tanh(M) using the Taylor series expansion of hyperbolic sine and cosine.
        
        tanh(M) = sinh(M) / cosh(M)
        """
        if not self._M._is_square():
            raise NotSquareError(self._M, 
                operation="tanh"
            )
        return self._M.sinh(terms=terms) / self._M.cosh(terms=terms)
    
    def arsinh(self, terms=50):
        """
        Compute the matrix hyperbolic arc sine arsinh(M) using its Taylor series expansion.

        arsinh(M) = ∑ (-1)ⁿ (2n)! / (4^n (n!)^2 (2n+1)) M^(2n+1)
        
        This converges for matrices with spectral radius ρ(M) < 1.
        """
        if not self._M._is_square():
            raise NotSquareError(self._M, 
                operation="arsinh"
            )
        return sum((-1)**n * _fact(2*n) * self._M**(2*n+1) / (4**n * (_fact(n))**2 * (2*n+1)) for n in range(terms))  
    
    def arcosh(self, terms=50):
        """
        Compute the matrix hyperbolic arc cosine arcosh(M) using its Taylor series expansion.

        arcosh(M) = ∑ (2n)! / (4^n (n!)^2 (2n)) M^(-2n)
        
        Converges for matrices whose eigenvalues lie in the interval [-1, 1]
        """
        if not self._M._is_square():
            raise NotSquareError(self._M, 
                operation="arcosh"
            )
        return (self._M*2).math.log(terms=terms) - sum(_fact(2*n) * self._M**(-2*n) / (4**n * (_fact(n))**2 * (2*n)) for n in range(terms))  
    
    def artanh(self, terms=100):
        """
        Compute the matrix hyperbolic arc tangent artanh(M) using its Taylor series expansion.

        artanh(M) = ∑ M^(2n+1) ⁄ (2n+1) 
        """
        if not self._M._is_square():
            raise NotSquareError(self._M, 
                operation="artan"
            )
        return sum(self._M**(2*n+1) / (2*n+1) for n in range(terms))
    

class MatrixMathMixin:
    @property
    def math(self) -> MathNamespace:
        """Access advanced matrix functions"""
        return MathNamespace(self)


# === utils ===
def _fact(n: int) -> int:
    if n==0:
        return 1
    return n * _fact(n-1)

