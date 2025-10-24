from .core import Matrix

__all__ = ["I", "O","det","tr","exp","sin","cos","sinh","cosh"]

I = Matrix.identity
O = Matrix.zeros
det = lambda matrix: matrix.det
tr = lambda matrix: matrix.tr

# === Math functions on matrices ===
exp = lambda matrix, terms=100: matrix.math.exp(terms=terms)
sin = lambda matrix, terms=50: matrix.math.sin(terms=terms)
cos = lambda matrix, terms=50: matrix.math.cos(terms=terms)
sinh = lambda matrix, terms=50: matrix.math.sinh(terms=terms)
cosh = lambda matrix, terms=50: matrix.math.cosh(terms=terms)