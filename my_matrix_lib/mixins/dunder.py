from ..exceptions import (
    IndexOutOfBoundsError,
    InvalidDataError,
    InvalidShapeError,
)

class DunderMixin:
    # === Dunder Operations ===
    def __add__(self, other): # +
        if isinstance(other, self.__class__):
            return self.matrix_addition(other)
        return self.scalar_addition(other)
        
    def __sub__(self, other): # -
        return self + -other

    def __mul__(self, other): # *
        if isinstance(other, self.__class__):
            return self.matrix_multiplication(other)
        return self.scalar_multiplication(other)

    def __truediv__(self, other): # /
        if isinstance(other, self.__class__):
            return self * other.inv
        return self * (1/other)

    def __pow__(self, other): # **
        return self.matrix_exponentiation(other)

    def __invert__(self): # ~
        return self.inv

    def __pos__(self): # +self
        return self

    def __neg__(self): # -self
        return self * -1


    # === Dunder NoName ===
    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        if not isinstance(other, self.__class__):
            return self.__mul__(other)

    def __rpow__(self, other):
        return self.scalar_exponentiation(other)


    # === Comparison Dunder Methods ===
    def __eq__(self, other): # =
        return self.elementwise_equal(other)

    def __ne__(self, other): # !=
        return self.elementwise_not_equal(other)
    
    def __lt__(self, other): # <
        return self.elementwise_less_than(other)
    
    def __gt__(self, other): # >
        return self.elementwise_greater_than(other)
    
    def __le__(self, other): # <=
        return self.elementwise_less_than_or_equal(other)
    
    def __ge__(self, other): # >=
        return self.elementwise_greater_than_or_equal(other)

    
    def __or__(self, other): # |
        return self.elementwise_OR(other)

    def __and__(self, other): # &&
        return self.elementwise_AND(other)
    
    # === N ===
    def __contains__(self, item): # item "in" matrix
        return item in [entry for row in self.data for entry in row]
    
    def __iter__(self): # for entry "in" matrix
        return (entry for row in self.data for entry in row)

    # === Indexing & Callable Access ===
    def __getitem__(self, idx: tuple[int]) -> any:
        # allow m[i, j]
        if not isinstance(idx, tuple) or not all(isinstance(i, int) for i in idx):
            raise InvalidDataError(idx, 'tuple[int]', operation='Matrix.__getitem__')
        if not len(idx) == 2:
            raise InvalidShapeError(idx, (2,), operation='Matrix.__getitem__')
        i, j = idx
        if (i-1) not in range(self.rows):
            raise IndexOutOfBoundsError(self, i, axis = 'row', operation='Matrix.__getitem__', reason='The first index is out of bounds')
        if (j-1) not in range(self.cols):
            raise IndexOutOfBoundsError(self, j, axis = 'col', operation='Matrix.__getitem__', reason='The second index is out of bounds')
        
        # convert from input 1-based to internal 0-based
        return self.data[i-1][j-1]


    def __setitem__(self, idx: tuple[int], entry: any) -> None:
        # allow m[i, j] = entry
        if not isinstance(idx, tuple) or not all(isinstance(i, int) for i in idx):
            raise InvalidDataError(idx, 'tuple[int]', operation='Matrix.__setitem__')
        if not len(idx) == 2:
            raise InvalidShapeError(idx, (2,), operation='Matrix.__getitem__')
        i, j = idx
        if (i-1) not in range(self.rows):
            raise IndexOutOfBoundsError(self, i, axis = 'row', operation='Matrix.__setitem__', reason='The first index is out of bounds')
        if (j-1) not in range(self.cols):
            raise IndexOutOfBoundsError(self, j, axis = 'col', operation='Matrix.__setitem__', reason='The second index is out of bounds')
        
        # convert from input 1-based to internal 0-based
        self.data[i-1][j-1] = entry


    def __call__(self, i, j):
        # allow m(i, j) as alternetiv shorthand, but NB! with 0-indexing
        return self[i+1, j+1]
