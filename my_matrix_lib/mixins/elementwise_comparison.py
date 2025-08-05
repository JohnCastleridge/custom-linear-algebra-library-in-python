from ..exceptions import (
    InvalidDimensionsError,
    InvalidDataError,
)

class ElementwiseComparisonMixin:
    def elementwise_equal(self, other, tol=1e-8):
        """
        Elementwise equality. Returns boolean matrix.
        """
        if self._have_same_size(other):
            raise InvalidDimensionsError(self, other, 
                operation="elementwise equality",
                reason="Matrices have different dimensions"
            )
            
        return self.__class__([
            [abs(self(i, j) - other(i, j)) <= tol
            for j in range(self.cols)]
            for i in range(self.rows)
        ])
    
    def elementwise_not_equal(self, other, tol=1e-8):
        """
    	Elementwise inequality. Returns boolean matrix.
        """
        if self._have_same_size(other):
            raise InvalidDimensionsError(self, other, 
                operation="elementwise inequality",
                reason="Matrices have different dimensions"
            )
        return self.__class__([
            [abs(self(i, j) - other(i, j)) > tol
            for j in range(self.cols)]
            for i in range(self.rows)
        ])
    
    def elementwise_less_than(self, other, tol=1e-8):
        """
    	Elementwise less than. Returns boolean matrix.
        """
        if self._have_same_size(other):
            raise InvalidDimensionsError(self, other, 
                operation="elementwise less than",
                reason="Matrices have different dimensions"
            )
        return self.__class__([
            [self(i, j) - other(i, j) < -tol
            for j in range(self.cols)]
            for i in range(self.rows)
        ])
    
    def elementwise_greater_than(self, other, tol=1e-8):
        """
    	Elementwise greater than. Returns boolean matrix.
        """
        if self._have_same_size(other):
            raise InvalidDimensionsError(self, other, 
                operation="elementwise greater than",
                reason="Matrices have different dimensions"
            )
        return self.__class__([
            [self(i, j) - other(i, j) > tol
            for j in range(self.cols)]
            for i in range(self.rows)
        ])
    
    def elementwise_less_than_or_equal(self, other, tol=1e-8):
        """
    	Elementwise less than or equal. Returns boolean matrix.
        """
        if self._have_same_size(other):
            raise InvalidDimensionsError(self, other, 
                operation="elementwise less than or equal",
                reason="Matrices have different dimensions"
            )
        return self.__class__([
            [self(i, j) - other(i, j) <= tol
            for j in range(self.cols)]
            for i in range(self.rows)
        ])
    
    def elementwise_greater_than_or_equal(self, other, tol=1e-8):
        """
    	Elementwise greater than or equal. Returns boolean matrix.
        """
        if self._have_same_size(other):
            raise InvalidDimensionsError(self, other, 
                operation="elementwise greater than or equal",
                reason="Matrices have different dimensions"
            )
        return self.__class__([
            [self(i, j) - other(i, j) >= -tol
            for j in range(self.cols)]
            for i in range(self.rows)
        ]) 
    
