from ..exceptions import (
    InvalidDimensionsError,
    InvalidDataError,
)

class BooleanLogicMixin:
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