class GlobalsMixin:
    
    _eps = 1e-8
    @classmethod
    def eps(cls):
        """Return the class-wide epsilon for this class."""
        return cls._eps