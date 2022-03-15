
class DifferentiableFunction:
    # attributes:
    # function ptrs:
    # calc = None
    # calcFirstDerivative = None

    # forbid instances of this class
    def __new__(cls, *args, **kwargs):
        if cls is DifferentiableFunction:
            msg = "Objects of this class cannot be created.\n" \
                + "Use classes derived from this one.\n"
            raise TypeError(msg)
        return object.__new__(cls)

    def __init__(self, function, firstDerivativeFunction):
        self.calc = function
        self.calcFirstDerivative = firstDerivativeFunction


class DifferentiableFunctionException(Exception):
    # attributes:
    # message = None
    # original_exception_msg = None

    def __init__(self, msg, original_exception_msg):
        super().__init__(msg)

        self.original_exception_msg = original_exception_msg
        self.message = msg

    def __str__(self):
        return self.message + ':\n' + self.original_exception_msg + '\n\n'
