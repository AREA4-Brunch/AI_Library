
class DataHandlerException(Exception):
    # attributes:
    # message = None
    # original_exception_msg = None

    def __init__(self, msg, original_exception_msg):
        super().__init__(msg)

        self.original_exception_msg = original_exception_msg
        self.message = msg

    def __str__(self):
        return self.message + ':\n' + self.original_exception_msg + '\n\n'


class ErrorCleanData(DataHandlerException):
    message = "Cleaning Data Error"

    def __init__(self, original_exception_msg_):
        super().__init__(self.message, original_exception_msg_)
