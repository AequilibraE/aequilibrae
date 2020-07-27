class SafeClass:
    def __init__(self, data_set: dict) -> None:
        self.__original__ = {}
        for k, v in data_set.items():
            self.__dict__[k] = v
            self.__original__[k] = v

    def __copy__(self):
        raise Exception('Link Types object cannot be copied')

    def __deepcopy__(self, memodict=None):
        raise Exception('Link Types object cannot be copied')
