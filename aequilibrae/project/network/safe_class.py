import shapely.wkb


class SafeClass:
    def __init__(self, data_set: dict) -> None:
        self.__original__ = {}
        for k, v in data_set.items():
            if k == 'geometry' and v is not None:
                v = shapely.wkb.loads(v)
            self.__dict__[k] = v
            self.__original__[k] = v

    def __copy__(self):
        raise Exception('Object cannot be copied')

    def __deepcopy__(self, memodict=None):
        raise Exception('Object cannot be copied')
