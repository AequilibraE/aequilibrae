import yaml
import os
from aequilibrae.parameters import Parameters


class Link:
    def __init__(self, link_id=None):
        p = Parameters()
        fields = p.parameters["network"]["links"]["fields"]
        one_way_fields = [list(x.keys())[0] for x in fields["one-way"]]
        one_way_fields = {x: None for x in one_way_fields}

        twf = [list(x.keys())[0] for x in fields["two-way"]]
        two_way_fields = {f"{x}_ab": None for x in twf}
        two_way_fields.update({f"{x}_ba": None for x in twf})

        one_way_fields.update(two_way_fields)
        self.__dict__.update(one_way_fields)

        if link_id is not None:
            self.link_id = link_id
            self._populate()

    def _populate(self):
        print(self.link_id)
