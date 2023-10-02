from attrs import asdict, astuple


class Cfg:
    def asdict(self):
        return asdict(self)

    def astuple(self):
        return astuple(self)
