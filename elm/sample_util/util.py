
import attr

@attr.s
class Sample(object):
    x = attr.ib()
    y = attr.ib(default=None)
    sample_weight = attr.ib(default=None)
