
import os
from pkg_resources import resource_stream, Requirement
def read_from_egg(tfile):
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), tfile)
    if not os.path.exists(template_path):
        path_in_egg = os.path.join("vst", tfile)
        buf = resource_stream(Requirement.parse("vst"), path_in_egg)
        _bytes = buf.read()
        tf = Template(str(_bytes))
    else:
        with open(template_path, 'r') as f:
            tf = Template(f.read())
    return tf

class IAMLPConfigError(ValueError):
    pass