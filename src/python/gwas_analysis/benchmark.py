from IPython.core import magic_arguments
from IPython.core.magic import cell_magic, Magics, magics_class

import os
import os.path as osp
import time


SESSION = int(time.time() * 1000)
DEFAULT_PATH_FORMAT = "benchmarks/benchmarks_{key}_{session}.csv"


def timeit(key, op, fn, path_format=DEFAULT_PATH_FORMAT):
    t = time.time()
    res = fn()
    c = int(time.time() * 1000)  # current time in ms
    e = int(c - (t * 1000))  # elapsed ms
    print("Elapsed time: {:.1f} seconds".format(e / 1000))
    path = path_format.format(key=key, session=SESSION)
    if not osp.exists(osp.dirname(path)):
        os.makedirs(osp.dirname(path))
    with open(path, "a") as fd:
        fd.write("\t".join(map(str, [SESSION, c, key, op, e])) + "\n")
    return res


@magics_class
class TimeOpMagic(Magics):
    def __init__(self, shell, key, path_format=DEFAULT_PATH_FORMAT):
        super(TimeOpMagic, self).__init__(shell)
        self.key = key
        self.path_format = path_format

    @magic_arguments.magic_arguments()
    @magic_arguments.argument("--op", "-o", action="store", help="Name of op")
    @cell_magic
    def timeop(self, line="", cell=None):
        args = magic_arguments.parse_argstring(self.timeop, line)
        # See: https://stackoverflow.com/questions/53204167/is-it-possible-to-combine-magics-in-ipython-jupyter
        # for why this can be used to chain magics (this one must come first)
        # Note that run_cell(cell) returns ExecutionResult
        return timeit(
            self.key,
            args.op,
            lambda: self.shell.run_cell(cell).result,
            path_format=self.path_format,
        )


def register_timeop_magic(ip, key):
    ip.register_magics(TimeOpMagic(ip, key))
