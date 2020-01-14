import time
from IPython.core.magic import cell_magic, magics_class, Magics
from IPython.core import magic_arguments

def timeit(key, op, fn):
    t = time.time()
    res = fn()
    c = int(time.time() * 1000) # current time in ms
    e = int(c - (t * 1000)) # elapsed ms
    print("Elapsed time: {:.1f} seconds".format(e/1000))
    with open('benchmarks.csv', 'a') as fd:
        fd.write('{}\t{}\t{}\t{}\n'.format(c, key, op, e))
    return res
    
@magics_class
class TimeOpMagic(Magics):
    
    def __init__(self, shell, key):
        super(TimeOpMagic, self).__init__(shell)
        self.key = key
        
    @magic_arguments.magic_arguments()
    @magic_arguments.argument('--op', '-o', action='store', help='Name of op')
    @cell_magic
    def timeop(self, line='', cell=None):
        args = magic_arguments.parse_argstring(self.timeop, line)
        return timeit(self.key, args.op, lambda: self.shell.ex(cell))
    
def register_timeop_magic(ip, key):
    ip.register_magics(TimeOpMagic(ip, 'hail'))