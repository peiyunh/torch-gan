import sys
import re
from matplotlib import pyplot as plt

stamp = 'v1_mpii'
regex = re.compile('R:([0-9\.]+).*F:([0-9\.]+).*G:([0-9\.]+).*?(\d+) (\d+)')

if len(sys.argv) < 2:
    logfile = 'logs512_mpii64/2016-03-18-15-06.log'
else:
    logfile = sys.argv[1]

err_Rs = []
err_Fs = []
err_Gs = []
iters = []

with open(logfile) as f:
    lines = f.readlines()
    for l in lines:
        if not l.startswith(stamp):
            continue
        m = regex.search(l)
        R, F, G, iter1, iter2 = m.groups()
        
        # print err_R, err_F, err_G, iter1, iter2
        iters.append(iter2)
        err_Rs.append(R)
        err_Fs.append(F)
        err_Gs.append(G)


plt.hold(True)
h1, = plt.plot(iters, err_Rs, label='real')
h2, = plt.plot(iters, err_Fs, label='fake')
h3, = plt.plot(iters, err_Gs, label='gen')
plt.legend(handles=[h1, h2, h3])
plt.show()
