import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib.pyplot as plt
import matplotlib.style as pltstyle
pltstyle.use('ggplot')
from matplotlib import rc
from matplotlib import rcParams
rc('font', **{'family': 'serif', 'serif': ['Latin Modern Roman']})
rc('text', usetex=True)
rc('font', size=8)
rc('font', weight='bold')
rc('text.latex', preamble=r'\usepackage{amsmath}')
testx = [1,2]
testy = [1,2]
fig, ax1 = plt.subplots()
ax1.set_xlabel('This Font is too large')
ax1.set_ylabel('And not bold')
ax1.plot(testx, testy, color='tab:red', marker='o', label='Right Font Style')
plt.legend()
plt.savefig('test.png', bbox_inches='tight')