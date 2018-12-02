import sys
import numpy as np
import scipy.stats

a = np.ones(shape=(int(sys.argv[1]), int(sys.argv[2])))
max_entropy = scipy.stats.entropy(a.flatten())
print(max_entropy)