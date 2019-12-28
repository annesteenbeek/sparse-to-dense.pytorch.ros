
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize

from scipy import stats


sparse = np.load('dev/sparse_samples.npy')
target = np.load('dev/target_samples.npy')

max_depth = 3.0
 
valid_mask = target < max_depth
 
sparse = sparse[valid_mask]
target = target[valid_mask]

error = target-sparse

error_norm = error / np.abs(np.sum(error))

rel_error = error/target

ae, loce, scalee = stats.skewnorm.fit(error)
sample = stats.skewnorm(ae, loce, scalee).rvs(len(error))

ax = plt.subplot(211)
sns.distplot(error, norm_hist=True)
plt.subplot(212, sharex=ax)
sns.distplot(sample, norm_hist=True)

plt.show()