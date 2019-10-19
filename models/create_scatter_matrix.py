
import pandas as pd
from pandas.plotting import scatter_matrix
import time

t0 = time.time()

df = pd.read_csv('../data/raw/creditcard.csv.zip',compression='zip')
df = df.sample(n=1000)

myfeatures = ['V{}'.format(i) for i in range(1,29)] + ['Amount']

# this takes long time to run, comment it.
scatter_matrix(df[myfeatures], diagonal='kde')
plt.savefig('../reports/figures/scatter_matrix_of_all_features.png',dpi=400)

t1 = time.time() - t0
print('Time taken: {:.0f} min {:.0f} secs'.format(*divmod(t1,60)))