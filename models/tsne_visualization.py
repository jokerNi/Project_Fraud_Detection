
# Ref: https://www.kaggle.com/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
import time

# random state
random_state=100
np.random.seed=random_state
np.random.set_state=random_state

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD

# visualization
import matplotlib.patches as mpatches

df = pd.read_csv('../data/raw/creditcard.csv.zip',compression='zip')


X = df.drop(['Time','Class'],axis=1)
y = df['Class']


# T-SNE Implementation
print('Running TSNE ...')
t0 = time.time()
X_reduced_tsne = TSNE(n_components=2,verbose=2,
                      random_state=random_state).fit_transform(X.values)
t1 = time.time() - t0
print('TSNE Time taken: {:.0f} min {:.0f} secs'.format(*divmod(t1,60)))

# PCA Implementation
t0 = time.time()
X_reduced_pca = PCA(n_components=2,verbose=2,
                    random_state=random_state).fit_transform(X.values)

t1 = time.time() - t0
print('PCA Time taken: {:.0f} min {:.0f} secs'.format(*divmod(t1,60)))

# TruncatedSVD
t0 = time.time()
X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized',
                             random_state=random_state,verbose=2,
                            ).fit_transform(X.values)
t1 = time.time() - t0
print('SVD Time taken: {:.0f} min {:.0f} secs'.format(*divmod(t1,60)))

#========================== Plotting=============================


f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
# labels = ['No Fraud', 'Fraud']
f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)


blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
red_patch = mpatches.Patch(color='#AF0000', label='Fraud')


# t-SNE scatter plot
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0),
            cmap='coolwarm', label='No Fraud', linewidths=2)
ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1),
            cmap='coolwarm', label='Fraud', linewidths=2)
ax1.set_title('t-SNE', fontsize=14)

ax1.grid(True)

ax1.legend(handles=[blue_patch, red_patch])


# PCA scatter plot
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 0),
            cmap='coolwarm', label='No Fraud', linewidths=2)
ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 1),
            cmap='coolwarm', label='Fraud', linewidths=2)
ax2.set_title('PCA', fontsize=14)

ax2.grid(True)

ax2.legend(handles=[blue_patch, red_patch])

# TruncatedSVD scatter plot
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 0),
            cmap='coolwarm', label='No Fraud', linewidths=2)
ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 1),
            cmap='coolwarm', label='Fraud', linewidths=2)
ax3.set_title('Truncated SVD', fontsize=14)

ax3.grid(True)

ax3.legend(handles=[blue_patch, red_patch])

plt.savefig('../reports/figures/tsne_visualization.png')

plt.close()
