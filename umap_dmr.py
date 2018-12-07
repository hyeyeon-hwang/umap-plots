import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap
from mpl_toolkits.mplot3d import Axes3D

sns.set( style='white', context='notebook', rc={'figure.figsize':(14,10)} )

# dDmr
'''
dmr = pd.read_csv("umap_dDmr.csv")

dmrReducer = umap.UMAP()
dmrEmbedding = dmrReducer.fit_transform(dmr)
print( dmr.head() )
print(dmrEmbedding)

plt.scatter(dmrEmbedding[:,0], dmrEmbedding[:, 1])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the dDmr dataset', fontsize=16);
plt.savefig('umap_dDmr.png')
plt.show()
'''

# rDmr
'''
rdmr = pd.read_csv("umap_rDmr.csv")

rdmrReducer = umap.UMAP()
rdmrEmbedding = rdmrReducer.fit_transform(rdmr)
print( rdmr.head() )
print(rdmrEmbedding)

plt.scatter(rdmrEmbedding[:,0], rdmrEmbedding[:, 1])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the rDmr dataset', fontsize=16);
plt.savefig('umap_rDmr.png')
plt.show()
'''

# aDmr
'''
admr = pd.read_csv("umap_aDmr.csv")

admrReducer = umap.UMAP()
admrEmbedding = admrReducer.fit_transform(admr)
print( admr.head() )
print(admrEmbedding)

plt.scatter(admrEmbedding[:,0], admrEmbedding[:, 1])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the aDmr dataset', fontsize=16);
plt.savefig('umap_aDmr.png')
plt.show()
'''

pdmr = pd.read_csv("umap_pDmr.csv")
# n_components = specifying number of dimensions
'''
pdmrReducer = umap.UMAP(n_components = 3)
pdmrEmbedding = pdmrReducer.fit_transform(pdmr)
print( pdmr.head() )
print(pdmrEmbedding)

# for 2D
# plt.scatter(pdmrEmbedding[:,0], pdmrEmbedding[:, 1])
# for 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pdmrEmbedding[:,0], pdmrEmbedding[:, 1], pdmrEmbedding[:, 2])
# plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the pDmr dataset', fontsize=16);
plt.savefig('3d umap_pDmr.png')
plt.show()
''' 

# Dimension Reduction 
# n_components = 50
pdmrReducer50 = umap.UMAP(n_components = 20)
pdmrEmbedding50 = pdmrReducer50.fit_transform(pdmr)
print( pdmr.head() )
print(pdmrEmbedding50)

