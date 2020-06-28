#!/usr/bin/env python
# coding: utf-8

# <table width=100%>
# <tr>
#     <td><h1 style="text-align: left; font-size:300%;">
#        Protein Expression in Mice with Down Syndrome
#     </h1></td>
#     <td width="20%">
#     <div style="text-align: right">
#     <b> Homework Machine Learning 2020</b>
#     <br> Emanuel Michele Soda <br>
#     <a href="emanuelmichele.soda@mail.polimi.it">emanuelmichele.soda@mail.polimi.it</a><br>
#     </div>
#     </td>
#     <td width="111px">
#         <a href="https://www.polimi.it">
#         <img align="right", width="100px" src='https://labolfattometrico.chem.polimi.it/wp-content/uploads/2019/12/POLIMI-corretto-3-1024x1024.jpg' alt=''>
#         </a>
#     </td>
# </tr>
# </table>

# # Importing Packages

# In[16]:


#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import sklearn as sk
from sklearn import preprocessing
from sklearn.decomposition import PCA


# In[17]:


train = pd.read_csv("Data/train.csv")
train.head()


# In[18]:


#get_ipython().run_line_magic('pinfo', 'train.corr')


# In[9]:


#plotto la correlation matrix
plt.figure(figsize=(12,10))
corrMat = train.corr()
print(corrMat.to_numpy()[:,-1])
sns.heatmap(corrMat);


# In[19]:


#scalo i dati per fare la PCA
scaled_Train = preprocessing.scale(train.drop(['class'], axis = 1))
pca = PCA()
pca.fit(scaled_Train)


# In[ ]:


#get_ipython().run_line_magic('pinfo', 'sk.preprocessing.scale')


# In[ ]:


#print(pca.explained_variance_ratio_)
Z = pca.transform(scaled_Train)
#calcolo la percentuale di variazione
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals = 1)
labels = ['PC' + str(s) for s in range(1, len(per_var) + 1)]

plt.figure(figsize=(44, 10))
plt.bar(x = range(1, len(per_var) + 1), height = per_var, tick_label = labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Plot 1')
plt.show()

#Z = pd.DataFrame(Z, columns = ['PCA1', 'PCA2'])

#Z.plot.scatter('PCA1', 'PCA1')
#sns.scatterplot(x = "PCA", y = "PCA2", hue = "day", data = Z);


# In[ ]:


pca_df = pd.DataFrame(Z, columns = labels)
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs = pca_df.PC1, ys = pca_df.PC2, zs = pca_df.PC3, marker= 'o', s = 150)


# In[ ]:


#get_ipython().run_line_magic('pinfo', 'ax.scatter')


# In[ ]:


test = pd.read_csv("Data/test.csv")
test.head()


# In[ ]:





# In[ ]:





# In[ ]:


sns.pairplot(train.head(), height=2.5)


# #####

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
