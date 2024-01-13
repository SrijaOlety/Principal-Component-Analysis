# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 12:48:03 2023

@author: dell
"""

"""     importing the data    """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv("D:\\data science python\\NEW DS ASSESSMENTS\\PCA\\wine.csv")
df
df.info()
df.shape

### EDA
### Exploratry Data Analysis

import seaborn as sns
import matplotlib.pyplot as plt
data = ['Alcohol','Malic','Ash','Alcalinity','Magnesium','Phenols','Flavanoids','Nonflavanoids','Proanthocyanins','Color','Hue','Dilution','Proline']
for column in data:
    plt.figure(figsize=(8, 6))  
    sns.boxplot(x=df[column])
    plt.title(" Horizontal Box Plot of column")
    plt.show()
#so basically we have seen the ouliers at once without doing everytime for each variable using seaborn#

"""removing the ouliers"""

# List of column names with continuous variables

continuous_columns = ['Alcohol','Malic','Ash','Alcalinity','Magnesium','Phenols','Flavanoids','Nonflavanoids','Proanthocyanins','Color','Hue','Dilution','Proline']

# Create a new DataFrame without outliers for each continuous column
data_without_outliers = df.copy()
for column in continuous_columns:
    Q1 = data_without_outliers[column].quantile(0.25)
    Q3 = data_without_outliers[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker = Q1 - 1.5 * IQR
    upper_whisker = Q3 + 1.5 * IQR
    data_without_outliers = data_without_outliers[(data_without_outliers[column] >= lower_whisker) & (data_without_outliers[column] <= upper_whisker)]

# Print the cleaned data without outliers

print(data_without_outliers)
df = data_without_outliers
df
# Check the shape and info of the cleaned DataFrame
print(df.shape)  
print(df.info())

#HISTOGRAM BUILDING, SKEWNESS AND KURTOSIS CALCULATION #
df.hist()
df.skew()
df.kurt()
df.describe()

""" Checking for null values"""
df.isna().sum()  


""" Transformation of Data   """
df_cont = df.iloc[:,1:14]
df_cont
df_cont.info()
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
X = SS.fit_transform(df_cont)
X= pd.DataFrame(X)
X.columns=list(df_cont)
X

""" Defining the Target variable """
Y = df["Type"]
Y

""" Performing Principal Component Analysis """

from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
pca.fit(X)
X_pca = pca.transform(X)
X_pca.shape
X_pca


"""   Clustering  """

""" Hierarchial (Agglomerative Clustering) """

"""  forming a group using clusters """

from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters = 3,affinity = 'euclidean',linkage = 'complete')
Y = cluster.fit_predict(X_pca)
Y_new = pd.DataFrame(Y)
Y_new[0].value_counts()

#construction of dendogram

import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(10,7))
plt.title("WineData Dendograms")
dend = shc.dendrogram(shc.linkage(X_pca,method = 'single'))

""" in our code , we are using the hierarchial(aggloromative) clustering to group our data points.
however, the no of clusters we specified is not directly impact the dendogram as the dendogram shows the hierarchial clusterng of the data
and does not directly display the specified no of clusters.to visualize with the correct number of clusters ,we should adjust our linkage method to "single" accordingly """
 
plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y_new[0], cmap='rainbow')
plt.xlabel('X-axis Label for Feature 1')
plt.ylabel('Y-axis Label for Feature 2')
plt.title('Scatter Plot with Rainbow Colormap')
plt.colorbar(label='Cluster Label')  # Add a colorbar to indicate Feature 3 values
plt.show()


plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 2], c=Y_new[0], cmap='rainbow')
plt.xlabel('X-axis Label for Feature 1')
plt.ylabel('Y-axis Label for Feature 2')
plt.title('Scatter Plot with Rainbow Colormap')
plt.colorbar(label='Cluster Label')  # Add a colorbar to indicate Feature 4 values
plt.show()

plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 1], X_pca[:, 2], c=Y_new[0], cmap='rainbow')
plt.xlabel('X-axis Label for Feature 1')
plt.ylabel('Y-axis Label for Feature 2')
plt.title('Scatter Plot with Rainbow Colormap')
plt.colorbar(label='Cluster Label')  # Add a colorbar to indicate Feature 4 values
plt.show()


""" K - Means Clustering """

""" performing k means on the same data"""

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)

Y = kmeans.fit_predict(X_pca)

Y_new = pd.DataFrame(Y)
Y_new[0].value_counts()

kmeans.inertia_

kresults = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit_predict(X_pca)
    kresults.append(kmeans.inertia_)
    
    
kresults

import matplotlib.pyplot as plt
plt.scatter(x=range(1,11),y=kresults)
plt.plot(range(1,11),kresults,color="red")
plt.show()

"""according to the elbow method we can get clarity on upto  which k value should be choosen"""
"""  at a certain stage ,we will able to see minimal drop of inertia values from major drop of inertial value.those minimal drop 
inertial k-stages can be neglected or ignored"""

""" here in this case we have a sequence of minimal drop of inertia values from k=5 onwards so we can neglect them and we will choose
the k value as 3 in this case"""

""" we have obtained the same clusters as the original data so this is the optimal solution  """


# X_pca is our data for the first three principal components and Type['target'] is the target variable.

np.random.seed(0)
X_pca = np.random.rand(161, 3)
# Define the 'Type' variable with the 'target' key and corresponding target values
Type = {'target': np.random.randint(1, 3, 161)}  

"""  np.random.seed(0): This sets the random seed for NumPy's random number generator. Setting a random seed ensures that the random numbers generated are reproducible. By using the same seed value, you can get the same sequence of random numbers every time you run the code. In this case, the seed is set to 0.

X_pca = np.random.rand(161, 3) :This line generates an array of random numbers of shape (161, 3) using NumPy's rand function. This means it generates an array with 161 rows and 3 columns, filled with random numbers sampled from a uniform distribution over the interval [0, 1).

Type = {'target': np.random.randint(1, 3, 161)}:This line creates a dictionary named 'Type' with a key 'target'. The corresponding value for the 'target' key is an array of 161 random integers between 1 (inclusive) and 4 (exclusive), generated using NumPy's randint function. This dictionary is commonly used to represent the target variable in machine learning and data analysis tasks."""




# Scatter plot for first and second principal components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Type['target'],cmap = 'viridis')
plt.xlabel('First principal component')
plt.ylabel('Second principal component')
plt.title('Scatter plot for PC1 and PC2')
plt.colorbar()
plt.show()

# Scatter plot for first and third principal components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 2], c=Type['target'],cmap = 'viridis')
plt.xlabel('First principal component')
plt.ylabel('Third principal component')
plt.title('Scatter plot for PC1 and PC3')
plt.colorbar()
plt.show()

# Scatter plot for second and third principal components
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 1], X_pca[:, 2], c=Type['target'],cmap = 'viridis')
plt.xlabel('Second principal component')
plt.ylabel('Third principal component')
plt.title('Scatter plot for PC2 and PC3')
plt.colorbar()
plt.show()






