import pandas as pd
import numpy as np
import matplotlib as plt

# Dimensionality Reduction using Linear method (PCA)

def DR_PCA(dataframe: pd.Series, no_of_components: int) -> pd.DataFrame:
   

"""
    Returns the pandas dataframe with reduced features (feature extraction) based on the number of components value
    given as input.

    This method is to perform Dimensionality Reduction using PCA (Principal Component Analysis) method. 
    It will take the number of components as input for performing Dimensionality Reduction.
    This tells that the result datframe should be reduced to the number of features specified here in 
    the second parameter.

    For example, if we give dataframe with 15 features as input and 3 as number of components, 
    it will reduce the features to 3 and return a dataframe with only 3 features.

    :param dataframe : The pandas series used for performing dimensionality reduction.
    :param no_of_components : It specifies how many features are needed in the result dataframe.

    return: Pandas dataframe with the principal components as features selected using the PCA based on the number of 
    components given in the input. Columns of dataframe will be numbered from 1,2,3..(based on number of 
    features identified using variance percentage).

"""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Get only the numeric columns ignoring the categorical columns
    num_cols = dataframe._get_numeric_data().columns
    x = dataframe.loc[:, num_cols].values

    # Standardization
    x = StandardScaler().fit_transform(x)

    # Perform Dimensionality Reduction with the number of components given in the paramteter
    pca = PCA(n_components=no_of_components)
    principalComponents = pca.fit_transform(x)

    column_list = [x + 1 for x in range(no_of_components)]
    # column_list = ['principal component 1', 'principal component 2']

    # Construct dataframe with principal components and its values
    principalDf = pd.DataFrame(data=principalComponents, columns=column_list)
    return principalDf


#Main entry function to call

def find_optimalepsilon(data: pd.Series):
    

"""
   Plots the graph with different epsilon values. This function can be executed before choosing epsilon
   value for identifying Point Outliers in MVTS using DBSCAN.

   This method takes the dataframe, reduces the dimension and identify the nearest neighbors using Nearest Neighbors 
   algorithm. Then it will find the k nearest neighbors, calcukate the distances and sort the distances.

   :param data: The dataframe for which anomalies need to be detected.

   The point after the line becomes steady is considered as optimal epsilon value.

"""
    from sklearn.neighbors import NearestNeighbors
    from matplotlib import pyplot as plt

    # Dimensiomnality Reduction
    dataset = DR_PCA(data, 2)

    # Identifying Nearest Neighbors and distances
    neighbors = NearestNeighbors(n_neighbors=20)
    neighbors_fit = neighbors.fit(dataset)
    distances, indices = neighbors_fit.kneighbors(dataset)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]

    # Plot the graph
    plt.plot(distances)