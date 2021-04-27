#Dimensionality reduction is the process of reducing the number of random variables under consideration,
#via obtaining a set of principal variables.
#It can be divided into feature extraction and feature selection.

#Feature Selection - In this method we select the best features from a set of features provided in the data set.
#We donot create new features , instead we choose the most important features from the given set of features and
#hence we remain with lesser dimensions or lesser features to work with.

#Feature Extraction -  In this method we create new features , and these features are not present in our original feature set .
#These features are not interpretable.

#PCA removes correlated features. After implementing the PCA on your dataset,
#all the Principal Components are independent of one another.
#There is no correlation among them.

import numpy as np
import pandas as pd

# Getting optimal number of features based on given percentage


def DR_PCA_WithPercentage(dataframe: pd.Series, variance_percentage: int) -> pd.DataFrame:
    """
    Returns the pandas dataframe with reduced features (feature extraction) based on the variance percentage 
    given as input.

    This approach is to perform Dimensionality Reduction using PCA (Principal Component Analysis) method.
    Without giving the number of components as input, variance percentage should be given to this method. 
    Based on the variance percentage, it will identify number of components needed for PCA method to perform 
    Dimensionality Reduction.

    For example, if 70 is the value given for variance percentage, it will calculate how many components from the 
    given dataframe needed to achieve that variance. It can be 3 or 4 or more components based on the feature values.

    :param dataframe : The pandas series used for performing dimensionality reduction.
    :param variance_percentage : It specifies how much variance is needed to perform dimensionality reduction.

    return: Pandas dataframe with the principal components as features selected using the PCA. Columns of dataframe
    will be numbered from 1,2,3..(based on number of features identified using variance percentage).

  """


    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Get only the numeric columns ignoring the categorical columns
    num_cols = dataframe._get_numeric_data().columns
    X_train = dataframe.loc[:, num_cols].values

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    pca = PCA()
    X_pca = pca.fit_transform(X_train_scaled)
    total_explained_variance = pca.explained_variance_ratio_.cumsum()
    threshold = variance_percentage / 100

    # Calculate number of components needed based on variance percentage
    n_over = len(total_explained_variance[total_explained_variance >= threshold])

    # Perform Dimensionality Reduction with the number of components identified
    pca = PCA(n_components=n_over)
    principalComponents = pca.fit_transform(X_train)
    column_list = [x + 1 for x in range(n_over)]
    # column_list = ['principal component 1', 'principal component 2']

    # Construct dataframe with principal components and its values
    principalDf = pd.DataFrame(data=principalComponents, columns=column_list)
    return principalDf


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


#Dimensionality Reduction using Linear method (Truncated SVD)

def DR_SVD(dataframe: pd.Series,no_of_components: int)-> pd.DataFrame:
    """
    Returns the pandas dataframe with reduced features (feature extraction) based on the number of components value
    given as input.
    
    This method is to perform Dimensionality Reduction using Truncated SVD (Singular Value Decomposition) method. 
    It will take the number of components as input for performing Dimensionality Reduction. This tells that the 
    result datframe should be reduced to the number of features specified here in the second parameter.
    
    For example, if we give dataframe with 15 features as input and 3 as number of components, 
    it will reduce the features to 3 and return a dataframe with only 3 features.
    
    :param dataframe : The pandas series used for performing dimensionality reduction.
    :param no_of_components : It specifies how many features are needed in the result dataframe.
    
    return: Pandas dataframe with the principal components as features selected using the Truncated SVD based 
    on the number of components given in the input. Columns of dataframe will be numbered from 1,2,3..(based 
    on number of features identified using variance percentage).
    
"""

    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    
    #Get the Truncated SVD object
    svd = TruncatedSVD(n_components=no_of_components, random_state=42)
    
    #Get only the numeric columns ignoring the categorical columns
    num_cols = dataframe._get_numeric_data().columns
    x = dataframe.loc[:, num_cols].values
    
    #Standardization
    x = StandardScaler().fit_transform(x)
    
    truncated_svd=svd.fit_transform(x)
    column_list = [x+1 for x in range(no_of_components)]
    
    #Construct dataframe with principal components and its values
    principalDf = pd.DataFrame(data = truncated_svd, columns = column_list )
    return principalDf

