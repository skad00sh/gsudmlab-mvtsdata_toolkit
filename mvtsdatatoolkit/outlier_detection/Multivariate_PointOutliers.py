import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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


def plotData(data: pd.Series, features_count: int):
    """
    Plotting given dataframe with the specified features count in a two dimensional space with the 
    anomalies pointed in red

    :param data: The multivariate timeseries dataframe with anomalies flagged as 1 under Anomaly column
    :param features_count: The number of features from the given dataframe to be plotted 

    The result will be scatter plot with Datetime against Values and the anomalies in red points.

    Assumption: Since the date time feature will be in the firsr column, the other feature number starts from 2.
    And also only 9 values given in colors array.

"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Setting different line colors for features
    colors = ["blue", "green", "orange", "yellow", "purple", "brown", "black", "cyan", "magenta"]
    j = 0

    # Get the rows from dataframe where Anomaly flag is set to 1
    a = data.loc[data['Anomaly'] == 1]

    # Start from 2 since the first column will be date time
    start = 2
    end = start + features_count

    # Plot the feature values
    for i in range(start, end):
        # Plot all the values of the corresponding feature
        ax.plot(data['date'], data[data.columns[i]], c=colors[j], label=i)

        # Plot the Anomaly point in red for all the features plotted in the previous line
        ax.scatter(a['date'], a[a.columns[i]], color='red')
        j = j + 1

    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.show()


#Main Entry function to call

def MVTS_PointOutliers_DBSCAN(data: pd.Series, minsample_value: int, epsilon_value: int):
    """
   Identifies all the point anomalies in the multivariate time series dataframe using DBSCAN method and 
   plots the dataframe with all the anomaly points marked in red.

   This method will take a multivariate time series data as input along with minimum sample value and epsilon value.
   Based on the values given for minsample_value and epsilon_value, it will form the clusters and identify 
   the anomalies. DBSCAN algorithm works with these two parameters - eps,minPoints. Dimensionality Reduction 
   (using PCA) will be done before feeding into the DBSCAN algorithm.

   The points which are not in any of the clusters will be marked -1. Based on that, the corresponding rows
   will have the anomaly flag as 1.

   :param data: The multivariate time series data in which anomalies are detected using DBSCAN algorithm.

   :param minsample_value: the minimum number of points to form a dense region. For example, if we set the 
    minPoints parameter as 5, then we need at least 5 points to form a dense region.

   :param epsilon_value: It specifies how close points should be to each other to be considered a part of 
    a cluster. It means that if the distance between two points is lower or equal to this value (eps), 
    these points are considered neighbors.

   It finally calls the plotData function to plot the given multivariate data with anomalies marked in red.

   Note: Please run OptimalEpsilon_DBSCAN file to choose epsilon value for DBSCAN Algorithm.

 """


    from sklearn.cluster import DBSCAN

    # Dimensionality Reduction using PCA
    resultdf = DR_PCA(data, 2)
    np.random.seed(6)

    # Use DBSCAN for outlier detection
    outlier_detection = DBSCAN(min_samples=minsample_value, eps=epsilon_value)
    clusters = outlier_detection.fit_predict(resultdf)
    # list(clusters).count(-1)

    # Identify the index positions where clusters value is 1
    x = np.where(clusters == -1)

    # By default, set 0 as anomaly flag for all rows in the dataframe
    data['Anomaly'] = 0

    # Assign row number to all the rows
    data['row_num'] = np.arange(len(data))

    # Set Anomaly flag to 1 based on the index positions
    for i in x:
        data['Anomaly'] = data.apply(lambda x: 1 if (x['row_num'] == i).any() else x['Anomaly'], axis=1)

    # Plot data with required number of features
    plotData(data, 2)