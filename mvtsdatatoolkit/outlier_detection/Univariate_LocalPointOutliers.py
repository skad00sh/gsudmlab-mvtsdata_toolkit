import pandas as pd
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

def findDerivatives_Univariate(uvts: pd.DataFrame , neighbor_threshold: int)-> pd.DataFrame:
    
    """
    Identifies the derivatives in the Univariate time series data with the specified neighbor threshold value.
    
    For example, if neighbor_threshold = 1, it will find the difference value between n2,n1 
    in the univariate time series data.
    ie., The first difference is given by the value out[i] = n[i+1] - n[i] 
    
    :param uvts : The UniVariateTimeSeries dataframe for identifying derivatives in it.
    :param neighbor_threshold : The number of times values are differenced. 
    If zero, the input is returned as-is.
    
    return : The same dataframe with date time column unchanged and 
    also with another column for corresponding derivative values.
    
    Assumption : In Time Series data - First column as Date, Second column as Value
    
    """
    
    #Save values in a series
    univariatedata=uvts[uvts.columns[1]]
    
    #Use numpy diff to find derivatives
    derivatives=univariatedata.diff(neighbor_threshold)
    
    #Create a new dataframe with derivatives and corresponding datetime column values
    derivativeTS=pd.DataFrame(derivatives)
    derivativeTS['DateTime']=uvts[uvts.columns[0]] 
    return derivativeTS

def Outliers_StdDev(data: pd.Series,distance_threshold: int)-> list:
    
    """
    Returns the outliers in a pandas series with the specified distance threshold value 
    by using standard deviation method.
    
    Tbis approach is to remove the outlier points by eliminating any points that were above (Mean + 3*SD) 
    and any points below (Mean - 3*SD). The recommended value for distance threshold will be 3 and this is parameterized
    for user convenience. SD is Standard deviation.
    
    Example: If x is a point in the dataframe column value, 
    x will be considered as outlier if x < (mean - 3*SD)  or x > (mean + 3*SD).
    
    :param data : The pandas series used for detecting outliers.
    :param distance_threshold : Specifies how many times the standard deviation value is away from the mean. 
    
    return : The list of outlier values in the given series.
    
    """
    outliers=[]
    #Find mean, standard deviation for the given pandas series
    data_mean, data_std = np.mean(data), np.std(data)
    anomaly_cutoff = data_std * distance_threshold 
    
    #Get the lower limit and upper limit values
    lower, upper = data_mean - anomaly_cutoff, data_mean + anomaly_cutoff
    
    #Get the values (outliers) which are greater than upper limit and lesser than lower limit
    outliers = [x for x in data if x < lower or x > upper]
    return outliers

def findPointAnomalies_Univariate(df: pd.DataFrame)-> pd.DataFrame:
    
    """
    This method will find point anomalies in the univariate time series dataframe. 
    The anomalies will be given flag as 1 and flags are stored under new column 'Anomaly'
    
    :param df : Pandas dataframe with derivatives of univariate time series values.
    
    return : The same dataframe with additional column 'Anomaly' for marking anomalies 
    corresponding to the datetime value.
    
    """
    
    # Call another function to get the list of outliers
    outliers=Outliers_StdDev(df[df.columns[0]],3)
    df['Anomaly']=0
    
    # Mark flag as '1' for the values in outliers list
    df.loc[df[df.columns[0]].isin(outliers), "Anomaly"] = 1
    df[df['Anomaly']==1]
    return df

#Find Local Outliers in Univariate Time Series - Main Entry function to call

def findLocalOutliers_Univariate(uvts: pd.DataFrame , neighbor_threshold=1):
    
    """
    The main method for identifying Local outlier points in univariate time series data.
    
    This will call 'findDerivatives_Univariate' method for identifying derivatives between neighbors.
    And then 'findPointAnomalies_Univariate' method for getting point anomalies list in the dataframe.
    
    :param uvts : The Univariate time series dataframe in which point outliers are going to be detected.
    :param neighbor_threshold : Number of neighbors to consider for identifying derivatives between them.
    
    Result will be plotting point outliers in the two dimensional space against values with DateTime.
    
    """
    newdf=findDerivatives_Univariate(uvts,neighbor_threshold)
    #newdf.rename({newdf.columns[0]: 'Values'}, axis=1, inplace=True) #REmove
    dfWithAnomalies=findPointAnomalies_Univariate(newdf)
    plotData(dfWithAnomalies)

def plotData(df: pd.DataFrame):
    
    """
    Plotting given dataframe in a two dimensional space with the anomalies pointed in red
    
    :param df: The univariate timeseries dataframe with anomalies flagged as 1 under Anomaly column
    
    The result will be scatter plot with Datetime against Values and the anomalies in red points.
    
    """
    fig, ax = plt.subplots(figsize=(10,6))
    
    #Get the positions where anomalies marked as 1
    a = df.loc[df['Anomaly'] == 1] #anomaly
    
    #Plotting Dervative values in blue 
    ax.plot(df['DateTime'], df[df.columns[0]], color='blue', label='Normal')
    
    #Plotting Anomaly points in red
    ax.scatter(a['DateTime'],a[a.columns[0]], color='red', label='Anomaly')
    plt.xlabel('DateTime')
    plt.ylabel('Values')
    plt.legend()
    plt.show();