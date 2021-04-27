import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import kurtosis
import matplotlib.pyplot as plt
from scipy.stats import skew


def Outliers_StdDev(data: pd.Series, distance_threshold: int) -> list:
    """
    Returns the outliers in a pandas series with the specified distance threshold value
    by using standard deviation method.

    Tbis approach is to remove the outlier points by eliminating any points that were above (Mean + 3*SD)
    and any points below (Mean - 3*SD). The recommended value for distance threshold will be 3 and
    this is parameterized for user convenience. SD is Standard deviation.

    Example: If x is a point in the dataframe column value,
    x will be considered as outlier if x < (mean - 3*SD)  or x > (mean + 3*SD).

    :param data : The pandas series used for detecting outliers.
    :param distance_threshold : Specifies how many times the standard deviation value is away from the mean.

    return : The list of outlier values in the given series.

    """
    outliers = []
    # Find mean, standard deviation for the given pandas series
    data_mean, data_std = np.mean(data), np.std(data)
    anomaly_cutoff = data_std * distance_threshold

    # Get the lower limit and upper limit values
    lower, upper = data_mean - anomaly_cutoff, data_mean + anomaly_cutoff

    # Get the values (outliers) which are greater than upper limit and lesser than lower limit
    outliers = [x for x in data if x < lower or x > upper]
    return outliers


def Calculate_Statistics(dataframe: pd.Series, step_size: int, length: int) -> pd.DataFrame:
    """
    Returns the pandas dataframe with different statistcal values as features like Mean, Standard Deviation,
    Kurtosis, Skewness, Subsequence start.

    This method will take the pandas dataframe as input along with step size and length. Based on the step size
    and length, the given dataframe will be divided into sub arrays like windows sliding. For every subarray, it
    will calculate Mean, Standard deviation, Kurtosis, Skewness values. At the end, a new dataframe will be created
    with all these calculated statistical values.

    For example, consider a dataframe with 12 values, step size as 2 and length as 3 are given as input.
    The sub arrays will be slided like {1 to 3, 3 to 6, 5 to 8, 7 to 10, 9 to 12}. For these subsequences,
    statistcal methods will be implemented to get the required values.

    :param dataframe : The pandas dataframe in which the statistical methods to be applied.
    :step_size : It specifies how much values we need to slide through the dataframe.
    :length : Pre defined value for choosing subsequence length.

    return : A pandas dataframe with calculated mean, standard deviation, kurtosis and skewness values for the
    subsequences based on the predefined length and step size.

    """
    # Initialize Values
    start = 1
    end = length

    # Initialize dictionary
    mean = {}
    std_dev = {}
    kurto = {}
    skewness = {}

    # Get the copy of data
    arr = dataframe.copy()
    subarray = []

    # Loop through the data to find mean,standard deviation,kurtosis and skewness
    while end != len(arr):
        j = 0

        # Get the subsequences as sub arrays
        for i in range(start, end):
            subarray.append(arr[i])
            j = j + 1

        # Calculate
        val = np.mean(subarray)
        st_val = np.std(subarray)
        kurt_val = kurtosis(subarray, fisher=True, bias=True)
        skewness_val = skew(subarray)

        # Assign values to corresponding dictionary value with subsequence start number as key
        mean[start] = val
        std_dev[start] = st_val
        kurto[start] = kurt_val[0]
        skewness[start] = skewness_val[0]

        # Slide the window based on the step size
        start = start + step_size
        end = end + step_size

    # Construct Dataframe with mean,standard deviation and kurtosis values

    newdf = pd.DataFrame()
    newdf['Mean'] = mean.values()
    newdf['StdDev'] = std_dev.values()
    newdf['Kurtosis'] = kurto.values()
    newdf['Skewness'] = skewness.values()
    newdf['Subsequence_start'] = mean.keys()

    return newdf


def plotData(df: pd.DataFrame):
    """
    Plotting given dataframe in a two dimensional space with the anomalies pointed in red

    :param df: The univariate timeseries dataframe with anomalies flagged as 1 under Anomaly column

    The result will be scatter plot with Datetime against Values and the anomalies in red points.

    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get the positions where anomalies marked as 1
    a = df.loc[df['Anomaly'] == 1]  # anomaly

    # Plotting Dervative values in blue
    ax.plot(df[df.columns[0]], df[df.columns[1]], color='blue', label='Normal')  # check

    # Plotting Anomaly points in red
    ax.scatter(a[a.columns[0]], a[a.columns[1]], color='red', label='Anomaly')
    plt.legend()
    plt.show();


def Univariate_SubsequenceOutliers(dataframe: pd.Series, step_size: int, length: int, statistics_option: str):
    """
    This is the main function to identify subsequence outliers in univariate time series data.

    This method get the numeric columns in a given dataframe, calculates statistical data like mean,standard deviation,
    kurtosis, skewness for the standardized values. Based on the user input for statistical method, outliers will be
    calculated.

    For example, if the user provided Kurtosis as the input, then outliers will be calculated based on kurtosis values
    of the sub arrays in the given dataframe.

    Anomalies will be detected based on the above method and set the flag to 1 for Anomalies. (Note: The subarrays
    are considered as individual points here). Again the subsequences corresponding to the points are derived and
    marked in the graph to show the subsequence anomalies.

    :param dataframe : The pandas dataframe in which the subsequence outliers need to be identified.
    :step_size : It specifies how much values we need to slide through the dataframe.
    :length : Pre defined value for choosing subsequence length.
    :statistics_option : Statistical values like Mean, Standard Deviation, Kurtosis, Skewness.

    Result will be plotting subsequence outliers in the two dimensional space against values with DateTime.

    """
    from sklearn.preprocessing import StandardScaler
    # Get only the numeric columns ignoring the categorical columns
    num_cols = dataframe._get_numeric_data().columns
    X_train = dataframe.loc[:, num_cols].values

    # Standardization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    df_Statistics = Calculate_Statistics(X_train_scaled, step_size, length)

    outliers = Outliers_StdDev(df_Statistics[statistics_option], 2)

    df_Statistics['Anomaly'] = 0
    df_Statistics.loc[df_Statistics[df_Statistics.columns[0]].isin(outliers), "Anomaly"] = 1

    # Show this in time series data
    subsequences = df_Statistics[df_Statistics['Anomaly'] == 1]
    subseq_list = []
    subseq_list = subsequences['Subsequence_start']
    dataframe['Anomaly'] = 0

    for i in range(len(subseq_list)):
        start = subseq_list[i]
        end = start + step_size
        for j in range(start, end):
            dataframe.at[j, 'Anomaly'] = 1

    plotData(dataframe)