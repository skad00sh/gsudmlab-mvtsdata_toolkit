from unittest import result
import pandas as pd
import numpy as np
import unittest

#from pandas._libs.missing import NA

# importing required functions from the method
from Univariate_GlobalPointOutliers import Outliers_StdDev, Outliers_IQR 

class TestUniGlobOutliers(unittest.TestCase):

    def sorted_output(self, func, lst, diff = False):
        """
        This function gives sorted output.
        Default value for `diff` is False. 
        """
        if diff == False:
           res = func(lst)
        else:
            res = func(lst, diff)
        res.sort()
        return res

    def test_stationarytrend(self):
        """ 
        This test ensures that if no outliers are found, empty list is the output.
        Results have been verfied using R-Script located in test_validation_R folder.
        """
        df = pd.read_csv('data_outlierdetection/stationarytrend.csv')['Temp']

        self.assertEqual(self.sorted_output(Outliers_StdDev, df, 3), [])
        self.assertEqual(self.sorted_output(Outliers_IQR, df), [])

    def test_stationarytrend_missing_values(self):
        """ 
        This test ensure that missing values handling for the outlier detetction.

        Example (for Standard Deviation):
        data = [1, 2, 3, 4, 5] -> mean = 3
        Correct:
            data = [1, 2, 3, , 5] -> mean = 2.75 (Total count of numbers 4)
        Wrong:
            data = [1, 2, 3, , 5] -> mean = 2.2 (Here blank is also being counted hence count of numbers 5)

        Results have been verfied using R-Script located in test_validation_R folder.
        """
        df = pd.read_csv('data_outlierdetection/stationarytrend.csv')['Temp_missing_val']

        self.assertEqual(self.sorted_output(Outliers_StdDev, df, 3), [])
        self.assertEqual(self.sorted_output(Outliers_IQR, df), [34, 36])

    def test_lowerBound_outliers(self):
        """

        """
        df = pd.read_csv('data_outlierdetection/lowerBound_outliers.csv')['Sales']
        
        self.assertEqual(self.sorted_output(Outliers_StdDev, df, 3), [99, 108, 136, 237])
        
        res_iqr = [5, 5, 5, 67, 68, 68, 69, 69, 69, 70, 72, 73, 74, 75, 78, 83, 84, 99, 108, 136, 237]
        self.assertEqual(self.sorted_output(Outliers_IQR, df), res_iqr)




    def test_temperature_dataset(self):
        """ Testing outliers on existing temperature dataset """
        # loading required dataset with column
        df = pd.read_csv('data_outlierdetection/Univariate_TempDataset.csv')['Temp']

        """ This result has been verfied using R-Script located in test_validation_R folder """
        res_std = [23.4, 23.9, 24.0, 24.1, 24.3, 24.8, 25.0, 25.0, 25.2, 26.3]
        self.assertEqual(self.sorted_output(Outliers_StdDev, df, 3), res_std)

        """ This result has been verfied using R-Script located in test_validation_R folder """
        res_iqr = [22.7, 22.8, 23.0, 23.4, 23.9, 24.0, 24.1, 24.3, 24.8, 25.0, 25.0, 25.2, 26.3]
        self.assertEqual(self.sorted_output(Outliers_IQR, df), res_iqr)

if __name__ == '__main__':
    unittest.main()