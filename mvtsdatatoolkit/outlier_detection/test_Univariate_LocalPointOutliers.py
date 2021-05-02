from unittest import result
import pandas as pd
import numpy as np
import unittest

#from pandas._libs.missing import NA

# importing required functions from the method
from Univariate_LocalPointOutliers import findDerivatives_Univariate, Outliers_StdDev 

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
    
    def test_stationarytrend_diff(self):
        """
        This test checks the result from findDerivatives_Univariate with NaN, non-NaN data.
        In this test, NaN values are dropped and pandas series is converted to list and then output is validated. 
        Validation list is the result of function used in R_script present in the root folder. 
        Example:
        >>> lst = [1, 2, NaN, 4]
        Correct result after applicaiton of function on `lst`:
        >>> result = [NaN, 1, NaN, 2]
        Wrong result/s:
        >>> wrong_result1 = [NaN, 1, -2, 4]
        >>> wrong_result2 = [NaN, 1, -2, NaN]
        """

        df = pd.read_csv("data_outlierdetection/stationarytrend.csv")
        df_ans = pd.read_csv("data_outlierdetection/stationarytrend_diff_ans.csv")

        for i, j in enumerate(df.columns.to_list()[1:]):
            check_lst = findDerivatives_Univariate(df[['Date', j]], 1).dropna().iloc[:, 0].to_list() 
            ans_lst = df_ans[['Date', df_ans.columns[i+1]]].dropna().iloc[:, 1].to_list()
            self.assertEqual(check_lst, ans_lst)

    def test_temperature_dataset(self, accuracy = 7):
        """
        This test checks the result from findDerivatives_Univariate considering n_digits decimal accuracy.
        Some data does not require high accuracy in difference whereas some data does.
        This test enables users to define decimal accuracy. 
        After rounding, values are asserted using AlmostEqual unnittest method.
        :param accuracy: this is an integer value used for decimal accuracy. Default value is 7.

        ***Example:***
        >>> lst = [1.00004, 2.0005, 3.006]
        After findDerivatives_Univariate function is applied, let's assume accuracy upto 3rd decimal digit is required.
        
        Desired result should be:
        >>> result_wo_rounding = [NaN, 1.00046, 1.00550] #result is not rounded
        >>> result_with_wounding = [NaN, 1.000, 1.005]   #result is rounded
        
        But users may want some different kind of rounding such as 1.00550 -> 1.006
        Hence in that case, their result could be
        >>> result_different_rounding = [NaN, 1.001, 1.006] #rounded but different methodlogy
        Hence to tackle this problem `assertAlmostEqual` is used.
        
        `assertAlmostEqual` internally rounds of the abosolute difference between two numbers to 7 decimal digits.
        That is the sole reason, user cannot input decimal accuracy value greater than 7.
        """
        if accuracy > 7:
            print('Accuracy cannot be greater than 7.')
            accuracy = 7
        df = pd.read_csv('data_outlierdetection/Univariate_TempDataset.csv')
        df_ans = pd.read_csv("data_outlierdetection/Univariate_TempDataset_diff_ans.csv")

        for i, j in enumerate(df.columns.to_list()[1:]):
            check_lst = findDerivatives_Univariate(df[['Date', j]], 1).dropna().iloc[:, 0].to_list()
            check_lst = [round(val, accuracy) for val in check_lst]
            ans_lst = df_ans[['Date', df_ans.columns[i+1]]].dropna().iloc[:, 1].to_list()
            ans_lst = [round(val, accuracy) for val in ans_lst]
            self.assertAlmostEqual(check_lst, ans_lst)
    
    def test_default_dataset(self):
        """
        Checking findDerivatives_Univariate on the default dataset.
        This test also uses rounding but it is by default set to 3 decimal values. 
        """
        df = pd.read_csv('data_outlierdetection/datatest.csv')
        df_ans = pd.read_csv("data_outlierdetection/datatest_diff_ans.csv")

        for i, j in enumerate(df.columns.to_list()[1:]):
            check_lst = findDerivatives_Univariate(df[['date', j]], 1).dropna().iloc[:, 0].to_list()
            check_lst = [round(val, 3) for val in check_lst]
            ans_lst = df_ans[['date', df_ans.columns[i+1]]].dropna().iloc[:, 1].to_list()
            ans_lst = [round(val, 3) for val in ans_lst]
            self.assertAlmostEqual(check_lst, ans_lst)
            
    def test_stationarytrend(self):
        """ 
        This test ensures that if no outliers are found, empty list is the output.
        Results have been verfied using R-Script located in test_validation_R folder.
        """
        df = pd.read_csv('data_outlierdetection/stationarytrend.csv')['Temp']

        self.assertEqual(self.sorted_output(Outliers_StdDev, df, 3), [])

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


if __name__ == '__main__':
    unittest.main()