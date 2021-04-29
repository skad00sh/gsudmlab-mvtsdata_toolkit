## test_Univariate_GlobalPointOutliers.py

### Output

```
...F.
======================================================================
FAIL: test_stationarytrend_missing_values (__main__.TestUniGlobOutliers)
This test ensure that missing values handling for the outlier detetction.
----------------------------------------------------------------------
Traceback (most recent call last):
  File ".\test_Univariate_GlobalPointOutliers.py", line 53, in test_stationarytrend_missing_values
    self.assertEqual(self.sorted_output(Outliers_IQR, df), [34, 36])
AssertionError: Lists differ: [] != [34, 36]

Second list contains 2 additional elements.
First extra element 0:
34

- []
+ [34, 36]

----------------------------------------------------------------------
Ran 5 tests in 0.041s

FAILED (failures=1)
```

#### What could be the problem?

Probably `np.percentile` in IQR method is unable to handle `nan`. ~~(But I am still confirming)~~

My final verdict is `np.percentile` can not handle `nan` values. 

```
>>> data = pd.Series([1, 2, 3])
>>> print(np.percentile(data, 25))
1.5

>>> data = pd.Series([1, np.nan, 3])
>>> print(np.percentile(data, 25))
nan
```

So `Outliers_IQR` function should either remove `nan` or throw an error if it is present in the data. 

```
>>> data = pd.Series([1, np.nan, 3])
>>> data = data[~np.isnan(data)]
>>> print(data)
0   1
1   3
dtype: int32

>>> print(np.percentile(data, 25))
 1.5
```

Box plots clearly shows the outliers:

1. Python
 
![python-boxplot](https://github.com/skad00sh/gsudmlab-mvtsdata_toolkit/blob/724345afb5857c28c52e25fb61f87e494403f709/mvtsdatatoolkit/outlier_detection/conflicts/Python%20boxplot%20stationarytrend_with_missingVal.jpg?raw=true)

2. R

![R-boxplot](https://github.com/skad00sh/gsudmlab-mvtsdata_toolkit/blob/724345afb5857c28c52e25fb61f87e494403f709/mvtsdatatoolkit/outlier_detection/conflicts/R%20boxplot%20stationarytrend_with_missingVal.jpeg?raw=true)
