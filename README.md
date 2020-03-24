# 2D-Blind-kMeans-Clustering

Implementing kMeans and kMeas++ on 2D points to find accurate value of k and classify accordingly.

## Usage

```python
python3 k_means_strategy_random.py
python3 k_means_strategy_farthest.py
```
## Required Package
```python
import os
import sys
import copy
import time
import numpy as np
import pandas as pd
from scipy.io import loadmat
from datetime import datetime
import matplotlib.pyplot as plt
```

## Strategies

> k-Cluster (2 ≤ k ≤ 10)  

> Objective:  To minimize the below given function
<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{i=1}^{k}&space;\sum_{x\epsilon&space;D_{i}}^{n}&space;\left&space;\|&space;x&space;-&space;\mu_{i}&space;\right&space;\|^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{i=1}^{k}&space;\sum_{x\epsilon&space;D_{i}}^{n}&space;\left&space;\|&space;x&space;-&space;\mu_{i}&space;\right&space;\|^2" title="\sum_{i=1}^{k} \sum_{x\epsilon D_{i}}^{n} \left \| x - \mu_{i} \right \|^2" /></a>


 

### Strategy 1
Randomly picked initial centers from the given samples.

#### Example

[Cluster Images - Strategy 1](https://github.com/asgaonkar/2D-Blind-kMeans-Clustering/tree/master/Images/Random)

k | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 
------------ | ------------ | ------------- | ------------ | ------------- |------------ | ------------- |------------ | ------------- |------------ |
**Objective** | 2498.11 | 1338.11 | 788.96 | 592.43 | 476.12 | 367.60 | 305.45 | 255.83 | 214.99 |

![Objective Function vs k-Cluster](https://raw.githubusercontent.com/asgaonkar/2D-Blind-kMeans-Clustering/master/Images/Objective%20Function%20(vs)%20k-Clusters%20-%20Random%20Centroid_Github.png)



### Strategy 2
First centeras random and for the i<sup>th</sup> center (i>1), selected a center such that the average distance of this chosen one to all previous (i-1) centers is maximal.

#### Example

[Cluster Images - Strategy 2](https://github.com/asgaonkar/2D-Blind-kMeans-Clustering/tree/master/Images/Farthest)

k | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 
------------ | ------------ | ------------- | ------------ | ------------- |------------ | ------------- |------------ | ------------- |------------ |
**Objective** | 1921.033 | 1293.777 | 797.960 | 651.403 | 476.118 | 362.866 | 350.209 | 258.431 | 239.065 |


![Objective Function vs k-Cluster](https://raw.githubusercontent.com/asgaonkar/2D-Blind-kMeans-Clustering/master/Images/Objective%20Function%20(vs)%20k-Clusters%20-%20Farthest%20Centroid_Github.png)

## Tasks Completed
1.    Implement the k-means algorithm with Strategy 1.
2.    Compute the objective function as a function of k (k = 2, 3, …, 10).
3.    Implement the k-means algorithm with Strategy 2.
4.    Compute the objective function as a function of k (k = 2, 3, …, 10).

## Author

**Atit Gaonkar** - [Github](https://github.com/asgaonkar) - [LinkedIn](https://www.linkedin.com/in/atit-gaonkar/)
