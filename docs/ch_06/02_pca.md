# Anonymizing data

``` python
import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
pd.set_option("display.max_columns", 6)
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | mean radius | mean texture | mean perimeter | ... | worst concave points | worst symmetry | worst fractal dimension |
|----|----|----|----|----|----|----|----|
| 0 | 17.99 | 10.38 | 122.80 | ... | 0.2654 | 0.4601 | 0.11890 |
| 1 | 20.57 | 17.77 | 132.90 | ... | 0.1860 | 0.2750 | 0.08902 |
| 2 | 19.69 | 21.25 | 130.00 | ... | 0.2430 | 0.3613 | 0.08758 |
| 3 | 11.42 | 20.38 | 77.58 | ... | 0.2575 | 0.6638 | 0.17300 |
| 4 | 20.29 | 14.34 | 135.10 | ... | 0.1625 | 0.2364 | 0.07678 |

<p>5 rows × 30 columns</p>
</div>

``` python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

pca = PCA(n_components=30, random_state=0)
pca_data = pca.fit_transform(df)

scaler = StandardScaler()
anon_df = pd.DataFrame(scaler.fit_transform(pca_data))
pd.set_option("display.max_columns", 8)
anon_df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|  | 0 | 1 | 2 | 3 | ... | 26 | 27 | 28 | 29 |
|----|----|----|----|----|----|----|----|----|----|
| 0 | 1.743043 | -3.440692 | 1.832695 | 1.179529 | ... | -1.033900 | 0.767070 | 1.406020 | 0.841434 |
| 1 | 1.906779 | 0.182972 | -1.335313 | -2.418269 | ... | -0.043492 | -0.798802 | 0.484854 | -1.267746 |
| 2 | 1.496120 | 0.458381 | -0.064503 | -0.568556 | ... | 0.092680 | 0.010964 | -0.547972 | 0.484234 |
| 3 | -0.611764 | -0.788775 | 0.327197 | 1.592188 | ... | 0.008095 | 0.811865 | -1.511794 | -1.978890 |
| 4 | 1.397781 | 2.216483 | 0.051866 | -1.150718 | ... | 1.716566 | 0.161769 | 1.260500 | 0.390467 |

<p>5 rows × 30 columns</p>
</div>

``` python
import numpy as np

np.sum(pca.explained_variance_ratio_)
```

    np.float64(1.0)
