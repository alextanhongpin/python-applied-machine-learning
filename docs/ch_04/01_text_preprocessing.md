# Text classification

``` python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

lines = [
    "Four score and 7 years ago our fathers brought forth,",
    "... a new NATION, conceived in liberty $$$,",
    "and dedicated to the PrOpOsItIoN that all men are created equal",
    "One nation's freedom equals #freedom for another $nation!",
]

# Vectorize the lines
vectorizer = CountVectorizer(stop_words="english")
word_matrix = vectorizer.fit_transform(lines)

# Show the resulting word matrix
feature_names = vectorizer.get_feature_names_out()
line_names = [f"Line {(i+1):d}" for i, _ in enumerate(word_matrix)]

df = pd.DataFrame(data=word_matrix.toarray(), index=line_names, columns=feature_names)
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

|  | ago | brought | conceived | created | dedicated | equal | equals | fathers | forth | freedom | liberty | men | nation | new | proposition | score | years |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| Line 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| Line 2 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| Line 3 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 |
| Line 4 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |

</div>

``` python
import re

lines = [
    "Four score and 777 years ago our fathers brought forth,",
    "... a new NATION, conceived in liberty $$$,",
    "and dedicated to the PrOpOsItIoN that all men are created equal",
    "One nation's freedom equals #freedom for another $nation!",
]


def preprocess_text(text):
    # Remove digits.
    return re.sub(r"\d+", "", text).lower()


vectorizer = CountVectorizer(stop_words="english", preprocessor=preprocess_text)
word_matrix = vectorizer.fit_transform(lines)

feature_names = vectorizer.get_feature_names_out()
line_names = [f"Line {(i+1):d}" for i, _ in enumerate(word_matrix)]

df = pd.DataFrame(data=word_matrix.toarray(), index=line_names, columns=feature_names)
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

|  | ago | brought | conceived | created | dedicated | equal | equals | fathers | forth | freedom | liberty | men | nation | new | proposition | score | years |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| Line 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
| Line 2 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| Line 3 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 |
| Line 4 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 |

</div>
