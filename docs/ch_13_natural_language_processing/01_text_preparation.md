# Text Preparation

``` python
from tensorflow.keras.preprocessing.text import Tokenizer

lines = [
    "The quick brown fox",
    "Jumpts over $$$ the lazy brown dog",
    "Who jumpts high into the blue sky after counting 123",
    "And quickly returns to earth",
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
```

``` python
sequences
```

    [[1, 4, 2, 5],
     [3, 6, 1, 7, 2, 8],
     [9, 3, 10, 11, 1, 12, 13, 14, 15, 16],
     [17, 18, 19, 20, 21]]

``` python
tokenizer.sequences_to_texts(sequences)
```

    ['the quick brown fox',
     'jumpts over the lazy brown dog',
     'who jumpts high into the blue sky after counting 123',
     'and quickly returns to earth']

Stop words needs to be removed manually

``` python
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")


def remove_stop_words(text):
    text = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    text = [word for word in text if word.isalpha() and not word in stop_words]
    return " ".join(text)


lines = list(map(remove_stop_words, lines))


tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)
sequences
```

    [nltk_data] Downloading package stopwords to
    [nltk_data]     /Users/alextanhongpin/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!

    [[3, 1, 4], [2, 5, 1, 6], [2, 7, 8, 9, 10], [11, 12, 13]]

``` python
tokenizer.sequences_to_texts(sequences)
```

    ['quick brown fox',
     'jumpts lazy brown dog',
     'jumpts high blue sky counting',
     'quickly returns earth']

All sequences must have the same length

``` python
from tensorflow.keras.preprocessing.sequence import pad_sequences

padded_sequences = pad_sequences(sequences, maxlen=4)
padded_sequences
```

    array([[ 0,  3,  1,  4],
           [ 2,  5,  1,  6],
           [ 7,  8,  9, 10],
           [ 0, 11, 12, 13]], dtype=int32)

``` python
tokenizer.sequences_to_texts(padded_sequences)
```

    ['quick brown fox',
     'jumpts lazy brown dog',
     'high blue sky counting',
     'quickly returns earth']

Padding and truncation happens on the left.
