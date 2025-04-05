# Factoring Word Order into Predictions

``` python
import pandas as pd

df = pd.read_csv("Data/reviews.csv", encoding="ISO-8859-1")
df = df.sample(frac=1, random_state=0)
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

|       | Text                                              | Sentiment |
|-------|---------------------------------------------------|-----------|
| 11841 | Al Pacino was once an actor capable of making ... | 0         |
| 19602 | After Chaplin made one of his best films: Doug... | 0         |
| 45519 | This movie is sort of a Carrie meets Heavy Met... | 1         |
| 25747 | I have fond memories of watching this visually... | 1         |
| 42642 | In the '70s, Charlton Heston starred in sci-fi... | 1         |

</div>

``` python
df = df.drop_duplicates()
df.groupby("Sentiment").describe()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead tr th {
        text-align: left;
    }
&#10;    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>

|  | Text |  |  |  |
|----|----|----|----|----|
|  | count | unique | top | freq |
| Sentiment |  |  |  |  |
| 0 | 24697 | 24697 | Al Pacino was once an actor capable of making ... | 1 |
| 1 | 24884 | 24884 | This movie is sort of a Carrie meets Heavy Met... | 1 |

</div>

``` python
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Embedding,
    Flatten,
    GlobalMaxPooling1D,
    InputLayer,
    MaxPooling1D,
    TextVectorization,
)
from tensorflow.keras.models import Sequential

max_words = 20000
max_length = 500

model = Sequential()
model.add(InputLayer(input_shape=(1,), dtype=tf.string))
model.add(TextVectorization(max_tokens=max_words, output_sequence_length=max_length))
model.add(Embedding(max_words, 32, input_length=max_length))
model.add(Conv1D(32, 7, activation="relu"))
model.add(MaxPooling1D(5))
model.add(Conv1D(32, 7, activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     text_vectorization (TextVe  (None, 500)               0         
     ctorization)                                                    
                                                                     
     embedding (Embedding)       (None, 500, 32)           640000    
                                                                     
     conv1d (Conv1D)             (None, 494, 32)           7200      
                                                                     
     max_pooling1d (MaxPooling1  (None, 98, 32)            0         
     D)                                                              
                                                                     
     conv1d_1 (Conv1D)           (None, 92, 32)            7200      
                                                                     
     global_max_pooling1d (Glob  (None, 32)                0         
     alMaxPooling1D)                                                 
                                                                     
     dense (Dense)               (None, 1)                 33        
                                                                     
    =================================================================
    Total params: 654433 (2.50 MB)
    Trainable params: 654433 (2.50 MB)
    Non-trainable params: 0 (0.00 Byte)
    _________________________________________________________________

``` python
X = df["Text"]
y = df["Sentiment"]

model.layers[0].adapt(X)
```

``` python
hist = model.fit(X, y, validation_split=0.5, epochs=5, batch_size=250)
```

    Epoch 1/5
    100/100 [==============================] - 4s 41ms/step - loss: 0.6183 - accuracy: 0.6517 - val_loss: 0.3658 - val_accuracy: 0.8407
    Epoch 2/5
    100/100 [==============================] - 4s 41ms/step - loss: 0.2795 - accuracy: 0.8839 - val_loss: 0.2984 - val_accuracy: 0.8758
    Epoch 3/5
    100/100 [==============================] - 4s 41ms/step - loss: 0.1574 - accuracy: 0.9428 - val_loss: 0.3266 - val_accuracy: 0.8716
    Epoch 4/5
    100/100 [==============================] - 4s 41ms/step - loss: 0.0939 - accuracy: 0.9715 - val_loss: 0.3820 - val_accuracy: 0.8696
    Epoch 5/5
      7/100 [=>............................] - ETA: 2s - loss: 0.0516 - accuracy: 0.9891

``` python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


acc = hist.history["accuracy"]
val_acc = hist.history["val_accuracy"]
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, "-", label="Training Accuracy")
plt.plot(epochs, val_acc, ":", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right");
```

``` python
text = "Excellent food and fantastic service!"
model.predict([text])[0][0]
```

``` python
text = "The long lines and poor customer service really turned me off"
model.predict([text])[0][0]
```
