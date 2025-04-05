# Arcface

``` python
from arcface import ArcFace
from astropy.utils.data import download_file
from PIL import Image

model_path = download_file(
    "https://www.digidow.eu/f/datasets/arcface-tensorflowlite/model.tflite"
)

af = ArcFace.ArcFace(model_path=model_path)
```

``` python
from PIL import Image

face0 = Image.open("face0.jpg")
face0
```

![](05_arcface_files/figure-commonmark/cell-3-output-1.png)

``` python
face1 = Image.open("face1.jpg")
face1
```

![](05_arcface_files/figure-commonmark/cell-4-output-1.png)

``` python
from sklearn.metrics.pairwise import cosine_similarity

face_emb1 = af.calc_emb("Faces/Jeff/Jeff-1.jpg")
face_emb2 = af.calc_emb("Faces/Jeff/Jeff-2.jpg")
face_emb3 = af.calc_emb("Faces/Lori/Lori-1.jpg")
face_emb4 = af.calc_emb("Faces/Lori/Lori-2.jpg")

sim = cosine_similarity([face_emb1, face_emb3], [face_emb1, face_emb2, face_emb3, face_emb4])
sim
```

    array([[ 0.9999999 ,  0.3375299 ,  0.12884738, -0.06618136],
           [ 0.12884738, -0.04451956,  1.0000001 ,  0.44539228]],
          dtype=float32)
