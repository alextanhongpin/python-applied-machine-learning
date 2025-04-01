
``` python
import numpy as np
import onnxruntime as rt

session = rt.InferenceSession("taxi.onnx")
input_name = session.get_inputs()[0].name  # float_input
label_name = session.get_outputs()[0].name  # output_probability


input = np.array(
    [
        [
            4.0,  # Day of week
            17.0, # Pickup time (hour of day)
            2.0,  # Distance to travel
        ]
    ],
    dtype=np.float32,
)
score = session.run([label_name], {input_name: input})[0][0][0]
score
```

    np.float32(11.4910555)
