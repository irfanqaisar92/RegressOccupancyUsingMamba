# Occupancy Prediction with Mamba

This repository contains a deep learning model that uses the Mamba architecture to predict indoor occupancy levels based on sensor data.

## Sample Code

```python
import torch
from torch import nn

class SimpleModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)
```


Let me know if you want to automatically generate or edit your `README.md` file with content from your model or Colab notebook!
