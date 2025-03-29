# Indoor Occupancy Regression using Mamba Architecture

This repository contains a Jupyter notebook for indoor occupancy regression modeling using a Mamba-based deep learning architecture. The model predicts the number of occupants in a room based on historical sensor and environmental data.

## Key Features
- Occupancy Number Regression
- Mamba architecture: sequence modeling with state-space models
- Time series preprocessing including normalization and sequence generation
- Model evaluation using MAE, RMSE, and R² metrics
- Includes training loop, loss visualization, and performance summary

## Requirements

```
mamba-ssm
torch
pandas
numpy
scikit-learn
matplotlib
notebook
```


## Dataset Info
This model was trained on occupancy time-series data from an indoor room setting, including features like:
- Hour of the day
- Relative humidity
- Temperature
- Previous occupancy values


## Visualization

![image](https://github.com/user-attachments/assets/294affe7-ed7e-4d42-9944-1f39bebb0c4a)
![image](https://github.com/user-attachments/assets/3f055b52-7a93-4978-8f6f-359b3242d75a)
![image](https://github.com/user-attachments/assets/1715c898-3529-4b8a-b861-bf5d0d7acdff)



## Acknowledgements
This project was developed as part of a research study on occupant-centric building control and deep-learning methods for indoor occupancy estimation.

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
