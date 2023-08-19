# FastAi Deep Learning for coders lesson 2 (deployment examples using Gradio and HuggingFace)

## Preface
In this post I will give examples of:
1. 
2. 
## creating a dog breed category predictor
```python
from fastai.vision.all import *
import fastai
```
```python
# This is needed when you work with google colab, local python envioroment or alike where the envioroment that your work in doesn't have that package installed.
# This is not needed if you host the code in HugginFace, because gradio package is already installed there.
pip install gradio
```
