# FastAi Deep Learning for coders lesson 2 (deployment examples using Gradio and HuggingFace) - Draft. 

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
```python
# import gradio package
import gradio as gr
```
The original model is resnet34 that was pretrained using ImageNet (image database). I downloaded it as a pretrained model, then fine tuned it (a concept called transfer learning) with Oxford-IIIT Pet Dataset.
Then I saved it in google drive that was loaded to colab.
The reason I do not present the fine tuning here of the model is because it took me a long time to run it on google colab free account.
There is an attached fine tuned model, so you can upload it and skip the fine tuning process. 

```python
learn_inf = load_learner('/content/drive/MyDrive/Colab Notebooks/oxford-iiit-pet/model.pkl') # load the model.
```
