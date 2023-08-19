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
There is an attached fine tuned model, so you can load it and skip the fine tuning process. 

```python
learn_inf = load_learner('/content/drive/MyDrive/Colab Notebooks/oxford-iiit-pet/model.pkl') # load the model.
```
To make a test image, search and download an image from google o whatever internet browser you use, that fits one of the dog breed categories defined in Oxford-IIIT Pet Dataset https://www.robots.ox.ac.uk/~vgg/data/pets/
This image should be the test image for the model to predict. make sure to name the downloaded image as test.jpg.

Lastly I want to make a GUI to the predictor so that I can input it the test image that I doanloaded, press a button and see the predicted category in the GUI. 
```python
# Gradio GUI + user defined method / function

import numpy as np
import gradio as gr

# input test image, resize it to meet model requirements, pass it to the predictor and output the category name as a string
def predict_category_from_image(img):
  img1 = img.resize((224, 224))
  result = learn_inf.predict(img1)
  return str(result[0])

# define the GUI
demo = gr.Interface(
    fn=predict_category_from_image,
    inputs=[gr.Image(type="pil")],
    outputs=["text"],
)

# lounch gradio GUI
demo.launch()
```
