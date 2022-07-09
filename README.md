<h1 align="center">Hybrid learning prototype</h1> 

<p align="center"> 
<a href="https://github.com/lprtk/hybrid-learning-prototype/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/lprtk/hybrid-learning-prototype"></a> 
<a href="https://github.com/lprtk/hybrid-learning-prototype/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/lprtk/hybrid-learning-prototype"></a> 
<a href="https://github.com/lprtk/hybrid-learning-prototype/stargazers"><img alt="Github Stars" src="https://img.shields.io/github/stars/lprtk/hybrid-learning-prototype"></a> 
<a href="https://github.com/lprtk/hybrid-learning-prototype/blob/master/LICENSE"><img alt="GitHub license" src="https://img.shields.io/github/license/lprtk/hybrid-learning-prototype"></a> 
<a href="https://github.com/lprtk/hybrid-learning-prototype/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a> 
</p> 

## Table of contents 
* [Overview :loudspeaker:](#Overview)
* [Content :mag_right:](#Content)
* [Requirements :page_with_curl:](#Requirements)
* [File details :open_file_folder:](#File-details)
* [Features :computer:](#Features) 

<a id="section01"></a> 
## Overview 

<p align="justify">The objective is to create a tool that can combine predictions from two different models, whether it is a regression or classification task. Generally, when doing time series for example, we can use traditional econometric models or more sophisticated Machine and Deep Learning models. Generally, econometric models provide good short-term predictions but poor long-term predictions while some Deep Learning models provide very good long-term predictions. Thanks to these classes, it is now possible to merge two forecast vectors according to an exponential coefficient: we give more weight in the short term to the forecasts of the first model (the econometric model for example) and conversely, we give more weight to the forecasts of the second model in the long term (Deep Learning model for example).<p>

<a id="section02"></a> 
## Content 

For the moment, two class with several functions are available:
<ul> 
<li><p align="justify">The JoiningRegressor class for regression task;</p></li> 
<li><p align="justify">The JoiningClassifier class for classification task.</p>
</li>
</ul> 

<a id="section03"></a> 
## Requirements
* **Python version 3.9.7** 
* **Install requirements.txt** 
```console
$ pip install -r requirements.txt 
``` 

* **Librairies used**
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
``` 

<a id="section04"></a> 
## File details
* **requirements**
* This folder contains a .txt file with all the packages and versions needed to run the project. 
* **joining_class**
* This folder contains a .py file with all class, functions and methods. 
* **example**
* This folder contains an example notebook to better understand how to use the different class and functions, and their outputs.

</br> 

Here is the project pattern: 
```
- project
    > hybrid-learning-prototype
        > requirements
            - requirements.txt
        > codefile 
            - joining_class.py
        > example 
            - joining_class.ipynb
```

<a id="section05"></a> 
## Features 
<p align="center"><a href="https://github.com/lprtk/lprtk">My profil</a> • 
<a href="https://github.com/lprtk/lprtk">My GitHub</a> • 
<a href="https://github.com/Mcompetitions/M5-methods/blob/master/M5-Competitors-Guide.pdf">Inspired by the M5 competition</a>
</p>
