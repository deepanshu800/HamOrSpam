# SpamDetection
**SpamDetection** is a Logistic Regression based model with Linear Discriminant Analysis (LDA) for identifying spam messages in text.

# Table of Contents
<ol>
  <li>
    <a href="#1-setup">Setup</a>
    <ul>
      <li><a href="#11-installing-jupyter">Installing with Jupyter</a></li>
      <li><a href="#12-installing-requirements">Installing Requirements</a></li>
    </ul>
  </li>
  <li><a href="#2-files-and-structure">Files and Structure</a></li>
  <li><a href="#3-technologies-used">Technologies Used</a></li>
  <li>
  <a href="#4-working">Working</a>
    <ul>
    <li><a href="#41-dataset-analysis">Dataset Analysis</a></li>
    <li><a href="#42-dimension-analysis-dimension_analysisipynb">Dimension Analysis</a></li>
    <li><a href="#43-model-selection-model_selectionipynb">Model Selection</a></li>
    <li><a href="#44-final-modelfinal_modelipynb">Final Model</a></li>
    <li><a href="#45-flask-application">Flask Application</a></li>
    </ul>
  </li>
  <li><a href="#5-ui">UI</a></li>
</ol>

# 1. Setup
If you only want to use the model, you do not need Jupyter.
## 1.1 Installing Jupyter
We recommend using Anaconda for development of this project
<br/><br/>
Using `pip`:
```sh
pip install jupyter
```
Using `conda`:
```sh
conda install -c conda-forge jupyter
```
### To open Jupyter (from conda environment):
```sh
cd <path-to-project-directory>
```
To open jupyter-lab
```sh
jupter-lab
```
To open jupyter-notebook
```sh
jupyter-notebook
```
Note: It is preferred to install Jupyter using `pip` instead of `conda` because the latter may contain older version of Jupyter
</br>
## 1.2 Installing Requirements
### a. Create a Virtual Environment
Using `conda`:
```sh
conda create -n <env-name> python=3.13.0
```
Using `Python venv`:
```sh
python -m venv <virtual-environment-name>
```
### b. Activate Virtual Environment
```sh
conda activate spam-detection-env
```
Or
```sh
<path-to-virtual-environment>/Scripts/activate
```
### c. Install Requirements
```sh
pip install -r requirements.txt
```
# 2. Files and Structure
**1. Dataset**
* `spam.csv`</br>Contains labeled data for training and testing the spam detection model.</br>

**2. Notebooks**
* `Dimension_Analysis.ipynb`</br>Reduce the dimensionality of dataset using various methods.
* `final_model.ipynb`</br>Inplements the final model after testing and tuning</br>
* `model_selection.ipynb`</br>Compares the performance of different models and selects the best one. Logistic Regression was chosen for its superior performance on linearly classifiable data.</br>

**3. Static**</br>
  Static Files are the files that the backend server uses to serve to the client.</br>
  
**4. Templates**
* `index.html`</br>
The landing page for the application.</br>
* `predict.html`</br>
The page for submitting and displaying predictions.
  
**5. Weights**</br>
Contains pre-trained model components:</br>
* `count_vectoriser.pkl`</br>
Converts text messages into feature vectors.
* `model_weight.pkl`</br>
Stores the trained Logistic Regression model.
* `selector_weight.pkl`
Features selector for dimensionality reduction.

**6. Flask Application**
* `app.py`</br>
The root file of the Flask application, containing routes and functions for predictions.
   
**7. Requirements**
* `requirements.txt`</br>
Lists all required Python libraries. Install them with:</br>
  `pip install -r requirements.txt `</br>
  **OR**</br>
  `pip install package_1, package_2, package_3,...`</br>
  
# 3. Technologies Used
**Programming Language**
  * Python 3.13.0: <a href="https://docs.python.org/3.13/" target="_blank"><img src="https://www.python.org/static/img/python-logo.png" width=100 align="center"></a>

**Libraries**
  * `NumPy 2.1.2`:<a href="https://numpy.org/doc/1.26/" target="_blank"><img src="https://numpy.org/images/logo.svg" width=50 align="center"></a></br>
    NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
  * `Pandas 2.2.3`:<a href="https://pandas.pydata.org/" target="_blank"><img src="https://pandas.pydata.org/static/img/pandas_white.svg" width=100 align="center"></a></br>
    Pandas is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series.
    
  * `Scikit-learn 1.5.2`: <a href="https://scikit-learn.org/stable/" target="_blank"><img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width=75 align="center"></a></br>
    scikit-learn is a free and open-source machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support-vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.
    
  * `NLTK 3.9.1`: <a href="https://www.nltk.org/" target="_blank">Documentation</a></br>
    The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and statistical natural language processing (NLP) for English written in the Python programming language. It supports classification, tokenization, stemming, tagging, parsing, and semantic reasoning functionalities.
    
  * `Matplotlib 3.9.2`: <a href="https://matplotlib.org/" target="_blank"><img src="https://matplotlib.org/stable/_static/logo_dark.svg" width=100 align="center"></a></br>
    Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits like Tkinter, wxPython, Qt, or GTK.
    
  * `Flask 3.1.0`: <a href="https://flask.palletsprojects.com/en/stable/" target="_blank"><img src="https://flask.palletsprojects.com/en/stable/_images/flask-horizontal.png" width=100 align="center"></a></br>
    Flask is a micro web framework written in Python. It is classified as a microframework because it does not require particular tools or libraries. It has no database abstraction layer, form validation, or any other components where pre-existing third-party libraries provide common functions.

# 4. Working
## 4.1 Dataset Analysis
* Our data was linear because linear models had better accuracy than non-linear models.
* Our dataset has low variance because subsets of dataset had similiar accuracies.

## 4.2 Dimension Analysis (`dimension_analysis.ipynb`)
* Our dataset had text messages with corressponding labels. After NLP, we produced a sparse matrix X of dimensions 5572 x 6221.
* Dimensionality reduction was needed and we implemented both feature selection and feature projection techniques.
* Linear Discriminat Analysis performed the best and reduced the dimensionality from 6221 -> 1.

## 4.3 Model Selection (`model_selection.ipynb`)
* Both linear and non-linear classification models were implemented
* Logistic Regression performed the best with evaluation metric:
  * Train Accuracy: 99.82%
  * Test Accuracy: 98.57%
  * Train F1 Score: 0.99
  * Test F1 Score: 0.94

## 4.4 Final Model(`Final_Model.ipynb`)
* Our final model performs classification by first reducing dimension of dataset and then training on Logistic Regression model. 
  
## 4.5 Flask Application
* A user-friendly interface allows users to input a message and receive predictions (Spam or Ham).

# 5. UI
### Landing Page
![Screenshot 2025-01-19 192926](https://github.com/user-attachments/assets/29a6eeaa-3499-423a-9937-d56f7c7d5d53)
### Prediction Page
![Screenshot 2025-01-19 192945](https://github.com/user-attachments/assets/2988b0b3-355f-4fe3-9422-21e70c0e974c)
## Author
<a href="https://github.com/deepanshu800/" target="_blank">Deepanshu Bhargav</a></br>
<a href="https://github.com/TDGrovin/" target="_blank">Grovin Singh Atwal</a>
