# SpamDetection
**SpamDetection** is a Logistic Regression based model with Linear Discriminant Analysis (LDA) for identifying spam messages in text.
# Setup
## Installing JupyterLab
Using `pip`:</br>
`pip install jupyterlab`</br></br>
Using Anaconda:</br>
`conda install -c conda-forge jupyterlab`</br></br>
If you only want to use the model, you don't need JupyterLab or Anaconda. Instead, install Flask in a virtual environment.
## Setting Up Flask
1. Create a Virtual Environment</br></br>
`conda create -n spam-detection-env python=3.12`</br></br>
2. Activate Virtual Environment</br></br>
   `conda activate spam-detection-env`</br></br>
3. Install Dependencies</br></br>
   `pip install Flask`</br></br>
4. Install Additional Packages</br></br>
   `pip install -r requirements.txt`</br></br>
## Starting JupyterLab
`conda activate spam-detection-env`</br>`jupyter lab`</br></br>
## Files and Structure
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
  
## Technologies Used
**Programming Language**
  * `Python 3.12.8`:<a href="https://www.python.org/downloads/release/python-3128/" target="_blank">Python Official</a></br>

**Libraries**
  * `NumPy 1.26.4`:<a href="https://numpy.org/doc/1.26/" target="_blank">Documentation</a></br>
    NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
  * `Pandas`:<a href="https://pandas.pydata.org/" target="_blank">Documentation</a></br>
    Pandas is a software library written for the Python programming language for data manipulation and analysis. In particular, it offers data structures and operations for manipulating numerical tables and time series.
  * `Scikit-learn`:<a href="https://scikit-learn.org/stable/" target="_blank">Documentation</a></br>
    scikit-learn is a free and open-source machine learning library for the Python programming language. It features various classification, regression and clustering algorithms including support-vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.
  * `NLTK`:<a href="https://www.nltk.org/" target="_blank">Documentation</a></br>
    The Natural Language Toolkit, or more commonly NLTK, is a suite of libraries and programs for symbolic and statistical natural language processing (NLP) for English written in the Python programming language. It supports classification, tokenization, stemming, tagging, parsing, and semantic reasoning functionalities.
  * `Matplotlib`:<a href="https://matplotlib.org/" target="_blank">Documentation</a></br>
    Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits like Tkinter, wxPython, Qt, or GTK.
  * `Flask`:<a href="https://flask.palletsprojects.com/en/stable/" target="_blank">Documentation</a></br>
    Flask is a micro web framework written in Python. It is classified as a microframework because it does not require particular tools or libraries. It has no database abstraction layer, form validation, or any other components where pre-existing third-party libraries provide common functions.

## WORKING
**1. Dataset Analysis** (`Dimension_Analysis.ipynb`)
* Analysing Data for variance, low bias, and is linearly classifiable.
  
**2. Model Selection** (`Model_Selection.ipynb`)
* Several models were compared, and Logistic Regression was chosen due to its high accuracy and F1 scores:
  * Train Accuracy: 99.82%
  * Test Accuracy: 98.57%
  * Train F1 Score: 0.99
  * Test F1 Score: 0.94

**3. Final Model** (`Final_Model.ipynb`)
* Implements the Logistic Regression model with LDA for dimensionality reduction.
  
**4. Flask Application**
* A user-friendly interface allows users to input a message and receive predictions (Spam or Ham).

## UI
### Landing Page
![Screenshot 2025-01-19 192926](https://github.com/user-attachments/assets/26d78b29-e2fd-40fd-a1be-fa5d00af1e3f)
### Prediction Page
![Screenshot 2025-01-19 192945](https://github.com/user-attachments/assets/76e77995-b44e-43f0-bfc3-4c124bc96589)
## Author
<a href="https://github.com/deepanshu800/" target="_blank">Deepanshu Bhargav</a></br>
<a href="https://github.com/TDGrovin/" target="_blank">Grovin Singh Atwal</a>
