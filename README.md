# Text Clustering

<img src ='https://miro.medium.com/max/963/1*cqbr8G-HWBGqPefIFc7kmg.png' alt='text-clustrimg-image'> 

<br />

# Project Overview 

In this project we will go together into Gutenberg Books data to produce similar clusters and compare them; analyze the pros and cons of algorithms, generate and communicate the insights using natural language processing NLP techniques

# Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install these packages.

pyLDAvis is a python library for interactive topic model visualization. This is a port of the fabulous R package by Carson Sievert and Kenny Shirley.

pyLDAvis is designed to help users interpret the topics in a topic model that has been fit to a corpus of text data. The package extracts information from a fitted LDA topic model to inform an interactive web-based visualization.

```bash
! pip install pyLDAvis
```

PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows. Fast + Explainable + Scalable
```
pip install pycaret[full]
```




# Usage

```python
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.probability import FreqDist
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import scipy.cluster.hierarchy as sch
from scipy import stats
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import pyLDAvis
import gensim.corpora as corpora
from gensim.models.ldamodel import LdaModel
import pyLDAvis.gensim_models as gensimvis
from sklearn.manifold import TSNE
%matplotlib inline
```

## How To Run this Notebook 
when you open the notebook run all cells 
There is a cell to install all the necessary libraries 
There will be an error when the LDA part is reached because there are packages dependency error
The kernel should be restarted and start running from LDA section first cell 