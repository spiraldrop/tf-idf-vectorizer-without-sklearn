# tf-idf-vectorizer-without-sklearn
TF-IDF is a statistical measure that evaluates how relevant a word is to a documents in a corpus. Here, I have implemented the metric without using sklearn's given function.

### About this project
I completed this assignment as a part of my Machine Learning course. I have used two functions, ```fit``` and```transform```. The ```fit()``` function computes the unique words and calculates their IDF values. And the ```transform()``` function creates a sparse matrix of the vocabulary from our corpus after calculating their TF-IDF values. A sparse matrix is used to efficiently display the values and save a significant amount of memory as well as speed up the processing of that data. I have also included a variation of our original task, named 'Task-2' in the document. It is much similar to the first task except in this, we calculate only the vectors/words with the top 50 IDF values. In Task-1, a simple list of sentences is used as corpus, whereas in Task-2, I have used a pickle file 'cleaned_strings' as corpus.


### Libraries needed
You need to install the following methods and libraries: 
```
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
from scipy.sparse import csr_matrix
import math
import operator
import itertools
from sklearn.preprocessing import normalize
import numpy as np
```
#### Link to the course
https://www.appliedaicourse.com/course/11/Applied-Machine-learning-course 
