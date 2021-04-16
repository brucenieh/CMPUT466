# CMPUT466

## Installation

#### Clone Repoistory

Clone the github repository to your workspace

#### Install Dependencies

The following python packages are required for the project:
```
nltk
numpy
pandas
sklearn
tensorflow
```

Install these packages with `pip3 install -r requirements.txt`


#### Download GloVe Embeddings

Download the GloVe embeddings using [this link](http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip) and extract the file `glove.6B.100d.txt` to the project root directory

#### Training and Predicting

The provided `main.py` file instantiates each model and performs training and prediction on the IMDb dataset. If you would like to test a model, it can be imported from the corresponding module in the `models` directory. If you are running the code for the first time, we need to build the training and testing dataset files, download relevant packages from nltk and build a vocabulary. This step only needs to be done once. It is time consuming to perform this step every time the program runs, so it would be meaningful to comment out [this line](https://github.com/brucenieh/CMPUT466/blob/1f0b78caee8d0e1523a3365270d1fa8052840565/main.py#L59) after it has been executed at least once.



To preprosess the data, we need to install a few nltk packages: 

```
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## References
n-gram Model in Python
 - From [Analytics Vidhya](https://medium.com/analytics-vidhya/a-comprehensive-guide-to-build-your-own-language-model-in-python-5141b3917d6d)
 - By [Mohd Sanad Zaki Rizvi](https://medium.com/@mohdsanadzakirizvi)
 - Retrieved on Feb 22, 2021

Global Vectors for Word Representation
 - From [Stanford](https://nlp.stanford.edu/projects/glove/)
 - By Jeffrey Pennington, Richard Socher, and Christopher D. Manning

Large Movie Review Dataset
 - From [Stanford](https://ai.stanford.edu/~amaas/data/sentiment/)
 - By Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts