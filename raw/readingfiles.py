import os
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 

remove_stop_words = False
lemmatize = False

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text=text.replace('{html}',"") 
    remove_chars = re.compile('<.*?>')
    clean_text = re.sub(remove_chars, '', text)
    remove_numbers = re.sub('[0-9]+', '', clean_text)
    result = word_tokenize(remove_numbers)

    if remove_stop_words:
        result = [word for word in result if not word in stop_words]
    if lemmatize:
        result = [lemmatizer.lemmatize(word) for word in result]
    return result


def extract_info(folder: str, category: str):
    prefix = os.path.join(os.getcwd(), 'raw', folder, category)
    file_list = [os.path.join(prefix, f) for f in os.listdir(prefix) if f.endswith('.txt')]
    text = list()
    for filename in file_list:
        open_file = open(filename, 'r', encoding='utf8')
        # text_data = open_file.read().split('\n')
        # text_data = list(filter(None, text_data))
        text_data = open_file.read()
        text_data = preprocess(text_data)
        text.append(text_data)

    category_df = pd.DataFrame({"text": text, "review": category})
    return category_df  

def make_datasets():
    testing_df = pd.DataFrame()
    training_df = pd.DataFrame()
    unsupervised = pd.DataFrame()

    for cat in ['neg', 'pos']:
        testing_df = testing_df.append(extract_info("test", cat))
        testing_df.to_csv(r'testing_data.csv')

        training_df = training_df.append(extract_info("train", cat))
        training_df.to_csv(r'training_data.csv')

        unsupervised = unsupervised.append(extract_info("train", "unsup"))
        unsupervised.to_csv(r'unsupervised_data.csv')
