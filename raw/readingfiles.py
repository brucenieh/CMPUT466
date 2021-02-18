import os
import pandas as pd


# Enter a local path to get the files
def extract_info(folder: str, category: str):
    path = 'C:/Users/diell/Desktop/aclImdb/' + folder + "/" + category
    prefix = os.path.abspath(path) 
    file_list = [os.path.join(prefix, f) for f in os.listdir(prefix) if f.endswith('.txt')]

    text = list()
    for filename in file_list:
        open_file = open(filename, 'r', encoding='utf8')
        text_data = open_file.read().split('\n')
        text_data = list(filter(None, text_data))
        text.append(text_data)

    category_df = pd.DataFrame({"text": text, "review": category})
    return category_df  

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

