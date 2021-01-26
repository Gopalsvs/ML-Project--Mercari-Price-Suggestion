# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import string
import nltk
import re
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from scipy import stats
from scipy.stats import norm,skew

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
class EDA():
    def __init__(self,path):
        self.train = pd.read_csv(path,delimiter = '\t',encoding='utf-8')
        
    #1. distribution plots and probability plot before converting price fto log form. 
    #2. To see skewness.
    def probability(self):
        fig1=plt.figure()
        sns.distplot((self.train['price']),fit=norm)
        plt.ylabel('freq')
        fig2=plt.figure()
        stats.probplot((self.train['price']),plot=plt)

        fig1=plt.figure()
        sns.distplot((np.log(self.train['price']+1)),fit=norm)
        plt.ylabel('freq')
        fig2=plt.figure()
        stats.probplot((np.log(self.train['price']+1)),plot=plt)

    def box_plot(self):
        sns.boxplot(np.log(self.train['price']+1))
        
    # removing data which has price as zero.
    def remove_zeros(self):
        price_zero_rows = self.train[self.train['price'] == 0].index
        self.train = self.train.drop(price_zero_rows)
        
    # Distribution plot betwen shipping condition's. How price varies with shipping value. 
    def ship_comp(self):
        fig = plt.figure(figsize = (10, 5))
        sns.distplot(np.log(self.train[self.train.shipping == 1]['price']+1), label = "shipping fee is paid by seller")
        sns.distplot(np.log(self.train[self.train.shipping == 0]['price']+1), label = "shipping fee is paid by buyer")
        plt.legend(fontsize = 10)
        
    # how item condition's are distributed. 
    def item_comp(self):
        self.train['log_price'] = np.log(self.train['price'] + 1)
        g = sns.FacetGrid(self.train, col = "item_condition_id")
        g.map(sns.distplot, "log_price")
        
    # Seeing the percentage of the data with respect to item condition.
    def percent_item(self):
        for i in range(5):
            print("No of items of item_condition"+str(i+1)+':'+str((self.train['item_condition_id']==i+1).value_counts()[1])+' percentage:'+str(((self.train['item_condition_id']==i+1).value_counts()[1]*100)/self.train.shape[0]))
        sns.countplot(x="item_condition_id", data=self.train)
        
    # replpace nan data.
    def replace_null(self):
        self.train['category_name'] = self.train['category_name'].replace({ np.nan:'Other/Other/Other'})
        self.train['brand_name'] = self.train['brand_name'].replace({ np.nan:'Other'})
        self.train['item_description'] = self.train['item_description'].replace({ np.nan:'undescribed'})
        
    # Extract category to observe which sub category is more present.
    def extract_category(self):
        extract = self.train['category_name'].str.extract("(?P<main_cat>[^/]+)/(?P<sub_cat1>[^/]+)/(?P<sub_cat2>[^/]+)")
        self.train['main_cat'] = extract['main_cat']
        self.train['sub_cat1'] = extract['sub_cat1']
        self.train['sub_cat2'] = extract['sub_cat2']
        self.train.drop(['category_name'], axis = 1, inplace = True)
        
        print(self.train['main_cat'].value_counts())
        fig, ax = plt.subplots(figsize = (10,5))
        self.train['main_cat'].value_counts().plot(ax=ax, kind='bar')
        
    # distribution of different brands 
    def brands(self):
        print(self.train["brand_name"].value_counts())

        plt.figure(figsize = (25, 10))
        sns.barplot(self.train["brand_name"].value_counts()[1:20].index, self.train["brand_name"].value_counts()[1:20].values)
    
    def length_item_description(text):
        return len(text.split())
    # length of desccription (max,min,mean)
    def len_item(self):
        self.train['len_item_description'] = self.train['item_description'].apply(EDA.length_item_description)
        fig1=plt.figure()
        sns.distplot(self.train['len_item_description'],fit=norm)
        plt.ylabel('freq')
        fig2=plt.figure()
        sns.distplot(np.log(self.train['len_item_description']), fit=norm)
        plt.ylabel('freq')

        max_len = self.train['len_item_description'].max()
        min_len = self.train['len_item_description'].min()
        mean_len = self.train['len_item_description'].mean()
        self.train['len_item_description'] = self.train['len_item_description'].apply(lambda x: (x - mean_len) / (max_len - min_len))
        
    # clean_text is funciton which cleans text and returns in the form of tokens after applying lemmatization for the tokens. 
    def clean_text(text):
        stopwords = nltk.corpus.stopwords.words('english')
        ps = nltk.PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        # negative words should be removed - "against", "no", "nor", "not", "couldn't", "didn't", "doesn't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "needn't", "shan't", "shouldn't", "wasn't"
        #                                    "weren't", "won't", "wouldn't"

        neg_words = ['against', 'no', 'nor', 'not']
        # expanding contractions.
        contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", 
                            "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", 
                            "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would",
                            "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will",
                            "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                            "I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                            "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                            "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is",
                            "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not",
                            "mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
                            "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not",
                            "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                            "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
                            "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", 
                            "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", 
                            "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is",
                            "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                            "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", 
                            "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                            "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have",
                            "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", 
                            "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", 
                            "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", 
                            "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", 
                            "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                            "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", 
                            "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

        for i in neg_words:
            stopwords.remove(i)
            
        for contraction,expand in contraction_dict.items():
            text = text.replace(contraction,expand) #replacing contractions
        text = "".join([word.lower() for word in text if word not in string.punctuation]) # lower the text and remove punctuation !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
        tokens = re.split('\W+', text) #split the lowered text into tokens, \W matches any character which is not a Unicode word character.
        #text = [ps.stem(word) for word in tokens if word not in stopwords] #stemming tokens
        text = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords] #lemmatization
        return text
    
    # wordcloud is used to observe which word is most used in name and item description.
    def wordcloud(self):
        tokens = EDA.clean_text(self.train['name'])
        frequency_dist = nltk.FreqDist(tokens)
        wordcloud = WordCloud().generate_from_frequencies(frequency_dist)
        fig = plt.figure(figsize = (7,6), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.tight_layout(pad = 0)
        
        tokens = EDA.clean_text(self.train['item_description'])
        frequency_dist = nltk.FreqDist(tokens)
        wordcloud = WordCloud().generate_from_frequencies(frequency_dist)
        fig = plt.figure(figsize = (7,6), facecolor = None)
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.tight_layout(pad = 0)
        
if __name__ == "__main__":
    path = "../input/mercari/train.tsv"
    
    eda = EDA(path)
    eda.probability()
    eda.box_plot()
    eda.remove_zeros()
    eda.ship_comp()
    eda.item_comp()
    eda.percent_item()
    eda.replace_null()
    eda.extract_category()
    eda.brands()
    eda.len_item()
    eda.wordcloud()
    
