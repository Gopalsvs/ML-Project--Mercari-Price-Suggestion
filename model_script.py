# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]

import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
import re
import lightgbm as lgbm

from scipy import stats
from scipy.stats import norm,skew
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, cross_val_predict,cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.sparse import hstack
from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.linear_model import Ridge , LinearRegression,Lasso
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor


stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()
lemmatizer = WordNetLemmatizer()

# negative words should be removed - "against", "no", "nor", "not", "couldn't", "didn't", "doesn't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't", "needn't", "shan't", "shouldn't", "wasn't"
#                                    "weren't", "won't", "wouldn't"

neg_words = ['against', 'no', 'nor', 'not']

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
# clean text which converts text into tokens after using lemmatization.  
def clean_text(text):
    for contraction,expand in contraction_dict.items():
        text = text.replace(contraction,expand) #replacing contractions
    text = "".join([word.lower() for word in text if word not in string.punctuation]) # lower the text and remove punctuation !"#$%&'()*+, -./:;<=>?@[\]^_`{|}~
    tokens = re.split('\W+', text) #split the lowered text into tokens, \W matches any character which is not a Unicode word character.
    #text = [ps.stem(word) for word in tokens if word not in stopwords] #stemming tokens
    text = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords] #lemmatization
    return text
# text polarity and average of the sentence.
def polarity(text):
    blob = TextBlob(text)
    sentence_polarity = list(map(lambda x:x.polarity,blob.sentences))
    overall_polarity = sum(sentence_polarity)/len(sentence_polarity)
    return overall_polarity
# extracting manfacturing date
def extract_year(text):
    matches = [int(year) for year in re.findall('[0-9]{4}', text) if int(year) >= 1970 and int(year) <= 2018]
    if matches:
        return int(max(matches))
    else:
        return 0
# extracting numerical features like quantity.
default = '1'
def quantity(tokens):
    if(len(tokens)>=2):
        if(tokens[0] == 'get' or tokens[0] == 'new'):
            if(tokens[1].isnumeric()):
                return tokens[1] 
    if(len(tokens) >=1):
        if(len(tokens)>=2):
            if(tokens[0].isnumeric() and not tokens[1].isnumeric()):
                return tokens[0]
        else:
            if(tokens[0].isnumeric()):
                return tokens[0]
    return default    



def length_item_description(text):
    return len(text.split())

def preprocessing(data):    
    # creating features which has category and brand
    data['has_category'] = (data['category_name'].notnull()).astype('category')
    data['has_brand'] = (data['brand_name'].notnull()).astype('category')
    
    # splitting category into seperate sub category
    data['category_name'] = data['category_name'].replace({ np.nan:'Other/Other/Other'})
    extract = data['category_name'].str.extract("(?P<main_cat>[^/]+)/(?P<sub_cat1>[^/]+)/(?P<sub_cat2>[^/]+)")
    data['main_cat'] = extract['main_cat']
    data['sub_cat1'] = extract['sub_cat1']
    data['sub_cat2'] = extract['sub_cat2']
    data.drop(['category_name'], axis = 1, inplace = True)
    # filling nan data
    data['brand_name'] = data['brand_name'].replace({ np.nan:'Other'})
    data['item_description'] = data['item_description'].replace({ np.nan:'undescribed', 'No description yet':'undescribed', 'no description yet':'undescribed'})
    # converting into log version
    data['log_price'] = np.log(data['price'] + 1)
    data.drop(['price'], axis = 1, inplace = True)
        
def features(data,num):
    # num(mod prime) is used to combine different possible features.
    if(num % 2 == 0):
        data['nb_comb'] = data['name'] + ' '+data['brand_name']
        data['dnb_comb'] = data['item_description'] + ' '+data['name'] + ' '+data['brand_name'] + ' '+data['main_cat'] + ' '+data['sub_cat1'] + ' '+data['sub_cat2'] 

    if(num % 3 == 0):
        data['len_dnb_comb'] = data['dnb_comb'].apply(length_item_description)
        max_len = data['len_dnb_comb'].max()
        min_len = data['len_dnb_comb'].min()
        mean_len = data['len_dnb_comb'].mean()
        data['len_dnb_comb'] = data['len_dnb_comb'].apply(lambda x: (x - mean_len) / (max_len - min_len))

    if(num % 5 == 0):
        data['polarity_item_description'] = data['item_description'].apply(polarity)
        data['year'] = data['item_description'].apply(extract_year)
        
        temp = data['name'].apply(lambda x:int(quantity(clean_text(x))))
        temp[temp<=0] = 1
        data['quantity'] = temp
    
    encode_features = ColumnTransformer([
                                   ('main_cat', OneHotEncoder(dtype = 'int'),['main_cat']),   
                                   ('sub_cat1', OneHotEncoder(dtype = 'int'),['sub_cat1']),
                                   ('sub_cat2',OneHotEncoder(dtype = 'int'),['sub_cat2']),
                                   ('item_condition',OneHotEncoder(dtype = 'int'),['item_condition_id']),
                                   ('shipping',OneHotEncoder(dtype = 'int'),['shipping']),
                                   ('has_brand',OneHotEncoder(dtype = 'int'),['has_brand']),
                                   ('brand_name',OneHotEncoder(dtype = 'int'),['brand_name']),
                                   ('has_category',OneHotEncoder(dtype = 'int'),['has_category']),
                                   ('nb_comb', HashingVectorizer(analyzer = 'word', token_pattern = '\w+', ngram_range=(1, 2), n_features=2 ** 21, norm='l2', stop_words = stopwords), 'nb_comb'),
                                   ('item_description', TfidfVectorizer(max_features = 2 ** 18, analyzer = clean_text), 'item_description'),  
                                   ('dnb_comb', HashingVectorizer(analyzer = 'word', token_pattern = '\w+', ngram_range=(1, 3), n_features=2 ** 22, stop_words = stopwords, norm='l2'), 'dnb_comb')
                                  ]) 
    #numpy array of total features for total data
    total_tfidf = encode_features.fit_transform(data)
    return total_tfidf

def accuracy(y_predict_train,y_predict_test,y_train,y_test):
    y_predict_test = np.exp(y_predict_test)-1
    y_predict_train = np.exp(y_predict_train)-1
    print("train error:" +str(np.sqrt(mean_squared_log_error(y_train, y_predict_train))))
    print("test error:" +str(np.sqrt(mean_squared_log_error(y_test, y_predict_test))))

# Two ridge models with different solver methods and 80% train and 20% validation
def model(features,y_values,train_rows):
    train_df_processed = features[: train_rows]
    X_train, X_test, y_train, y_test = train_test_split(train_df_processed, y_values[:train_rows], test_size = 0.2, random_state = 42) 
    y_test = np.exp(y_test)-1 
    y_train = np.exp(y_train)-1

    Ridge_model2 = Ridge(alpha = 1.73, fit_intercept=True, normalize=False, copy_X=True, max_iter=1000, solver = 'sag', random_state = 42)
    Ridge_model2.fit(X_train, y_train)
    
    
    y_predict_train = Ridge_model2.predict(X_train)
    y_predict_test = Ridge_model2.predict(X_test) 
    accuracy(y_predict_train,y_predict_test,y_train,y_test)
    
    Ridge_model1 = Ridge(alpha = 1.75, fit_intercept=True, normalize=False, copy_X=True, max_iter=1000, solver = 'sparse_cg', random_state=None)
    Ridge_model1.fit(X_train, y_train)
    
    y_predict_train = Ridge_model1.predict(X_train)
    y_predict_test = Ridge_model1.predict(X_test) 
    accuracy(y_predict_train,y_predict_test,y_train,y_test)

    
#     reg = LinearRegression()
#     reg.fit(X_train,y_train)
    
#     y_predict_test = reg.predict(X_test) 
#     y_predict_train = reg.predict(X_train)
#     accuracy(y_predict_train,y_predict_test,y_train,y_test)
    
    
#     regr = SVR(kernel = 'linear')
#     regr.fit(X_train,y_train)
    
#     y_predict_test = regr.predict(X_test) 
#     y_predict_train = regr.predict(X_train)
#     accuracy(y_predict_train,y_predict_test,y_train,y_test)
    

#     ada_boost = AdaBoostRegressor(n_estimators = 10, base_estimator=Ridge(alpha = 1.75, fit_intercept=True, normalize=False, copy_X=True, max_iter=1000, solver = 'sparse_cg', random_state=None))
#     ada_boost.fit(X_train,y_train)
    
#     y_predict_train = ada_boost.predict(X_train)
#     y_predict_test = ada_boost.predict(X_test) 
#     accuracy(y_predict_train,y_predict_test,y_train,y_test)


#     params = {
#             'learning_rate': 0.75,
#             'application': 'regression',
#             'max_depth': 3,
#             'num_leaves': 100,
#             'verbosity': -1,
#             'nthread': 4
#         }
#     train_data = lgbm.Dataset(X_train, label = y_train)
#     model = lgbm.train(params, train_set = train_data, num_boost_round = 300, verbose_eval = 100)

#     regressor = RandomForestRegressor(n_estimators = 100, random_state = 0) 
#     regressor.fit(X_train, y_train)
    
    
#     y_predict_train = regressor.predict(X_train)
#     y_predict_test = regressor.predict(X_test) 
#     accuracy(y_predict_train,y_predict_test,y_train,y_test)
      
    
    # model is combination of two ridge models with different solver methods and trained with full train_data
def submission(features,y_values,train_rows,test_submission):
    Ridge_model1 = Ridge(alpha = 1.75, fit_intercept=True, normalize=False, copy_X=True, max_iter=1000, solver = 'sparse_cg', random_state=None)
    Ridge_model1.fit(features[:train_rows], y_values[:train_rows])
    
    Ridge_model2 = Ridge(alpha = 1.73, fit_intercept=True, normalize=False, copy_X=True, max_iter=1000, solver = 'sag', random_state = 42)
    Ridge_model2.fit(features[:train_rows], y_values[:train_rows])
    
    y_test_predict = 0.5 * Ridge_model1.predict(features) + 0.5 * Ridge_model2.predict(features)
    y_test_predict = np.exp(y_test_predict)-1
    
    test_predict = pd.DataFrame(data = y_test_predict, columns = ['price'])
    test_submission = pd.concat((test_submission, test_predict), axis=1)
    
    test_submission.loc[test_submission['price'] < 0.0, 'price'] = 0.0
    test_submission.to_csv('submission_test.csv', index=False)


# def rmsle(y, y_pred, **kwargs):
#     out = np.sqrt(mean_squared_log_error(np.exp(y)-1, np.exp(y_pred)-1))
#     return out

# def cross_valid_score():
#     rmsle_scorer = make_scorer(rmsle, greater_is_better=False) 
#     cv_r2_scores_rf = cross_val_score(Ridge_model, train_df_processed, total_df['log_price'][:train_rows], cv=5, scoring = rmsle_scorer) 
#     print(cv_r2_scores_rf) 
#     print("Mean 5-Fold R Squared: {}".format(np.mean(cv_r2_scores_rf)))

# def residual_plot(y_test,y_predicted_test):
#     #residual plot
#     x_plot = plt.scatter(y_predict_test, (y_test - y_predict_test), c='b')
#     plt.hlines(y=0, xmin= 0, xmax=50)
#     plt.title('Residual plot')


if __name__ == "__main__":
    train_df = pd.read_csv("../input/mercari/train.tsv",delimiter='\t',encoding='utf-8')
    test_df = pd.read_csv("../input/mercari/test.tsv",delimiter='\t',encoding='utf-8')
    train_rows = train_df.shape[0]

    test_submission = test_df['id']
    
    train_df.drop(['train_id'], axis =1, inplace = True)
    test_df.drop(['id'], axis = 1, inplace = True)
    
    price_zero_rows = train_df[train_df['price'] == 0].index
    train_df = train_df.drop(price_zero_rows)
    test_df['price'] = 0
    
    total_df = pd.concat([train_df, test_df], ignore_index = True)

    preprocessing(total_df)
    total_features = features(total_df,2)
    model(total_features,total_df['log_price'],train_rows)
#     submission(total_features,total_df['log_price'],train_rows,test_submission)
