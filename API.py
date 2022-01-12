import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.utils import resample

from bs4 import BeautifulSoup
import requests

import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop = stopwords.words('english')
new_stopping_words=stop[:len(stop)-36]
new_stopping_words.remove("not")


class GBPrediction:
    def __init__(self):
        self.clf  = RandomForestClassifier()
        print("Data init Done")


    def resample(self,df):
        #create two different dataframe of majority and minority class 
        df_majority = df[(df['Recommended IND']==1)] 
        df_minority = df[(df['Recommended IND']==0)] 

        # upsample minority class
        df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples=15539 , # to match majority class
                                 random_state=42)  # reproducible results
                                 
        # Combine majority class with upsampled minority class
        df_upsampled = pd.concat([df_minority_upsampled, df_majority])
        return df_upsampled
    
    def get_data(self):
        self.df = pd.read_csv('Reviews.csv')
        columns = ['Review Text','Recommended IND']
        self.df = self.df[columns]
        #resample
        self.df = self.resample(self.df)
        print("Data Importing Done")
    

    def remove_punc(self,df):
        new_text= re.sub("n't",' not', df)
        new_text= re.sub('[^\w\s]','', new_text)
        return new_text

    def tokenizze(self,df):
        newdata= word_tokenize(df)
        return newdata

    def remove_num(self,df):
        text_without_num=[w for w in df if w.isalpha()]
        return text_without_num

    def remove_stops(self,df):
        newdata=[t for t in df if t not in new_stopping_words ]
        return newdata

    def lemmatizze(self,df):
        newdata= [WordNetLemmatizer().lemmatize(t) for t in df]
        return newdata

    def Cleaning_process(self,data):
        processed_text=self.remove_punc(str(data))
        tokenized_data=self.tokenizze(processed_text.lower())
        textwithoutnum= self.remove_num(tokenized_data)
        data=self.remove_stops(textwithoutnum)
        final_data=self.lemmatizze(data)
        return " ".join(final_data)
    
    def vectorizer(self, data):
        data = self.tf_idf_vectorizer.transform(data)
        return data

    def preprocess(self):
        self.get_data()
        self.tokenizer = Tokenizer(num_words=1000)
        self.tf_idf_vectorizer = TfidfVectorizer()
        
        x = self.df["Review Text"].apply(self.Cleaning_process)
        y = self.df[['Recommended IND']]
        self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(x,y,test_size=0.2, stratify=y,random_state=101)
        
        self.tokenizer.fit_on_texts(x.values)
        self.tf_idf_vectorizer.fit_transform(self.x_train)        
        
        self.x_test_vec = self.vectorizer(self.x_test)
        self.x_train_vec = self.vectorizer(self.x_train)

        return "Data Preprocessing Done"

    def train(self):
        self.preprocess()
        self.clf.fit(self.x_train_vec, np.ravel(self.y_train))
        print("Model Training Done")
        
    def get_Revs(self,url,Headers):
        webpage = requests.get(url, headers=Headers)
        soup = BeautifulSoup(webpage.content, "html.parser")
        reviews=soup.find_all("div", {"data-hook":"review"})
        reviewlist=[]
        for rev in reviews:
            body=rev.find("span", {"data-hook": "review-body"})
            reviewlist.append(body.text)
        newlist=[]
        for rev in reviewlist:
            newdata=re.sub("The media could not be loaded.","",rev)
            newdata=re.sub("\n","",newdata)
            newlist.append(newdata)  
        return newlist
    
    def predict(self,test):
        t = self.Cleaning_process(test)
        t = self.vectorizer([t])
        label   = self.clf.predict(t)
        return label
    
    def dl_predict(self,review):
        self.model = load_model("./DL/bilstm2.h5")
        review = self.Cleaning_process(review)
        review = [review]
        tokens = self.tokenizer.texts_to_sequences(review)  
        tokens_pad = pad_sequences(tokens, maxlen=100)
        mod_pred = self.model.predict(tokens_pad)
        return mod_pred 


