import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import wordnet
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk import pos_tag
from sklearn.metrics import pairwise_distances
from nltk import word_tokenize
from nltk.corpus import stopwords
from django.shortcuts import render
from django.views import View
from django.http import HttpResponse
# Create your views here.



def text_normalization(text):
        text = str(text).lower()
        spl_char_text = re.sub(r'[^ a-z]', '', text)
        tokens = nltk.word_tokenize(spl_char_text)
        lema = wordnet.WordNetLemmatizer()
        tags_list = pos_tag(tokens, tagset=None)
        lema_words = []  # empty list
        for token, pos_token in tags_list:
            if pos_token.startswith('V'):
                pos_val = 'v'
            elif pos_token.startswith('J'):
                pos_val = 'a'
            elif pos_token.startswith('R'):
                pos_val = 'r'
            else:
                pos_val = 'n'
            lema_token = lema.lemmatize(token, pos_val)
            lema_words.append(lema_token)

        return " ".join(lema_words)

def stopword_(text):
        tag_list = pos_tag(nltk.word_tokenize(text), tagset=None)
        stop = stopwords.words('english')
        lema = wordnet.WordNetLemmatizer()
        lema_word = []
        for token, pos_token in tag_list:
            if token in stop:
                continue
            if pos_token.startswith('V'):
                pos_val = 'v'
            elif pos_token.startswith('J'):
                pos_val = 'a'
            elif pos_token.startswith('R'):
                pos_val = 'r'
            else:
                pos_val = 'n'
            lema_token = lema.lemmatize(token, pos_val)
            lema_word.append(lema_token)
        return " ".join(lema_word)

class Chatbot(View):

    df=pd.read_excel('chat/dataset/dialog_talk_agent.xlsx')
    df.ffill(axis=0, inplace=True)
    df['lemmatiz']=df['Context'].apply(lambda  x : text_normalization(x))
    tf=TfidfVectorizer()
    x_tf=tf.fit_transform(df['lemmatiz']).toarray()
    df_tf=pd.DataFrame(x_tf,columns=tf.get_feature_names())
    print(df_tf.head())
    def get(self,request):
        return render(request,'chat.html')
    def post(self,request):
        msg=request.POST.get('msg')
        lma=text_normalization(msg)
        x_tf = self.tf.transform([lma])
        cos=1-pairwise_distances(self.df_tf,x_tf,metric='cosine')
        index=cos.argmax()
        return HttpResponse(self.df['Text Response'].loc[index])

