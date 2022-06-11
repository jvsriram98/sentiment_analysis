import re
import numpy as np
import pandas as pd
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from textblob import Word
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score
import torch
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import StratifiedShuffleSplit
#from wordcloud import WordCloud
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc
#data=pd.read_csv('C:\Users\jvsri\OneDrive\Desktop\Basic_Sentiment_Classification\data.csv')
data=pd.read_csv('data.csv',encoding='ISO-8859-1')
#data = data[['text','sentiment']]
#print(data['sentiment'])
mapping = {'neutral': 1, 'negative': 0}
data.sentiment = data.sentiment.map(mapping)
   
#cleaning data
data['text'] = data['text'].apply(lambda x: str(x))

def Preprocessing(text):
    text = re.sub(r'[^\w\s]','',text)
    text = text.lower()
    text = [w for w in text.split(' ') if w not in stopwords.words('english')]
    text = [WordNetLemmatizer().lemmatize(token) for token in text]
    text = [WordNetLemmatizer().lemmatize(token,pos='v') for token in text]
    text = " ".join(text)
    return text

data['text'] = data.text.apply(lambda x:Preprocessing(x))
plt.figure(figsize=(16,20))
plt.style.use('fivethirtyeight')

plt.subplot(3,1,1)
train_len = [len(l) for l in data.text]
plt.hist(train_len,bins=50)
plt.title('Distribution of train text length')
plt.xlabel('Length')
plt.show()

#print(data.text)
txt=data.text.apply(gensim.utils.simple_preprocess)
print(txt)
rev=[0]*18899
for i in range(0,18899):
    str=" "
    re=txt[i]
    rev[i]=str.join(re)

cv1=CountVectorizer()
gen=cv1.fit_transform(rev)
#print(gen)
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(gen,data.sentiment,train_size=0.8)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=50)
rf.fit(X_train,y_train)

pre=rf.predict(X_test)

from sklearn.metrics import accuracy_score


predic=rf.predict(X_test)
print(accuracy_score(predic,y_test))

plt.figure(figsize=(16,8))
fpr,tpr,threshold = roc_curve(y_test,predic)
roc_auc = auc(fpr, tpr)
plt.title('ROC Curve')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

