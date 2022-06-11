import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist
from wordcloud import WordCloud
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

#print(data.text)
from fast_ml.model_development import train_valid_test_split

x_train, y_train, x_valid, y_valid, x_test, y_test = train_valid_test_split(data, target = 'sentiment', 
                                                                            method='random',
                                                                            train_size=0.8, valid_size=0.1, test_size=0.1)

x_train = x_train['text']
x_valid = x_valid['text']
x_test = x_test['text']


#EDA is done we can see neutral words and negative words
plt.figure(figsize=(20,20))
pos_freq = FreqDist(' '.join(data[data['sentiment'] == 1].text).split(' '))
wc = WordCloud().generate_from_frequencies(frequencies=pos_freq)
plt.imshow(wc,interpolation='bilinear')
plt.title('Neutral Common Text')
plt.axis('off')
plt.show()
plt.figure(figsize=(20,6))
pos_freq.plot(50,cumulative=False,title='Neutral Common Text')
plt.show()

plt.figure(figsize=(20,20))
neg_freq = FreqDist(' '.join(data[data['sentiment'] == 0].text).split(' '))
wc = WordCloud().generate_from_frequencies(frequencies=neg_freq)
plt.imshow(wc,interpolation='bilinear')
plt.title('Negative Common Text')
plt.axis('off')
plt.show()

plt.figure(figsize=(20,6))
neg_freq.plot(50,cumulative=False,title='Negative Common Text',color='red')
plt.show()


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
#Tokenize the sentences
tokenizer = Tokenizer()
#preparing vocabulary
tokenizer.fit_on_texts(x_train)
#converting text into integer sequences
x_train = tokenizer.texts_to_sequences(x_train)
x_test = tokenizer.texts_to_sequences(x_test)
x_train=pad_sequences(x_train,maxlen=120)
x_test=pad_sequences(x_test,maxlen=120)
size_of_vocabulary = len(tokenizer.word_index)+1

#print(x_train,y_train,x_valid,y_valid)

from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
#embedding layer
model.add(Embedding(size_of_vocabulary,128,input_length=120))
#lstm layer
model.add(Bidirectional(LSTM(64,return_sequences=True,dropout=0.2)))
#Global Maxpooling
model.add(GlobalMaxPooling1D())
#Dense Layer
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.05))
model.add(Dense(1,activation='sigmoid'))
#Add loss function, metrics, optimizer
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#Adding callbacks
es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=3)
mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True,verbose=1)
#summary
model.summary()

history = model.fit(x_train,y_train,batch_size=128,epochs=4,
                    validation_data=(x_valid,y_valid),verbose=1,callbacks=[es,mc])

model = load_model('best_model.h5')
# evaluate
loss,acc = model.evaluate(x_test,y_test)
print('Test Accuracy: {}%'.format(acc*100))
ypred = model.predict(x_test)
ypred[ypred>0.5]=1
ypred[ypred<=0.5]=0
#Confusion Metrics
print(confusion_matrix(y_test,ypred))
print(classification_report(y_test,ypred))

