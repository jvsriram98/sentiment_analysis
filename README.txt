# sentiment_analysis

assignment.py uses randomforest to classify. I have used randsomForest because dataset is small and has only two sentiments to classify. Using neural networks might not be 
working properly as NN needs larger dataset to avoid overfitting. 

Sentiment.py uses Bidirectional LSTM. This is a sequential model. Works better on larger and well-organized data. 

small part of EDA is done fig one shows one type analysis of data. We can observe that length of the sentences aren't very small to drop them.

In future, Naive Bayes or deep forest methods can be used to get better accuracy on this dataset. We can also use pretrained embedding like bert or roberta to train and then classify using logistic regression or svm.