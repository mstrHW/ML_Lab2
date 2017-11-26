import codecs
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
import numpy as np
from sklearn.model_selection import cross_val_score

###===1==###
file = codecs.open("SMSSpamCollection.txt", 'r', "UTF-8")

###===2===###
language = 'english'
stemmer = SnowballStemmer(language)
stopwords = nltk.corpus.stopwords.words(language)

X = []
Y = []

for line in file:

    y = line[:4]
    x = line[4:]

    lowercase_line = x.lower()
    nopunct_line = ''.join(re.findall(r'[a-z ]', lowercase_line))
    words = nopunct_line.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    result_words = [word for word in stemmed_words if word not in stopwords]
    X.append(' '.join(result_words))

    y = ''.join(re.findall(r'[a-z]', y))
    Y.append(y)

labels = np.array(Y) == 'spam'

###===3===###
vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(X).todense()
print(bag_of_words.shape)
print(vectorizer.vocabulary_)

###===4===###
cross_val_n = 10
gnb = GaussianNB()
scores = cross_val_score(gnb, bag_of_words, labels, cv=cross_val_n)
mean = scores.mean()
print('Cross_val : {}'.format(mean))

mnb = MultinomialNB()
scores = cross_val_score(mnb, bag_of_words, labels, cv=cross_val_n)
mean = scores.mean()
print('Cross_val : {}'.format(mean))

bnb = BernoulliNB()
scores = cross_val_score(bnb, bag_of_words, labels, cv=cross_val_n)
mean = scores.mean()
print('Cross_val : {}'.format(mean))


print('Ok')

