from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import casual_tokenize
import numpy as np
import pandas as pd
np.random.seed(42)
pd.set_option('display.width', 75)

counter = CountVectorizer(tokenizer=casual_tokenize)
sms = pd.read_csv('sms-spam.csv', index_col=[0])
index = ['sms{}{}'.format(i, '!'*j) for (i, j) in zip(range(len(sms)), sms.spam)]
sms = pd.DataFrame(sms.values, columns=sms.columns, index=index)
sms['spam'] = sms.spam.astype(int)

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize.casual import casual_tokenize
tfidf = TfidfVectorizer(tokenizer=casual_tokenize)
tfidf_docs = tfidf.fit_transform(raw_documents=sms.text).toarray()
tfidf_docs = pd.DataFrame(tfidf_docs)
tfidf_docs = tfidf_docs - tfidf_docs.mean()

bow_docs = pd.DataFrame(counter.fit_transform(raw_documents=sms.text).toarray(), index=index)
column_nums, terms = zip(*sorted(zip(tfidf.vocabulary_.values(), tfidf.vocabulary_.keys())))
bow_docs.columns = terms
# print(bow_docs.loc['sms0'][bow_docs.loc['sms0'] > 0])

from sklearn.decomposition import LatentDirichletAllocation as LDiA
ldia = LDiA(n_components=16, learning_method='batch')
ldia = ldia.fit(bow_docs)
columns = ['topic{}'.format(i) for i in range(ldia.n_components)]
components = pd.DataFrame(ldia.components_.T, index=terms, columns=columns)
# print(components['topic3'].round(2).head(3))
ldia16_topic_vectors = ldia.transform(bow_docs)
ldia16_topic_vectors = pd.DataFrame(ldia16_topic_vectors, index=index, columns=columns)
# print(ldia16_topic_vectors.round(2))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(ldia16_topic_vectors, sms.spam, test_size=0.5, random_state=271828)
lda = LDA(n_components=1)
lda = lda.fit(X_train, y_train)
sms['ldia16_spam'] = lda.predict(ldia16_topic_vectors)
acc = round(float(lda.score(X_test, y_test)), 2)
print(sms.ldia16_spam)
print(acc)
