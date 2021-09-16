from collections import Counter
import nltk
from nltk.tokenize import TreebankWordTokenizer
import math


def cosine_sim(vec1, vec2):
    vec1 = [val for val in vec1.values()]
    vec2 = [val for val in vec2.values()]

    dot_prod = 0
    for i, v in enumerate(vec1):
        dot_prod += v * vec2[i]

    mag_1 = math.sqrt(sum([x**2 for x in vec1]))
    mag_2 = math.sqrt(sum([x**2 for x in vec2]))

    return dot_prod / (mag_1 * mag_2)


tokenizer = TreebankWordTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
fp = open('kite_text.txt', 'r')
kite_text = fp.read()

tokens = tokenizer.tokenize(kite_text.lower())
token_counts = Counter(tokens)
tokens = [x for x in tokens if x not in stop_words]
kite_counts = Counter(tokens)

document_vector = []
doc_length = len(tokens)
for key, value in kite_counts.most_common():
    document_vector.append(value / doc_length)
# print(document_vector)
docs = ["The faster Harry go to the store, the faster and faster Harry would get home."]
docs.append("Harry is hairy and faster than Jill.")
docs.append("Jill is not as hairy as Harry.")
doc_tokens = []
for doc in docs:
    doc_tokens += [sorted(tokenizer.tokenize(doc.lower()))]
# print(len(doc_tokens))
all_doc_tokens = sum(doc_tokens, [])
# print(len(all_doc_tokens))
lexion = sorted(set(all_doc_tokens))
# print(lexion)

from collections import OrderedDict
zero_vector = OrderedDict((token, 0) for token in lexion)

import copy
doc_vectors = []
for doc in docs:
    vec = copy.copy(zero_vector)
    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)
    for key, value in token_counts.items():
        vec[key] = value / len(lexion)
    doc_vectors.append(vec)
# print(doc_vectors)
document_tfidf_vectors = []
for doc in docs:
    vec = copy.copy(zero_vector)
    tokens = tokenizer.tokenize(doc.lower())
    token_counts = Counter(tokens)

    for key, value in token_counts.items():
        docs_containing_key = 0
        for _doc in docs:
            if key in _doc:
                docs_containing_key += 1
        tf = value / len(lexion)
        if docs_containing_key:
            idf = len(docs) / docs_containing_key
        else:
            idf = 0
        vec[key] = tf * idf
    document_tfidf_vectors.append(vec)
# print(document_tfidf_vectors)
query = "How long does it take to get to the store?"
query_vec = copy.copy(zero_vector)

tokens = tokenizer.tokenize(query.lower())
token_counts = Counter(tokens)
for key, value in token_counts.items():
    docs_containing_key = 0
    for _doc in docs:
        if key in _doc.lower():
            docs_containing_key += 1
    if docs_containing_key == 0:
        continue
    tf = value / len(tokens)
    idf = len(docs) / docs_containing_key
    query_vec[key] = tf * idf

a = cosine_sim(query_vec, document_tfidf_vectors[0])
b = cosine_sim(query_vec, document_tfidf_vectors[1])
c = cosine_sim(query_vec, document_tfidf_vectors[2])
# print(a, b, c)

# ChatBot
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = docs
vectorizer = TfidfVectorizer(min_df=1)
model = vectorizer.fit_transform(corpus)
print(model.todense().round(2))