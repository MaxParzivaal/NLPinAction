from collections import Counter
import nltk
from nltk.tokenize import TreebankWordTokenizer

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
print(doc_vectors)