from collections import Counter
import nltk
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()
stop_words = nltk.corpus.stopwords.words('english')
fp1 = open('kite_text.txt', 'r')
kite_text = fp1.read()
kite_intro = kite_text.lower()
intro_tokens = tokenizer.tokenize(kite_intro)
fp2 = open('kite_history.txt', 'r')
kite_history = fp2.read()
kite_history = kite_history.lower()
history_tokens = tokenizer.tokenize(kite_history)
intro_total = len(intro_tokens)
history_total = len(history_tokens)
# print(intro_total, history_total)

intro_tf = {}
history_tf = {}
intro_counts = Counter(intro_tokens)
intro_tf['kite'] = intro_counts['kite'] / intro_total
intro_tf['and'] = intro_counts['and'] / intro_total
history_counts = Counter(history_tokens)
history_tf['kite'] = history_counts['kite'] / history_total
history_tf['and'] = history_counts['and'] / history_total
# print('Term Frequency of "kite" in intro is: {:.4f}'.format(intro_tf['kite']))
# print('Term Frequency of "kite" in history is: {:.4f}'.format(history_tf['kite']))
# print('Term Frequency of "and" in intro is: {:.4f}'.format(intro_tf['and']))
# print('Term Frequency of "and" in history is: {:.4f}'.format(history_tf['and']))


def get_num_docs_containing(key_word):
    num_docs_containing_keyword = 0
    for doc in [intro_tokens, history_tokens]:
        if key_word in doc:
            num_docs_containing_keyword += 1
    return num_docs_containing_keyword


intro_tf['china'] = intro_counts['china'] / intro_total
history_tf['china'] = history_counts['china'] / history_total

num_docs = 2
intro_idf = {}
history_idf = {}
num_docs_containing_and = get_num_docs_containing('and')
num_docs_containing_kite = get_num_docs_containing('kite')
num_docs_containing_china = get_num_docs_containing('china')
intro_idf['and'] = num_docs / num_docs_containing_and
history_idf['and'] = num_docs / num_docs_containing_and
intro_idf['kite'] = num_docs / num_docs_containing_kite
history_idf['kite'] = num_docs / num_docs_containing_kite
intro_idf['china'] = num_docs / num_docs_containing_china
history_idf['china'] = num_docs / num_docs_containing_china

intro_tfidf = {}
intro_tfidf['and'] = intro_tf['and'] * intro_idf['and']
intro_tfidf['kite'] = intro_tf['kite'] * intro_idf['kite']
intro_tfidf['china'] = intro_tf['china'] * intro_idf['china']
history_tfidf = {}
history_tfidf['and'] = history_tf['and'] * history_idf['and']
history_tfidf['kite'] = history_tf['kite'] * history_idf['kite']
history_tfidf['china'] = history_tf['china'] * history_idf['china']

