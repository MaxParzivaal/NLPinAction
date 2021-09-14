import pandas as pd
from nltk.tokenize import casual_tokenize
from collections import Counter
from sklearn.naive_bayes import MultinomialNB

pd.set_option('display.width', 75)
movies = pd.read_table('hutto_movies.txt', names=['id', 'sentiment', 'text'], index_col='id')
bags_of_words = []

for text in movies.text:
    bags_of_words.append(Counter(casual_tokenize(text)))
df_bows = pd.DataFrame.from_records(bags_of_words)
df_bows = df_bows.fillna(0).astype(int)

products = pd.read_table('hutto_products.txt', names=['id', 'sentiment', 'text'])
# print(df_bows.shape)
# print(df_bows.head())
# print(df_bows.head()[list(bags_of_words[0].keys())])
nb = MultinomialNB()
nb = nb.fit(df_bows, movies.sentiment > 0)
# movies['predicted_sentiment'] = ''
a = nb.predict_proba(df_bows)
b = []
for i in a:
    x, y = i[0], i[1]
    tmp = x*(-8)+4 if x>y else y*8-4
    b.append(tmp)
# print(b)
# movies['predicted_sentiment'] = nb.predict_proba(df_bows) * 8 - 4
movies['predicted_sentiment'] = b
movies['error'] = (movies.predicted_sentiment - movies.sentiment).abs()
movies['sentiment_ispostive'] = (movies.sentiment > 0).astype(int)
movies['predicted_ispositiv'] = (movies.predicted_sentiment > 0).astype(int)
# del movies['text']
# movies.to_csv('./output.csv', sep='\t', encoding='utf-8')
bags_of_words = []
for text in products.text:
    bags_of_words.append(Counter(casual_tokenize(text)))
df_product_bows = pd.DataFrame.from_records(bags_of_words)
df_product_bows = df_product_bows.fillna(0).astype(int)
df_all_bows = df_bows.append(df_product_bows)
# print(df_all_bows.shape)
df_product_bows = df_all_bows.iloc[len(movies):][df_bows.columns]
# print(df_product_bows.shape)
df_product_bows = df_product_bows.fillna(0).astype(int)
products['ispos'] = (products.sentiment > 0).astype(int)
products['predicted_ispos'] = nb.predict(df_product_bows.values).astype(int)
del products['text']
products.to_csv('./output.csv', sep='\t', encoding='utf-8')
acc = (products.predicted_ispos == products.ispos).sum() / len(products)
print('acc:', acc)