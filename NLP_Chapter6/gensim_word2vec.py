# The URL of DataSet:
# https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz

from gensim.models.keyedvectors import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True, limit=200000)

# 给出相关单词
sim1 = word_vectors.most_similar(positive=['cooking', 'potatoes'], topn=5)
sim2 = word_vectors.most_similar(positive=['canada', 'america'], topn=1)
print(sim1)
print(sim2)

# 检测不相关单词
doesnt_word = word_vectors.doesnt_match("laptop phone milk computer".split())
print(doesnt_word)

# king + woman - man = queen
oper_1 = word_vectors.most_similar(positive=['king', 'woman'], negative=['man'], topn=2)
print(oper_1)

# 查看两个单词相似度
words_sim = word_vectors.similarity('princess', 'queen')
print(words_sim)

print(word_vectors['phone'].shape)
