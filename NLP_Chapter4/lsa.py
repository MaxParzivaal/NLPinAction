import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

VOCABULARY = vocabulary='cat dog apple lion NYC love'.lower().split()
DOCS = open('cats_and_dogs_sorted.txt', 'r', encoding='utf-8').read().splitlines()[:11]
# DOCS = list(DOCS)


def docs_to_tdm(docs=DOCS, vocabulary=VOCABULARY, verbosity=0):
    tfidfer = TfidfVectorizer(min_df=1, max_df=.99, stop_words=None, token_pattern=r'(?u)\b\w+\b',
                              vocabulary=vocabulary)
    tfidf_dense = pd.DataFrame(tfidfer.fit_transform(docs).todense())
    id_words = [(i, w) for (w, i) in tfidfer.vocabulary_.items()]
    tfidf_dense.columns = list(zip(*sorted(id_words)))[1]

    tfidfer.use_idf = False
    tfidfer.norm = None
    bow_dense = pd.DataFrame(tfidfer.fit_transform(docs).todense())
    bow_dense.columns = list(zip(*sorted(id_words)))[1]
    bow_dense = bow_dense.astype(int)
    tfidfer.use_idf = True
    tfidfer.norm = 'l2'
    if verbosity:
        print(tfidf_dense.T)
    return bow_dense.T, tfidf_dense.T, tfidfer


def lsa(tdm, verbosity=0):
    if verbosity:
        print(tdm)
        #         0   1   2   3   4   5   6   7   8   9   10
        # cat     0   0   0   0   0   0   1   1   1   0   1
        # dog     0   0   0   0   0   0   0   0   0   0   1
        # apple   1   1   0   1   1   1   0   0   0   0   0
        # lion    0   0   0   0   0   0   0   1   0   0   0
        # love    0   0   1   0   0   0   0   0   1   1   0
        # nyc     1   1   1   1   1   0   0   0   0   1   0

    u, s, vt = np.linalg.svd(tdm)

    u = pd.DataFrame(u, index=tdm.index)
    if verbosity:
        print('U')
        print(u.round(2))
        # U
        #           0     1     2     3     4     5
        # cat   -0.04  0.83 -0.38 -0.00  0.11  0.38
        # dog   -0.00  0.21 -0.18 -0.71 -0.39 -0.52
        # apple -0.62 -0.21 -0.51  0.00  0.49 -0.27
        # lion  -0.00  0.21 -0.18  0.71 -0.39 -0.52
        # love  -0.22  0.42  0.69  0.00  0.41 -0.37
        # nyc   -0.75  0.00  0.24 -0.00 -0.52  0.32

    vt = pd.DataFrame(vt, index=['d{}'.format(i) for i in range(len(vt))])
    if verbosity:
        print('VT')
        print(vt.round(2))
        # VT
        #        0     1     2     3     4     5     6     7     8     9     10
        # d0  -0.44 -0.44 -0.31 -0.44 -0.44 -0.20 -0.01 -0.01 -0.08 -0.31 -0.01
        # d1  -0.09 -0.09  0.19 -0.09 -0.09 -0.09  0.37  0.47  0.56  0.19  0.47
        # d2  -0.16 -0.16  0.52 -0.16 -0.16 -0.29 -0.22 -0.32  0.17  0.52 -0.32
        # d3   0.00 -0.00  0.00  0.00  0.00  0.00 -0.00  0.71  0.00  0.00 -0.71
        # d4  -0.04 -0.04 -0.14 -0.04 -0.04  0.58  0.13 -0.33  0.62 -0.14 -0.33
        # d5   0.09  0.09 -0.10  0.09  0.09 -0.51  0.73 -0.27  0.01 -0.10 -0.27
        # d6  -0.55  0.24  0.15  0.36 -0.38  0.32  0.32  0.00 -0.32  0.17  0.00
        # d7  -0.32  0.46  0.23 -0.64  0.41  0.09  0.09  0.00 -0.09 -0.14  0.00
        # d8  -0.52  0.27 -0.24  0.39  0.22 -0.36 -0.36 -0.00  0.36 -0.12  0.00
        # d9  -0.14 -0.14 -0.58 -0.14  0.32  0.10  0.10 -0.00 -0.10  0.68 -0.00
        # d10 -0.27 -0.63  0.31  0.23  0.55  0.12  0.12 -0.00 -0.12 -0.19 -0.00

    # Reconstruct the original term-document matrix.
    # The sum of the squares of the error is 0.

    return {'u': u, 's': s, 'vt': vt, 'tdm': tdm}


def accuracy_study(tdm=None, u=None, s=None, vt=None, verbosity=0, **kwargs):
    """ Reconstruct the term-document matrix and measure error as SVD terms are truncated
    """
    smat = np.zeros((len(u), len(vt)))
    np.fill_diagonal(smat, s)
    smat = pd.DataFrame(smat, columns=vt.index, index=u.index)
    if verbosity:
        print()
        print('Sigma:')
        print(smat.round(2))
        print()
        print('Sigma without zeroing any dim:')
        print(np.diag(smat.round(2)))
    tdm_prime = u.values.dot(smat.values).dot(vt.values)
    if verbosity:
        print()
        print('Reconstructed Term-Document Matrix')
        print(tdm_prime.round(2))

    err = [np.sqrt(((tdm_prime - tdm).values.flatten() ** 2).sum() / np.product(tdm.shape))]
    if verbosity:
        print()
        print('Error without reducing dimensions:')
        print(err[-1])
    # 2.3481474529927113e-15

    smat2 = smat.copy()
    for numdim in range(len(s) - 1, 0, -1):
        smat2.iloc[numdim, numdim] = 0
        if verbosity:
            print('Sigma after zeroing out dim {}'.format(numdim))
            print(np.diag(smat2.round(2)))
            #           d0    d1   d2   d3   d4   d5
            # ship    2.16  0.00  0.0  0.0  0.0  0.0
            # boat    0.00  1.59  0.0  0.0  0.0  0.0
            # ocean   0.00  0.00  0.0  0.0  0.0  0.0
            # voyage  0.00  0.00  0.0  0.0  0.0  0.0
            # trip    0.00  0.00  0.0  0.0  0.0  0.0

        tdm_prime2 = u.values.dot(smat2.values).dot(vt.values)
        err += [np.sqrt(((tdm_prime2 - tdm).values.flatten() ** 2).sum() / np.product(tdm.shape))]
        if verbosity:
            print('Error after zeroing out dim {}'.format(numdim))
            print(err[-1])
    return err


def lsa_models(vocabulary='cat dog apple lion NYC love'.lower().split(), docs=11, verbosity=0):
    # vocabulary = 'cat dog apple lion NYC love big small bright'.lower().split()
    if isinstance(docs, int):
        docs = open('cats_and_dogs_sorted.txt', 'r', encoding='utf-8').read().splitlines()[:11]
    tdm, tfidfdm, tfidfer = docs_to_tdm(docs=docs, vocabulary=vocabulary)
    lsa_bow_model = lsa(tdm)  # (tdm - tdm.mean(axis=1)) # SVD fails to converge if you center, like PCA does
    lsa_bow_model['vocabulary'] = tdm.index.values
    lsa_bow_model['docs'] = docs
    err = accuracy_study(verbosity=verbosity, **lsa_bow_model)
    lsa_bow_model['err'] = err
    lsa_bow_model['accuracy'] = list(1. - np.array(err))

    lsa_tfidf_model = lsa(tdm=tfidfdm)
    lsa_bow_model['vocabulary'] = tfidfdm.index.values
    lsa_tfidf_model['docs'] = docs
    err = accuracy_study(verbosity=verbosity, **lsa_tfidf_model)
    lsa_tfidf_model['err'] = err
    lsa_tfidf_model['accuracy'] = list(1. - np.array(err))

    return lsa_bow_model, lsa_tfidf_model


def prettify_tdm(tdm=None, docs=[], vocabulary=[], **kwargs):
    bow_pretty = tdm.T.copy()[vocabulary]
    bow_pretty['text'] = docs
    for col in vocabulary:
        bow_pretty.loc[bow_pretty[col] == 0, col] = ''
    return bow_pretty


bow_svd, tfidf_svd = lsa_models()
prettify_tdm(**bow_svd)
tdm = bow_svd['tdm']
# print(tdm)

U, s, Vt = np.linalg.svd(tdm)
tmp = pd.DataFrame(U, index=tdm.index).round(2)
# print(tmp)

s.round(1)
S = np.zeros((len(U), len(Vt)))
np.fill_diagonal(S, s)
tmp_2 = pd.DataFrame(S).round(1)
# print(tmp_2)

tmp_3 = pd.DataFrame(Vt).round(2)
# print(tmp_3)

err = []
for numdim in range(len(s), 0, -1):
    S[numdim - 1, numdim - 1] = 0
    reconstructed_tdm = U.dot(S).dot(Vt)
    err.append(np.sqrt(((reconstructed_tdm - tdm).values.flatten() ** 2).sum() / np.product(tdm.shape)))
print(np.array(err).round(2))
