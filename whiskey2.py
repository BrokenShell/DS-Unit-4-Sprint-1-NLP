import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
import spacy

nlp = spacy.load("en_core_web_lg")


def get_word_vectors(docs):
    return [nlp(doc).vector for doc in docs]


df = pd.read_csv('data/train.csv')
y1 = df['ratingCategory']
x1 = pd.DataFrame(get_word_vectors(df['description']))
test = pd.read_csv("data/test.csv")


gnb = GaussianNB()
KNN = KNeighborsClassifier(n_neighbors=1)
MNB = MultinomialNB()
BNB = BernoulliNB()
LR = LogisticRegression()
SDG = SGDClassifier()
SVC = SVC()
LSVC = LinearSVC()


BNB.fit(x1, y1)
y2_BNB_model = BNB.predict(pd.DataFrame(get_word_vectors(test['description'])))
submission = pd.DataFrame({'id': test['id'], 'ratingCategory': y2_BNB_model})
submission['ratingCategory'] = submission['ratingCategory'].astype('int64')
submission.to_csv('data/submission-y2_BNB_model.csv', index=False)

gnb.fit(x1, y1)
y2_GNB_model = gnb.predict(pd.DataFrame(get_word_vectors(test['description'])))
submission = pd.DataFrame({'id': test['id'], 'ratingCategory': y2_GNB_model})
submission['ratingCategory'] = submission['ratingCategory'].astype('int64')
submission.to_csv('data/submission-y2_GNB_model.csv', index=False)

KNN.fit(x1, y1)
y2_KNN_model = KNN.predict(pd.DataFrame(get_word_vectors(test['description'])))
submission = pd.DataFrame({'id': test['id'], 'ratingCategory': y2_KNN_model})
submission['ratingCategory'] = submission['ratingCategory'].astype('int64')
submission.to_csv('data/submission-y2_KNN_model.csv', index=False)

LR.fit(x1, y1)
y2_LR_model = LR.predict(pd.DataFrame(get_word_vectors(test['description'])))
submission = pd.DataFrame({'id': test['id'], 'ratingCategory': y2_LR_model})
submission['ratingCategory'] = submission['ratingCategory'].astype('int64')
submission.to_csv('data/submission-y2_LR_model.csv', index=False)

LSVC.fit(x1, y1)
y2_LSVC_model = LSVC.predict(pd.DataFrame(get_word_vectors(test['description'])))
submission = pd.DataFrame({'id': test['id'], 'ratingCategory': y2_LSVC_model})
submission['ratingCategory'] = submission['ratingCategory'].astype('int64')
submission.to_csv('data/submission-y2_LSVC_model.csv', index=False)

SDG.fit(x1, y1)
y2_SDG_model = SDG.predict(pd.DataFrame(get_word_vectors(test['description'])))
submission = pd.DataFrame({'id': test['id'], 'ratingCategory': y2_SDG_model})
submission['ratingCategory'] = submission['ratingCategory'].astype('int64')
submission.to_csv('data/submission-y2_SDG_model.csv', index=False)

SVC.fit(x1, y1)
y2_SVC_model = SVC.predict(pd.DataFrame(get_word_vectors(test['description'])))
submission = pd.DataFrame({'id': test['id'], 'ratingCategory': y2_SVC_model})
submission['ratingCategory'] = submission['ratingCategory'].astype('int64')
submission.to_csv('data/submission-y2_SVC_model.csv', index=False)
