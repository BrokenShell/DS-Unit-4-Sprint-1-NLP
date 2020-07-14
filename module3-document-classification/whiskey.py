"""
Hyper-parameter Tuning

Cross validation
    - Grad Student Descent
    - Grid Search
    - Random Search

"""
import pandas as pd
import numpy as np
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import spacy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
nlp = spacy.load("en_core_web_lg")


def create_model(depth, drop, activation, input_shape, output_size, n_layers):
    model = Sequential()
    model.add(Dense(depth, activation='tanh', input_shape=input_shape))
    model.add(Dropout(drop))
    for _ in range(n_layers):
        model.add(Dense(depth, activation=activation))
        model.add(Dropout(drop))
    if output_size == 1:
        model.add(Dense(output_size, activation='sigmoid'))
        model.compile(
            loss='binary_crossentropy',
            metrics=['accuracy'],
            optimizer='adam',
        )
    else:
        model.add(Dense(output_size, activation='softmax'))
        model.compile(
            loss='categorical_crossentropy',
            metrics=['accuracy'],
            optimizer='adam',
        )
    return model


def get_word_vectors(docs):
    return [nlp(doc).vector for doc in docs]


df = pd.read_csv('data/train.csv')
y = df['ratingCategory']
X = pd.DataFrame(get_word_vectors(df['description']))
X_test = pd.read_csv("data/test.csv")

classifier = KerasClassifier(
    build_fn=create_model,
    input_shape=X.columns.shape,
    output_size=3,
    verbose=1,
    batch_size=12,
    epochs=160,
    drop=0.15,
    activation='linear',
    depth=160,
    n_layers=5,
)

param_distributions = {
    # 'batch_size': range(2, 24, 2),
    # 'epochs': range(160, 250, 20),
    # 'drop': [0.1, 0.15, 0.2, 0.25, 0.3],
    # 'activation': ['tanh', 'relu', 'linear', 'softmax', 'softplus', 'softsign', 'sigmoid', 'hard_sigmoid'],
    # 'depth': range(32, 256, 32),
    # 'n_layers': range(1, 10),
}

search = RandomizedSearchCV(
    estimator=classifier,
    param_distributions=param_distributions,
    cv=3,
    n_iter=1,
    n_jobs=8,
    random_state=42,
)
search.fit(X, y)

print(search.best_score_)
print(search.best_params_)
pred = search.predict(pd.DataFrame(get_word_vectors(X_test['description'])))
submission = pd.DataFrame({'id': X_test['id'], 'ratingCategory': pred})
submission['ratingCategory'] = submission['ratingCategory'].astype('int64')
submission.to_csv('data/submission-keras3.csv', index=False)
