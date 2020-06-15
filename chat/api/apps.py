from django.apps import AppConfig
from rest_framework import generics, permissions, status, views
from rest_framework.response import Response
from django.conf import settings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import pickle
import datetime
from pathlib import Path
from django.http import JsonResponse

from django.contrib.auth import authenticate
from django.views.decorators.csrf import csrf_exempt
from rest_framework.authtoken.models import Token
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny


class ApiConfig(AppConfig):
    name = 'api'

    df = pd.read_excel(r'models/chat.xlsx')

    df['Response_id'] = df['Response'].factorize()[0]
    Response_id_df = df[['Response', 'Response_id']].drop_duplicates().sort_values('Response_id')
    Response_to_id = dict (Response_id_df.values)
    id_to_Response = dict(Response_id_df[['Response_id', 'Response']].values)

    tfidf = TfidfVectorizer(sublinear_tf=True, max_df=5, norm='l2', encoding='latin-1',
                            ngram_range=(2, 2), stop_words='english')
    features = tfidf.fit_transform(df['Message']).toarray()
    labels = df.Response_id
    Response_to_id.items()
    sorted(Response_to_id.items())

    N = 2
    for Response, Response_id in sorted(Response_to_id.items()):
        features_chi2 = chi2(features, labels == Response_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        # print("# '{}':".format(Response))
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]

    X_train, X_test, y_train, y_test = train_test_split(df['Message'], df['Response'], random_state=0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    mod = svm.LinearSVC()
    clf = mod.fit(X_train_tfidf, y_train)

    root = Path(".")
    my_path = root/"models"/"chat.pkl"

    with open(my_path, 'wb') as file:
        pickle.dump(clf, file)

    with open(my_path, 'rb') as file:
        model = pickle.load(file)



