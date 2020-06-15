from django.urls import path
from .views import Predict, login
from rest_framework.authtoken.views import obtain_auth_token

# USING REGULARS URLS
app_name = 'api'

urlpatterns = [
    # path('train/', Train.as_view(), name='train'), # new
    path('predict/', Predict.as_view(), name='predict'),
    path('api-token-auth/', obtain_auth_token, name='api_token_auth'),
    path('login/', login),
]