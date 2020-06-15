from rest_framework import generics, permissions, status, views
from sklearn.feature_extraction.text import CountVectorizer
from django.http import JsonResponse
from .apps import ApiConfig

from django.contrib.auth import authenticate
from django.views.decorators.csrf import csrf_exempt
from rest_framework.authtoken.models import Token
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny

class Predict(views.APIView):
    permission_classes = (permissions.IsAuthenticated, )
    def post(self, request):
        predictions = []

        query = request.data['query']
        

        try:
            result = list(ApiConfig.model.predict(ApiConfig.count_vect.transform([query])))
            predictions.append(result[0])
            result_ = "".join(str(x) for x in predictions)
        except Exception as err:
            return JsonResponse(str(err), status=status.HTTP_400_BAD_REQUEST, safe=False)

        response_dict = {query: result_}

        return JsonResponse(response_dict, status=status.HTTP_200_OK)


# LOGIN CLASS
@csrf_exempt
@api_view(["POST"])
@permission_classes((AllowAny,))
def login(request):
    username = request.data.get("username")
    password = request.data.get("password")

    if username is None or password is None:
        return JsonResponse({'error':'Please provide both username and password'},status=status.HTTP_400_BAD_REQUEST)
    user = authenticate(username=username, password=password)
    if not user:
        return JsonResponse({'error':'invalid credentials'}, status=status.HTTP_404_NOT_FOUND)
    token, _ = Token.objects.get_or_create(user=user)
    return JsonResponse({'token': token.key}, status=status.HTTP_200_OK)


        