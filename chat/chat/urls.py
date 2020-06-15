"""chat URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from rest_framework.documentation import include_docs_urls
from rest_framework.schemas import get_schema_view
from django.views.generic import TemplateView

API_TITLE = 'Chat API'
API_DESCRIPTION = 'An ML Chat'
schema_view = get_schema_view(title=API_TITLE)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/', include('api.urls')),
    path('api-auth/', include('rest_framework.urls')),
    path('api/v1/rest-auth/', include('rest_auth.urls')), # new
    path('api/v1/rest-auth/registration/', include('rest_auth.registration.urls')), #new
    path('openapi', get_schema_view(title=API_TITLE, description=API_DESCRIPTION), name='openapi-schema'),
    path('swagger-ui/', TemplateView.as_view(template_name='blog_project/swagger-ui.html',
    extra_context={'schema_url':'openapi-schema'}), name='swagger-ui'),
]
