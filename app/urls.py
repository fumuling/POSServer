from django.urls import path
from .views import pos_list

app_name = 'app'
urlpatterns = [
    path('app/', pos_list, name = 'app'),
]