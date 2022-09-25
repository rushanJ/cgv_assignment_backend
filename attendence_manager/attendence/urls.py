from django.urls import path
from .views import  FileAPIView

urlpatterns = [

    # path('generic/user/', UserAPIView.as_view()),

    path('', FileAPIView.as_view()),


]
