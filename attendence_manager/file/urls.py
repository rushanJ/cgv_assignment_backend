from django.urls import path
from .views import  FileAPIView,ModulesAPIView,DatesAPIView

urlpatterns = [

    # path('generic/user/', UserAPIView.as_view()),

    path('', FileAPIView.as_view()),
    path('modules',ModulesAPIView.as_view()),
    path('dates', DatesAPIView.as_view()),


]
