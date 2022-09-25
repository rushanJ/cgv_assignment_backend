import os

from django.http import HttpResponse
from rest_framework import parsers

from .models import Attendence
from .serializers import AttendenceSerializer
from rest_framework import generics
from rest_framework import mixins
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import filters
import re
import django_filters.rest_framework
from django_filters.rest_framework import DjangoFilterBackend

def removeSpecialChars(string):

    string= re.sub(r'^.*?\n', '', string)
    string=string.split("\n")
    return string[0]

class FileAPIView(generics.GenericAPIView, mixins.ListModelMixin, mixins.CreateModelMixin, mixins.UpdateModelMixin,
                  mixins.DestroyModelMixin, mixins.RetrieveModelMixin):
    queryset = Attendence.objects.all()
    serializer_class = AttendenceSerializer

    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['moduleName', 'date']


    def get(self, request, *args, **kwargs):

        return self.list(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        return self.create(request, *args, **kwargs)

