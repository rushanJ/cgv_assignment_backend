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


def removeSpecialChars(string):

    string= re.sub(r'^.*?\n', '', string)
    string=string.split("\n")
    return string[0]

class FileAPIView(generics.GenericAPIView, mixins.ListModelMixin, mixins.CreateModelMixin, mixins.UpdateModelMixin,
                  mixins.DestroyModelMixin, mixins.RetrieveModelMixin):
    serializer_class = AttendenceSerializer
    queryset = Attendence.objects.all()



    def get(self, request, *args, **kwargs):
        string="a a) Se ma\nW Shashini Minosha De Silva\n|) $5 ooo"
        s=removeSpecialChars(string)
        print( )
        return self.list(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        return self.create(request, *args, **kwargs)

