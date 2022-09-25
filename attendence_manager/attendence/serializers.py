from rest_framework import serializers
from .models  import Attendence


class AttendenceSerializer(serializers.ModelSerializer):
      class Meta:
            model = Attendence
            # fields = ['id','title', 'author', 'email', 'date']
            fields = ['id','userId','name', 'moduleCode', 'moduleName', 'date', 'attendence']


