from django.db import models

class Attendence(models.Model):
    # user = models.ForeignKey(User, related_name='files', on_delete=models.CASCADE)
    # user=models.CharField(("user"), max_length=100)
    # description=models.CharField(("description"), max_length=1000)
    userId = models.CharField(("userId"), max_length=100)
    name = models.TextField()
    moduleCode = models.CharField(("moduleCode"), max_length=20)
    moduleName = models.CharField(("moduleName"), max_length=20)
    date = models.CharField(("date"), max_length=20)
    attendence = models.BooleanField()

    # file = models.FileField(max_length=None, null = True)

    # date=models.DateTimeField(("date"), auto_now=False, auto_now_add=True)
    class Meta:
        db_table = 'Attendence'

    def __str__(self):
        return self.name
