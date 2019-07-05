from django.db import models


# Create your models here.
class shows(models.Model):
    name = models.CharField(max_length = 50)
    time = models.CharField(max_length = 50)
    address = models.CharField(max_length = 100)
    author = models.CharField(max_length = 50)
    average_list = models.CharField(max_length = 10000)
    average_number = models.IntegerField()


class activity(models.Model):
    name = models.CharField(max_length = 50)
    student = models.CharField(max_length = 50)
    state = models.CharField(max_length = 10000)
