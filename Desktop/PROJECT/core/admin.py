from django.contrib import admin

# Register your models here.

from core import models
admin.site.register(models.shows)
admin.site.register(models.activity)