import django
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "PROJECT.settings")
django.setup()
from core.models import *


for i in shows.objects.filter().values():
    print(i)
print('--------------------------')
for i in activity.objects.filter().values():
    print(i)

