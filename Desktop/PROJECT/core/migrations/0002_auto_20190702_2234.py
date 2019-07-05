# Generated by Django 2.1.1 on 2019-07-02 14:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='shows',
            name='author',
            field=models.CharField(default='', max_length=50),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='shows',
            name='address',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='shows',
            name='name',
            field=models.CharField(max_length=50),
        ),
        migrations.AlterField(
            model_name='shows',
            name='time',
            field=models.CharField(max_length=50),
        ),
    ]
