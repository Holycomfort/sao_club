from django import forms


class AddForm(forms.Form):
    name = forms.CharField()
    time = forms.CharField()
    address = forms.CharField()
    author = forms.CharField()
    photos = forms.FileField()
