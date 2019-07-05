student_number = 30
video_length = 500

import random

def random_name():
    name = [chr(int(random.random()*26)+97) for i in range(3)] + ['_', str(int(random.random()*100))]
    return ''.join(name)

def random_state(length):
    a = []
    for i in range(length):
        a.append(int(random.random() * 300) % 3)
    return a

def random_dict(number, length):
    a = {}
    for i in range(number):
        a[random_name()] = random_state(length)
    return a

student_init = random_dict(30,500)

import os
path1 = 'static/data'
if not os.path.exists(path1):
    os.makedirs(path1)

f = open(path1 + '/random_data.txt','w')
f.write(str(student_init))