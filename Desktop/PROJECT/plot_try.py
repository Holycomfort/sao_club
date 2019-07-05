merge = 25

import os
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签

path1 = 'static/data'
f = open(path1 + '/random_data.txt','r')
student_init = eval(f.read())
if type(student_init) is dict:
    student_name = list(student_init.keys())
    student_init = list(student_init.values())

student_merge = []
frame_real = int(len(student_init[0])/merge)

student_number = len(student_init)
video_length = len(student_init[0])

for k in range(len(student_init)):
    one = []
    for i in range(frame_real):
        plus = sum(student_init[k][i*merge:(i+1)*merge])
        one.append(plus)
    student_merge.append(one)

average = []
for i in range(frame_real):
    sum = 0
    for student in student_merge:
        sum += student[i]
    sum /= len(student_merge)
    average.append(sum)
print(average)

path1 = 'static/figure/plot/act1'
if not os.path.exists(path1):
    os.makedirs(path1)

plt.xlabel('时间')
plt.ylabel('平均专注度')
plt.plot(average, color = 'orange')
plt.yticks(list(range(merge-10,merge+10,4)))
plt.savefig(path1 + '/average.png')

for i in range(student_number):
    print(i)
    plt.cla()
    plt.plot(student_merge[i])
    plt.xlabel('时间')
    plt.ylabel('专注度')
    plt.yticks(list(range(0,2*merge,int(2*merge/10))))
    plt.savefig(path1 + '/' + str(student_name[i]) + '.png')


#拟合？
'''coef5 = np.polyfit(range(frame_real), student_merge[0], 15)
poly_fit5 = np.poly1d(coef5)
t = list(range(0, frame_real*10-20))
t = [i/10 for i in t]
plt.plot(t, poly_fit5(t), 'r',label="五阶拟合")
print(poly_fit5)'''
