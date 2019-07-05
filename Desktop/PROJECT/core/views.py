from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .forms import AddForm
from core import models
import cv2
from core.face import face_detector, substitute
import zipfile
import os
import matplotlib.pyplot as plt
# 用来正常显示中文标签
plt.rcParams['font.sans-serif'] = ['SimHei']

merge = 25
allface_dic = {}


# Create your views here.
def first_page(request):
    if request.method == 'POST':
        response = HttpResponseRedirect('./act1.html')
        name = request.POST.get('key')
        temp = models.shows.objects.filter(name=name).values()
        id = temp[0]['id']
        response.set_cookie('key', id)
        print(id)
        return response
    else:
        r = range(1, len(models.shows.objects.all())+1)
        return render(request, 'first_page.html', {'show_list': zip(r, models.shows.objects.all())})


def act1(request):
    feature = request.COOKIES.get('key')
    temp = models.shows.objects.filter(id=feature).values()
    name = temp[0]['name']

    student_name = []
    student_state = []
    for i in models.activity.objects.filter(name=name).values():
        student_name.append(i['student'])
        student_state.append(i['state'])
    return render(request, 'act1.html', {'stu_num': range(len(student_name)), 'name': name,
                            'student': zip(student_name, student_state)})


def add(request):
    if request.method == 'POST':
        form = AddForm(request.POST, request.FILES)
        if form.is_valid():
            # save
            name = form.cleaned_data['name']
            time = form.cleaned_data['time']
            address = form.cleaned_data['address']
            author = form.cleaned_data['author']
            print(1)
            photos = form.cleaned_data['photos']
            print(2)
            if photos.name.split('.')[1] != "zip":
                return HttpResponse("只能上传zip文件！")
            path = default_storage.save('static/figure/rawphoto/'+photos.name,
                                        ContentFile(photos.read()))
            print('3:', path)
            extracting = zipfile.ZipFile(path)
            extracting.extractall('static/figure/rawphoto/')
            os.remove(path)
            print(4)

            fm = 1
            while True:
                try:
                    frame = cv2.imread('static/figure/rawphoto/'+name
                                       +'/'+str(fm)+'.jpg')
                    outlist = face_detector(frame)
                    if fm is 1:
                        for i, f in enumerate(outlist):
                            allface_dic[str(i)] = f
                            allface_dic[str(i)].time = [f.station_analyze()]
                    else:
                        substitute(allface_dic, outlist, fm)
                    print([(id, face.position, face.frontal) for id, face in allface_dic.items()])

                    for id in allface_dic:
                        face = allface_dic[id].position
                        if face is not None:
                            cv2.rectangle(frame, (face[0], face[1]),
                                          (face[0] + face[2], face[1] + face[3]), (0, 255, 0), 2)
                    fm += 1
                except Exception as e:
                    print(repr(e))
                    break

            print(allface_dic)
            dic = {tid: tface.time for tid, tface in allface_dic.items()}
            for i, j in dic.items():
                models.activity.objects.create(name=name, student=i, state=j)

            # plot
            path0 = 'static/data'
            if not os.path.exists(path0):
                os.makedirs(path0)
            f = open('static/data/%s.txt' % name, 'w')
            f.write(str(dic))
            f.close()
            path1 = 'static/figure/plot/%s' % name
            if not os.path.exists(path1):
                os.makedirs(path1)

            student_name = list(dic.keys())
            student_state = list(dic.values())
            student_merge = []
            frame_real = int(len(student_state[0]) / merge)
            student_number = len(student_state)
            video_length = len(student_state[0])

            for k in range(student_number):
                one = []
                for i in range(frame_real):
                    plus = sum(student_state[k][i * merge:(i + 1) * merge])
                    one.append(plus)
                student_merge.append(one)
            average = []
            for i in range(frame_real):
                sum1 = 0
                for student in student_merge:
                    sum1 += student[i]
                sum1 /= len(student_merge)
                average.append(sum1)
            print(average)
            plt.cla()
            plt.xlabel('时间')
            plt.ylabel('平均专注度')
            plt.plot(average, color='orange')
            plt.yticks(list(range(merge - 10, merge + 10, 4)))
            plt.savefig(path1 + '/average.png')

            for i in range(student_number):
                print(i)
                plt.cla()
                plt.plot(student_merge[i])
                plt.xlabel('时间')
                plt.ylabel('专注度')
                plt.yticks(list(range(0, 2 * merge, int(2 * merge / 10))))
                plt.savefig(path1 + '/' + str(student_name[i]) + '.png')

            models.shows.objects.create(name=name, time=time, address=address,
                                        author=author, average_list=average,
                                        average_number=sum(average)/len(average)/merge*100)

            return HttpResponse('提交成功！')

    else:
        form = AddForm()
    return render(request, 'post_.html', {'form': form})

