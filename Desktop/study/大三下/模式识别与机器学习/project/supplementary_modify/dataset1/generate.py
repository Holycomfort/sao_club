import os

f_train = open('./train.txt', 'w')
f_valid = open('./valid.txt', 'w')

for index, filename in enumerate(os.listdir('./train')):
    if index < 140:
        f_train.write('./dataset1/train/' + filename + ' ')
        f_train.write('./dataset1/train_GT/SEG/man_seg' + filename[1:] + '\n')
    else:
        f_valid.write('./dataset1/train/' + filename + ' ')
        f_valid.write('./dataset1/train_GT/SEG/man_seg' + filename[1:] + '\n')
