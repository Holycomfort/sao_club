import os

f_train = open('./dataset1/train.txt', 'w')
f_valid = open('./dataset1/valid.txt', 'w')

for index, filename in enumerate(os.listdir('./dataset1/train')):
    if index < 130:
        f_train.write('./dataset1/train/' + filename + ' ')
        f_train.write('./dataset1/train_GT/SEG/man_seg' + filename[1:] + '\n')
    else:
        f_valid.write('./dataset1/train/' + filename + ' ')
        f_valid.write('./dataset1/train_GT/SEG/man_seg' + filename[1:] + '\n')
