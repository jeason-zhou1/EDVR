import os 
import re
import numpy as np

f = open('./result.txt')
videonumbers = 1000
lines = f.readlines()
a = [ [] for i in range(videonumbers)]
ccc = 0
for num,line in enumerate(lines):
    if not line.find('[')>-1:# 寻找视频名字的关键字
        print(num)
    else:
        items = re.split('[\[\] ]',line[:-1])
        temp = []
        for i in range(len(items)):
            # print(i,items)
            if items[i]!='':
                temp.append(int(items[i]))
        # items.remove('')
        # print(temp)
        if temp[1]-temp[0]>7:
            print(temp)
            a[num].append(temp)
# print(a)
a= np.asarray(a)
print(a)
np.save('youku',a)
# print(ccc)