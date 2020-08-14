# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:09:25 2019

@author: tomson
"""

import pickle
#with open('results.pkl','rb') as file:
#    a=pickle.load(file)
from matplotlib import pyplot as plt 
import numpy as np
#y1=[item[0] for item in a]
#y2=[item[3] for item in a]
#y3=[item[6] for item in a]
#x=range(1,401)
#plt.plot(x,y1)
#plt.show()
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 15,
}
plt.figure(figsize=(6, 4), dpi=100)
x1=[85.3,85.7,86.3,85.9,86.0,85.6,85.7,85.5,85.7,85.7]
x1=[77.3,77.6,78.2,78.0,77.8,77.9,77.6,77.4,77.4,77.6]
#x2=[92.4,92.6,92.7,92.9,92.7,92.7,92.6,92.5,92.7,92.7]
#x1=[88.8,89.4,89.6,89.6,89.4,89.4,89.5,89.5,89.5,89.4]
#x2=[88.4,88.5,88.5,88.8,89.0,88.8,88.9,88.9,88.7,88.8]
y=range(1,11)
#y=np.arange(0,1,0.1)+0.1
#x1=[93.6,93.8,93.5,93.3,93.1,93.0,92.8,92.6,92.5,92.3]
#x1=[89.6,89.9,90.1,89.8,89.7,89.5,89.4,89.2,88.9,88.7]
print(y)
#movie_name = ['SA','MA','MIA','LV','BERT']
#first_day = [0.6, 0.4, 1.4, 1.0, 1.9]
#first_weekend = [0.4, 0.5, 1.6, 1.2, 1.3]

# 先得到movie_name长度, 再得到下标组成列表
#x = range(len(movie_name))

#plt.bar(x, first_day, width=0.2,label='F1 for ECE')
# 向右移动0.2, 柱状条宽度为0.2
#plt.bar([i + 0.2 for i in x], first_weekend, width=0.2,label='EM for ED+ECPE')
#plt.plot()

plt.plot(y,x1,color="deeppink",linewidth=2,linestyle=':', marker='o')
#plt.plot(y,x2,color="blue",linewidth=2,linestyle='-',label='semi', marker='*')
# 底部汉字移动到两个柱状条中间(本来汉字是在左边蓝色柱状条下面, 向右移动0.1)
#plt.xticks([i + 0.1 for i in x], movie_name)
plt.xticks(fontproperties = 'Times New Roman', size = 20)
plt.yticks(fontproperties = 'Times New Roman', size = 20)
#plt.legend(loc='upper right',prop=font1)
plt.ylabel('Accuracy(%)', fontsize=20,fontdict=font1)
plt.xlabel('T (iterations)', fontsize=20,fontdict=font1)
#plt.xlabel('Gaussian Filter Value', fontsize=20,fontdict=font1)
plt.tight_layout()
plt.savefig('haha1.eps')
plt.show()