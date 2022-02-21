import pickle
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator

# sentences = []
# for root, dirs, files in os.walk("data/original_data/"):
#     for name in files:
#         for line in pickle.load(open(root + "/" + name, "rb")):
#             sentences.append(line['sent'])
# pickle.dump(sentences, open('data/original_data/sentences.pkl', 'wb'))

sentences = pickle.load(open('data/original_data/sentences.pkl', 'rb'))
lengths = [len(s) for s in sentences]
lengths.sort()
count = [0]*11
for l in lengths:
    if l==0:
        count[0]+=1
    elif l>0 and l<=50:
        count[1]+=1
    elif l>50 and l<=100:
        count[2]+=1
    elif l>100 and l<=150:
        count[3]+=1
    elif l>150 and l<=200:
        count[4]+=1
    elif l>200 and l<=250:
        count[5]+=1
    elif l>250 and l<=300:
        count[6]+=1
    elif l>300 and l<=350:
        count[7]+=1
    elif l>350 and l<=400:
        count[8]+=1
    elif l>400 and l<=450:
        count[9]+=1
    elif l>450 and l<=500:
        count[10]+=1
print(count)

y = np.array(count)
x = np.array([i*50 for i in range(11)])
plt.title("Sentence length distribution")
plt.xlabel("Sentence length")
plt.ylabel("Number of sentence")
ax = plt.gca()
# ax为两条坐标轴的实例
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_major_locator(MultipleLocator(5000))
plt.xlim(0, 501)
plt.ylim(0, 27000)
plt.plot(x, y, color='red')
# plt.plot(epochs, y2, marker='h', color='b', label='CNN-5')
# plt.plot(epochs, y3, marker='v', color='g', label='ResCNN-5')
# plt.plot(epochs, y4, marker='^', color='r', label='ResCNN-9')
# plt.plot(epochs, y5, marker='D', color='y', label='ResCNN-11')
# plt.plot(epochs, y6, marker='p', color='m', label='ResCNN-13')
plt.legend()
plt.show()