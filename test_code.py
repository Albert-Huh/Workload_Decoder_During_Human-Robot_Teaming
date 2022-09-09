import numpy as np

a = np.array([1,0,3])
b = np.array([[1,2],[0,4]])
c = np.array([[[1.0,2.0],[1.0,2.0]],[[3.0,3.0],[3.0,3.0]], [[5.0,2.0],[5.0,2.0]],[[4.0,4.0],[4.0,4.0]]])
# print(a[2])
# print(b[-1])
# print(c[-1][-1])
d = np.sum(c, axis = -1, keepdims=True)
print(d)
# c /= d
e = c.mean(axis=-1)
print(len(c.shape))
print(e)
print(e.reshape(len(c),-1))
print(b[:,(0)&(1)])

C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
for pair in [(x, y) for x in C_range for y in gamma_range]:
    print(pair)
    print(pair[0])
    print(pair[1])

k_range = np.arange(20)
weight_fuc = ['uniform', 'distance']
opt_KNN_param = []
acc_max = 0
for KNN_param in [(x, y) for x in k_range for y in weight_fuc]:
    print(KNN_param)
    print(KNN_param[0])
    print(KNN_param[1])
# for i in range(4):
#     print(i)
'''
dic_list = [{'ch':'ch1','egg': 4, 'tomato': 1}, {'ch':'ch2','egg': 5, 'tomato': 3}]

chs = [ k['ch'] for k in dic_list]
print(chs)
eggs = [ k['egg'] for k in dic_list]
print(eggs)
print(max(eggs))

dic_list = [{'ch':'ch1','egg': np.array([1,1,1]), 'tomato': 1}, {'ch':'ch2','egg': np.array([4,5,7]), 'tomato': 3}]

chs = [ k['ch'] for k in dic_list]
print(chs)
eggs = [ k['egg'] for k in dic_list]
print(eggs)
print(np.amax(eggs))
'''