import numpy as np
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