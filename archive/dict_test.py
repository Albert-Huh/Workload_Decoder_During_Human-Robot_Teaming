d = {}
d['fruit'] = []
d['number'] = []
d['boo'] = [True, False, False, True]

print(d)

text = 'apple'
d['fruit'].append(text)
d['fruit'].append('banana')
print(d.keys())

l = [1, 2, 3]
d['number'].append(l)
print(d)
key_idx = ['fruit', 'number', 'boo']
print(d[key_idx[2]])