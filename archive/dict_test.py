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

new_keys = []
check_list = ['fruit', 'boo']
for i in check_list:
    if i in d.keys():
        print(i)
        new_keys.append(i)
print(new_keys)
new_d = {new_key: d[new_key] for new_key in new_keys}
print(new_d)