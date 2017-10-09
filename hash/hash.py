name = "david"
list1 = []
for c in name:
    list1.append(ord(c) % 17)

nameLen = len(name)
for i in range(nameLen):
    print ("key: %s | keyValue: %s | index: %s" % (name[i],ord(name[i]),list1[i]))

david = "david"
david += "david"

print david
