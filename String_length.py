T = int(input('Test cases:'))

for i in range(T):
    string = input('STR:')
    words = string.split()
    init = words[0]
    last = words[-1]
    output = ''
    # if init[0] == '@':
    #     n = len(init)-1
    #     output = str(n)
    
    for j,word in enumerate(words):
        if init[0] == '@' and j == 0:
            n = len(init)-1
            output = str(n)+','
            continue
        elif last == word:
            output += str(len(word))
        else:
            temp = str(len(word))+','
            output += temp
    print(output)
