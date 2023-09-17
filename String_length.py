# cook your dish here
T = int(input())

for i in range(T):
    string = input()
    if string.startswith("@"):
        string = string[1:] 
    words = string.split()
    output = ""

    for word in words:
        output += str(len(word)) + ","
    
    output = output.rstrip(",")
    print(output)
