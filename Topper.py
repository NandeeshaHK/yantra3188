T = int(input())
for i in range(T):
    students = []
    students_score = []
    N = int(input())
    for j in range(N):
        name_score = input()
        words = name_score.split()
        students.append(words[0])
        students_score.append(float(words[1]))
    topper = 0
    output = ''
    max_score = max(students_score)
    for k in range(N):
        if students_score[k] == max_score:
            if output == '':
                output = students[k]
            else:
                output +=  ' ' + students[k]
    toppers = output.split()
    toppers.sort()
    for k in toppers:
        print(k)
