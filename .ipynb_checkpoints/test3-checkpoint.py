import numpy as np

n = 21

def count_circles(p):
    visited = {i: False for i in range(n)}
    counter = 0
    for i in range(n):
        if not visited[i]:
            counter += 1
            visited[i] = True
            j = p[i]
            while j != i:
                visited[j] = True
                j = p[j]
    return counter


avg = lambda L: sum(L)/len(L)

record = []
while 1+1 == 2:
    p = np.random.permutation([i for i in range(n)])
    cc = count_circles(p)
    record.append(cc)
    print(p,cc,avg(record))