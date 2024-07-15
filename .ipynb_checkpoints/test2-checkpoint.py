table = {}

maxn = 30

for n in range(maxn):
    if n not in table:
        table[n] = {}
    for k in range(maxn):
        if n == k == 0:
            table[n][k] = 1
        elif n == 0 or k == 0:
            table[n][k] = 0
        else:
            table[n][k] = (n-1)*table[n-1][k] + table[n-1][k-1]

for n in range(maxn):
    for k in range(maxn):
        print(table[n][k], end=" ")
    print()

n = 21
print(sum(table[n][k]*k for k in range(n+1)) / sum(table[n][k] for k in range(n+1)))