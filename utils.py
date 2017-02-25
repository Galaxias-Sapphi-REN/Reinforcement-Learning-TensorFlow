#coding:utf-8

def printboard(state):
    n_row = len(state)
    n_total = n_row**2
    n_line = n_row*9+1
    b = '-'*n_line+'\n|'

    for i in range(n_total):
        if i%n_row!=0:
            b = b + ' {' + str(i) + '} |'
            if i == n_total-1:
                b = b + '\n' + '-'*n_line
        elif i%n_row == 0 and i != 0:
            b = b + '\n' + '-'*n_line + '\n|'
            b = b + ' {' + str(i) + '} |'
        else:
            b = b + ' {' + str(i) + '} |'

    cells = []
    for i in range(n_row):
        for j in range(n_row):
            cells.append(state[i][j].center(6))
    print b.format(*cells)

