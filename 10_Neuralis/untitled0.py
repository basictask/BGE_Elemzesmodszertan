# -*- coding: utf-8 -*-
def get_quadrant(j):
    r = j%6
    b = 3
    c = 18
    if ((r > 0) & (r <= b) & (j/c<=1)):
        return 1
    elif (((r == 0) or (r > b)) & (j/c<=1)):
        return 2    
    elif ((r > 0) & (r <= b) & (j/c>1)):
        return 3
    elif (((r==0) or (r > b)) & (j/c>1)):
        return 4
    
for i in range(1,65):
    print(i, '\t', get_quadrant(i))
    if(i%3==0):
        print()
