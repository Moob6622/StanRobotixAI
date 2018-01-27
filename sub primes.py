from math import sqrt
from math import fabs

O = int(input())
for i in range (98/100*int(sqrt(O)), int(sqrt(O))) :
    if i%2==0 or i%5==0 :
        pass

else :
    if O%i==0 :
        print (i, O/i)
