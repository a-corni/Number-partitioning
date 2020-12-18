import numpy as np
import sys
sys.path.append('/mnt/d/Users/Antoine/Documents/EntropicaLabs')
import QAOA
###Define a set to partition

#Manually defined set
s = np.array([4,5,6,7,8])

#Randomly generated : 
#n = 5
#s = np.rand(0, (2**15//n)*n, n)

###Define number of steps 
p = 3

###Get partition

(s1, s2) = QAOA.bipartition(s,p)
print("Initial set s = ", s, " is partitioned into : ")
print("- set 1 : ", s1)
print("- set 2 : ", s2)
