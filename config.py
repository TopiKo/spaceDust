import numpy as np

r = np.array([[0.,0.], [1.,0.], [0.,1.], [1.,1.]])
v = np.zeros((len(r), 2))
a = np.zeros((len(r), 2))
masses_ = np.ones(len(r))*2

#r_ = np.ma.array(r, mask=False)
v_ = np.ma.array(v, mask=False)
a_ = np.ma.array(a, mask=False)
F_ = np.zeros(len(r_))
dir_ij_arr_ = np.ma.zeros((len(r), len(r), len(r[0])))
