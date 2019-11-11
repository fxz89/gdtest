from gdtest.solvers import GDOptProblem
import numpy as np

tol = 0.0001
lr_r = 0.8
iter_max = 1000

H = np.array([[2,0,0], [0,2,0], [0,0,2]])
g = np.array([[-5],[-5],[-5]])

problem = GDOptProblem(iter_max, lr_r, tol)
# problem = GDOptProblem()
problem.solve(H,g)