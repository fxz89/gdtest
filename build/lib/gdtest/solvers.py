import numpy as np
import sympy as sym
from scipy.sparse import coo_matrix

STATUS_SOLVED = "optimum reached"
STATUS_UNKNOWN = "maximum iteration reached"

class OptimizationProblem:
    """ 
    Base Optimization Problem
    It represents problem of the form
    
    min (1/2)x^THx + g^Tx

    """

    def __init__(self, max_iterations=1000, learning_rate=0.01, tolerance=10**(-3)):
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.tolerance = tolerance

    def evaluate_Jacobian(self, x_1, x_2):
        """
        Calculate Jacobian matrix

        Parameters
        ----------

        x_1 : coefficient matrix
        x_2 : coefficient vector

        Returns
        -------
        J: Jacobian matrix
        """

        pass

    def solve(self, x_1, x_2):
        """
        Solve optimization problem

        Parameters
        ----------

        x_1 : coefficient matrix
        x_2 : coefficient vector
        """

        pass


class GDOptProblem(OptimizationProblem):
    """ 
    Gradient descent method for optimization problem

    min x^THx + g^Tx
    """

    def __init__(self, max_iterations=1000, learning_rate=0.01, tolerance=10**(-3)):
        OptimizationProblem.__init__(self, max_iterations, learning_rate, tolerance)

        self.iteration = 0
        self.fval = 0
        self.variables = None
        self.status = STATUS_UNKNOWN
        self.J = None
        self.f = None

    def solve(self, x_1, x_2):
        """
        Solve optimization problem

        Parameters
        ----------
        x_1 : coefficient matrix
        x_2 : coefficient vector
        """

        # Checks
        assert(x_1.shape[0] == x_1.shape[1])
        assert(x_1.shape[0] == x_2.shape[0])

        # Local variables
        curr_x = np.random.uniform(0,1,size = (len(x_2), 1))
        prev_x = curr_x + 0.5*np.array(len(x_2), dtype = np.float64)
        fval = self.fval
        residual = self.tolerance + 1
        iteration = 0

        # Parameters
        tolerance = self.tolerance
        max_iterations = self.max_iterations
        learning_rate = self.learning_rate

        self.__function(x_1, x_2)
        self.evaluate_Jacobian(x_1, x_2)

        J = self.J
        f = self.f

        # Iterations of GD method
        print('iteration \t update step \t function value \t residual')
        while abs(residual) > tolerance and iteration < max_iterations:
            prev_x = np.copy(curr_x)
            curr_x -= learning_rate*J(prev_x)
            residual = abs(np.amax(curr_x - prev_x))
            iteration += 1
            fval = f(curr_x)
            print('{} \t {} \t {} \t {}'.format(iteration, curr_x, fval, residual))

        if abs(residual) <= tolerance:
            self.status = STATUS_SOLVED

        self.fval = fval
        self.variables = curr_x.T.tolist()

        self.__show()


    def evaluate_Jacobian(self, H, g):
        """
        Calculate Jacobian matrix

        Parameters
        ----------
        H : coefficient matrix
        g : coefficient vector

        Returns
        -------
        J: Jacobian function
        """

        H = coo_matrix(H)
        g = coo_matrix(g)
        self.J = lambda x: H*x+g

    def __function(self, H, g):
        """Build function"""
        H = coo_matrix(H)
        g = coo_matrix(g)
        self.f = lambda x: (1/2)*np.dot(x.T, H*x) + g.T*x        

    def __show(self):
        """Print Result"""
        print('\n Final Result')
        print('---------------')
        print('Stopping criteria: \t{}'.format(self.status))
        print('Final function value: \t{}'.format(self.fval))
        print('Decision Variables: \t{}'.format(self.variables))