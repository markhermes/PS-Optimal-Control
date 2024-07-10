import numpy as np
import matplotlib.pyplot as plt

class PseudoSpectral:
    
    def __init__(self,
                 N_,
                 time_start_    = -1,
                 time_end_      = +1,
                 dynamic_constraint_threshold_ = 0.01,
                 func_vals_ = 0):
        self.N = N_;    
        self.t_start = time_start_;    
        self.t_end = time_end_
        self.true_deriv = []; 
        self.numerical_diff = []
        self.dynamic_constraint_threshold = dynamic_constraint_threshold_
        
        self.cgl_nodes   = self.chebyshev_gauss_lobatto_nodes(N_)      # time
        self.cgl_weights = self.clenshaw_curtis_weights(N_)            # integration of the function
        self.D           = self.chebyshev_differentiation_matrix(N_)   # diff of the function
        
        self.map_nodes_and_weights()

        self.func_vals   = func_vals_


    def chebyshev_gauss_lobatto_nodes(self,N):
        """Generate Chebyshev-Gauss-Lobatto nodes."""
        if N < 2:
            raise ValueError("N must be 2 or greater for CGL nodes.")
        
        # Compute the Chebyshev-Gauss-Lobatto nodes
        nodes = -np.cos(np.pi * np.arange(N+1) / N)
        # nodes = np.cos(N * np.acos()) 

        return nodes

    def clenshaw_curtis_weights(self,N):
        """Generate Clenshaw-Curtis weights."""
        
        w = np.zeros(N+1)
        
        if np.mod(N,2) == 0: # if N is even
            w[0] = 1/(N**2-1)
            w[N] = w[0]
            for i in range(1,N//2 + 1): # need to // to represent integer division
                sum = 0
                # iterate for the sum since we have to half the first and last element
                for j in range(0,N//2 + 1):
                    sumarg = (1/(1-4*j**2))*np.cos(2*np.pi*j*i/N)
                    if(j == 0  or j == N//2):
                        sumarg = sumarg/2
                    sum += sumarg
                w[i] = 4/N*sum
                w[N-i] = w[i]

        else:           # N is odd
            print(range(1,((N-1)//2)+1))
            w[0] = 1/N**2
            w[N] = w[0]
            for i in range(1,((N-1)//2+1)):
                sum = 0
                # iterate for the sum since we have to half the first and last element
                for j in range(0,(N-1)//2+1):
                    sumarg = (1/(1-4*j**2))*np.cos(2*np.pi*j*i/N)
                    if(j == 0  or j == N/2):
                        sumarg = sumarg/2
                    sum += sumarg
                w[i] = 4/N*sum
                w[N-i] = w[i]


        weights = w
        return weights


    def chebyshev_differentiation_matrix(self,N):
        """Generate the Chebyshev differentiation matrix at the CGL nodes."""
        if N < 2:
            raise ValueError("N must be 2 or greater for Chebyshev differentiation matrix.")
        
        x = self.cgl_nodes
        c = np.ones(N+1)
        c[0] = 2
        c[1:N] = 1
        c[-1] = 2


        # Initialize the differentiation matrix
        D = np.zeros((N+1, N+1))
        
        # Fill in the matrix entries
        for i in range(1,N):
            for j in range(1,N):
                if i == j:
                        D[i, j] = x[i] / (2 * (1 - x[i]**2))
                else:
                    D[i, j] = c[i] / c[j] * (-1)**(i + j) / (x[i] - x[j])
        
        # Adjust the boundary conditions for Chebyshev-Gauss-Lobatto nodes
        D[0, 0] = -(2 * N**2 + 1) / 6
        D[N, N] = D[0, 0]
        
        return D
    
    def compute_numerical_derivs(self):
        self.numerical_diff = np.gradient(self.func_vals, self.cgl_nodes)
        return self.numerical_diff
    
    def map_nodes_and_weights(self):

        new_range   = (self.t_end - self.t_start)
        self.t_range = new_range

        # map the nodes: add + 1 then scale then add offset
        self.cgl_nodes = (self.cgl_nodes + 1) * new_range/2 + self.t_start

        # map the weights to the new range : range/oldrange=(1-(-1))
        self.cgl_weights *= new_range/2

        # map the differential operator
        self.D *= 2/new_range


    def integrate_quadrature(self,y):
        return( y@self.cgl_weights)
    
    def plot_data(self, 
                  use_numerical_deriv_ = False): # the differential matrix is very oscillatory 
                                                # skip debugging this for now and compute numerical estimates):

    
        plt.figure()
        plt.plot(self.cgl_nodes,self.func_vals,'-ok',label = 'Function')
        # plt.plot(self.cgl_nodes,self.true_deriv,'-r',label = 'True Derivative')

        # plot the integrated sum if the function values are not empty
        if self.func_vals.any():
            int_val = round(self.integrate_quadrature(self.func_vals),2)
            plt.title(str(f'Integrated Value {int_val}'))

        # plot the numerical deriv if that is selected
        if(use_numerical_deriv_ == True):
            self.compute_numerical_derivs()
            plt.plot(self.cgl_nodes,self.numerical_diff,'*r',label = 'Numerical Gradient')            
        else:
            plt.plot(self.cgl_nodes,self.D@self.func_vals,'or',label = 'Derivative')
    
        
        plt.legend()

        plt.show()


if __name__ == "__main__":
        
    # generate the nodes, weights, and differentiation matrix
    N = 20
    ps_obj = PseudoSpectral(N,time_start_=0,time_end_=10)

    test_arg = 2*np.pi/7
    ps_obj.func_vals = np.cos(ps_obj.cgl_nodes * test_arg)
    ps_obj.true_deriv = -test_arg * np.sin(ps_obj.cgl_nodes * test_arg)
    ps_obj.plot_data(use_numerical_deriv_= True)
