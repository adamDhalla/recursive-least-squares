#recursive least squares --adam dhalla
import numpy as np 
import numpy.linalg 

class RecursiveLeastSquares():
    """Creates a RecursiveLeastSquares object, that will efficiently returned modified x's 
       as more data is inputted using the .addData() method. Updates to x are calculated using
       the Sherman-Morrison-Woodburry Formula. 

       - initA (Ndarray)
                the initial A matrix in least squares, before adding any data. The calculations 
                will be based off this initial matrix. Is size (examples, variables). Think of 
                the A matrix in the Normal Equations. 

       - initB (Ndarray)
               the initial "answers" B matrix. Same B in the Normal Equations. Size (examples, 1)
    """
    def __init__(self, initA, initb):
        self.A = initA  
        self.b = initb 

        # create the initial P matrix, the (A^T*A)^-1 matrix.
        # we don't link it to self.A, self.b since these will change, and after
        # the first P, we will use S-M-W to calculate P instead. 

        initialP = np.linalg.inv(np.dot(initA.transpose(), initA))
        self.P = initialP

        # do least squares automatically the first round 

        # self.K is the other part of the normal equation that multiplies P, (A^T)*B
        self.K = None

        # do least squares automatically for first time
        self.x = np.dot(initialP, np.dot(initA.transpose(), initb)) 
    
    def addData(self, newA, newb):
        """add data to the least squares problems and returns an updated x.
           
           - newA    (ndarray)
                     adding more rows to the A matrix. Often a row vector (if adding one 
                     more data point). Otherwise, size is (newpoints, variables)
             
           - newb    (ndarray)
                     adds corresponding 'output' for the newA. A (1, 1) ndarray if adding
                     only one more data point. Else, size is (newpoints, 1)

            Returns the updated x. 
        """ 

        newA = newA.reshape(-1, (np.shape(newA)[0]))
        self.A = np.concatenate([self.A, newA])

        newb = newb.reshape(-1, 1)
        self.b = np.concatenate([self.b, newb])

        # create P by using Sherman-Morrison-Woodburry
        # I separate the formula into chunks for readability, see README for details

        # size of I depends on rows of data inputted
        I = (np.eye(np.shape(newA)[0]))


        PIn = np.linalg.inv(I + np.dot(newA, np.dot(self.P, newA.transpose())))
        PinA = np.dot(np.dot(newA.transpose(), PIn), newA)
        PinAP = np.dot(np.dot(self.P, PinA), self.P)
        P = self.P - PinAP

        # create K 
        self.P = P
        self.K = np.dot(self.P, newA.transpose())


        Q = newb - np.dot(newA, self.x)

        self.x = self.x + np.dot(self.K, Q)
        return self.x
