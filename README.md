# recursive least squares
#### Adam Dhalla [adamdhalla.com](https://adamdhalla.com/)

Using the Sherman-Morrison-Woodbury Formula (Inverse Matrix Lemma) to efficiently calculate updates in online least squares problem, where the data matrix A is being constantly added to with new data points, as well as the b being constantly updated with new corresponding outputs. 

## **Input Syntax** 

Initiate the RecursiveLeastSquares class with an initial A and b - numpy Ndarrays. These A's and b's represent the data, and are what you would normally put into a Normal Equation. 

```
A = np.array([[1, 0], [1, 1], [1, 2]])
b = np.array([[3], [4], [7]])

rls = RecursiveLeastSquares(A, b) 
x0 = rls.x

newA = np.array([1, 3])
newb = np.array([11])

rls.addData(newA, newb)
x1 = rls.x
```

You can also choose to get many other class attributes from the RecursiveLeastSquares object, which can be seen in the code, but aren't really needed for the functioning of the code.

## **Under the Hood** 
*I have wrote a complete article on the mathematical foundations and explanation of this code [here](https://adamdhalla.medium.com/recursive-least-squares-b2407126c257)*
Using the Sherman-Morrison-Woodbury (S-M-W) we can find a more efficient way to compute least squares in situtations where we are continuously adding data to a matrix, instead of doing the normal equations and recalculating the computationally expensive A transpose A inverse term. 

Using S-M-W, each iteration of RLS has a matrix P, equal to the current A transpose times the current A inverse. We can indirectly calculate this term without actually doing the nasty computation, and with S-M-W, by knowing the original A transpose A inverse term we can infer what the new updated Anew transpose A inverse is. 

We can then multiply this matrix P by another term which returns us the current x. I explain more of these in much more detail on the article.
