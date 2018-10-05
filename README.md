# mnist_gradientDescent

<pre>

This is simple classification of data using Gradient Desent Algorithm.


Gradient Descent Algorithm is applied using following four equation:

Input x                 : Set the corresponding activation a1 for the input layer

Feedforward             : For each l=2,3,…,L compute 
                          Z[l]  =  W[l]a[l−1]+B[l]    and
                          A[l]  =  σ(Z[l])

Output error δ[L]         : Compute the vector 
                          δ[L]  =  ∇aC ⊙ σ′(Z[L])

Backpropagate the error : For each l=L−1,L−2,…,2 compute 
                          δ[l]  =  ((W[l+1])T δ[l+1]) ⊙ σ′(Z[l])

Output                  : The gradient of the cost function is given by 
                          ∂C/∂W[l](j,k)  =  a[l−1](k) δ[l](j)   and 
                          ∂C/∂B[l](l)    =  δ[l](j)

Implementation:

err[-1]                       = loss * activator.derivative[-1]
for lay_num in (total_layers-1):
  err[-L]                       = hadmardProduct(dotProduct(weight[-L+1].transpose, err[-L+1]), activator.derivative[-L])
  delta_C/delta_bias[-L][j]     = delta_err[-L]
  delta_C/delta_weight[-L][j]   = delta_err[-L] * output[-L-1]

</pre>
