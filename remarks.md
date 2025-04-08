## Remark fromm profesor

### Question 3:

We have nonlinearity which differs from standard logistic. 
Loop n = 3000 with each x is a Gaussian N(o, I)
x_i in R^8
x_i are independent of each other because of the Identity matrix. 
We can also use for loop. 

### Question 4: 

- We will have large bias because of the nonlinearity. Maybe include a computation and graph of variane of the predictors. 
- Remarks: there is an intercept. number of parameter is p = 8 + 1 = 9.
- Give the formula to GD (explainations, math derivatives, well written, update rule)
- Define Z as: 1 column + X. 
- In order to get GD, we need: 
    - Gradient. We will divided by n to have something more like a gradient. 
    - Learning rate: 1/K with K is the number of iterations.
    - Initialize value of beta. Choose random value from standard gaussian or 0. 
    - Update rule. beta^k+1 = beta^k - gamma_k+1 * gradient with k > 0.
- SGD:
    - the point in exam is to consider general problems where the computing time is measured with respected to gradient evaluation. 
    specially number of time sigmoid(beta_i) is evaluated. 
    - We dont want to focus on logistic regression where some benefit are observed easily using vectorize operation.
    We can show which method decrease the Risk faster. 
    - Implementation: fix the number of iterations: K = 100 for GD. 
    For each stage: we need n evaluation of q_beta. 
    Do it K time. 
    cardinality of q_beta = 300_000. 
    - Choose minibatch size = m = 30. (include mini_batch size in SGD function)
    - We need 30 evaluation at each SGD iterations. 
    R_m^{SGD} = 1/m \sum (Y_i - q_beta(X_i)) * Z_i with Z_i = column(1 X_i)
    - So in the end, after \tilt{K} we have: \tilt{K}*m = Kn.
    We need \tilt{K} = Kn/m = 10_000.
    We dont need to be precise at the beginning of our iterations. The advantage of SGD over GD. 
    We dont need to use epoch in our code. 

### Question 5: 
- Just look at misclassification rate: propotion of wrong prediction. 

\hat{R_n}(given beta) = 1/n \sum(Indicator{Y_i <> h_beta(X_i)})

with h_beta(X) = 1 if q_beta(X) >= 1/2
= 0 if q_Beta(X) < 1/2. 

Or we can use Bernoulli(q_beta(X))

- Generate new sample for evaluation. (n=1000 sample size)
- For real data, we need to use test train spit for cross validation. 
