## Model definition
def model(x,theta):
    logit = theta.T @ x
    a = np.exp(logit)
    s = np.sum(a, axis=0)
    b = s @ np.ones((theta.shape[1], 1))
    return a / b
 ## Cost function
 # for one example
def single_cost(xi,yi,theta,c):
    x = xi.reshape(-1,1)
    h = model(x,theta)
    c = c.reshape(-1,1) 
    b = np.round(yi == c).reshape(-1,1)
    return np.sum(b * np.log(h)) 
# for the whole example  
def cost(x,y,theta):
    c = np.unique(y)
    rs = []
    for i in range(x.shape[1]):
        xi = x[:,i]
        yi = y[i]
        sc = single_cost(xi,yi,theta,c)
        rs.append(sc)
    return -1 * np.sum(rs)
 ### Gradient computing
def single_grad(xi,yi,theta, c):
    a = np.round(yi==c).reshape(-1,1) 
    x = xi.reshape(-1,1)
    b = model(x,theta)
    d = x @ (a - b).T
    return - d
def grad(x,y,theta):
    rs = []
    c = np.unique(y)
    grad = np.zeros(theta.shape)
    for i in range(x.shape[1]):
        grad = grad + single_grad(x[:,i],y[i],theta,c) 
    return -1 * grad
 
 ### Gradient descent
def gradient_descent(x,y,theta,etha=0.001,n_iterations=100):
    costs = []
    for i in range(n_iterations):
        theta = theta + etha * grad(x,y,theta)
        costs.append(cost(x,y,theta))
    return theta,costs  