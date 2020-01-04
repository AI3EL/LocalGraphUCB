import numpy as np

alpha = 0.1
beta = 0.7
mu = 5

eps = 10e-6



def function(mu, mu_a, t_idx, N_a, t0):
    '''
    Valeur de la fonction à optimiser tf + phi
    '''
    return t0*(-mu) - np.log(3 * np.log(t_idx) / N_a - (mu + mu_a * (np.log(mu_a/mu) - 1)))


def function_true(mu):
    '''
    Valeur de la fonction à optimiser f
    '''
    return -mu


def line_search(mu, mu_a, t_idx, N_a, t, df, dx, t0):
    '''
    Backtracking line search. Récursif.
    '''
    alpha = 0.1
    beta = 0.7

    #print("dx" + str(dx))

    if function(mu + t*dx, mu_a, t_idx, N_a, t0) <= (function(mu, mu_a, t_idx, N_a, t0) + alpha*t*df*dx) \
            or ((mu_a/(mu + t*dx))>0 and ((3 * np.log(t_idx) / N_a - (mu + t*dx + mu_a * (np.log(mu_a/(mu + t*dx)) - 1)))>0)): #Il faut éviter que le log soit infini.
        return mu + t*dx
    else:
        return line_search(mu, mu_a, t_idx, N_a, beta*t, df, dx, t0)


def centering_step(mu, mu_a, t_idx, N_a, t, eps):
    '''
    Centering step, calculant les différentes valeurs de gradient de f, le pas x et la valeur lambda carré. Récursif.
    '''
    #print("mu: " + str(mu))
    #print("function: " + str(function(mu, mu_a, t_idx, N_a, t)))
    #print("function true: " + str(function_true(mu)))
    #print("limite: " + str(np.log(3 * np.log(t_idx) / N_a - (mu + mu_a * (np.log(mu_a/mu) - 1)))))
    denom = (3 * np.log(t_idx) / N_a - (mu + mu_a * (np.log(mu_a/mu) - 1)))
    #print(3 * np.log(t_idx) / N_a)
    #print("denom: " + str(denom))
    df = -t + (1 - mu_a/mu)/denom
    d2f = (1 - mu_a/mu)**2/denom**2 + (mu_a/mu**2)/denom
    dx = -1*df/d2f
    l2 = df**2 /d2f
    if l2/2 <= eps:
        return mu
    v1 = line_search(mu, mu_a, t_idx, N_a, t=1, df=df, dx=dx, t0=t)
    return centering_step(v1, mu_a, t_idx, N_a, t, eps)


def barr_method_inter(mu_a, t_idx, N_a, v0, eps, t, mu):
    '''
    Fonction récursif de la méthode de la barrière.
    '''
    v_center = centering_step(v0, mu_a, t_idx, N_a, t, eps)
    if 1/t < eps:
        return v_center
    else:
        t = mu*t
        return barr_method_inter(mu_a, t_idx, N_a, v0, eps, t, mu)


def barr_method(mu_a, t_idx, N_a, v0, eps=0.01, mu=5):
    '''
    Fonction de la méthode de la barrière.
    '''
    #print("FIRST MU: " + str(v0))
    #print("MUA: " + str(mu_a))
    #print("NA: " + str(N_a))
    #print('tidx: ' + str(t_idx))
    return barr_method_inter(mu_a, t_idx, N_a, v0, eps, 1, mu)