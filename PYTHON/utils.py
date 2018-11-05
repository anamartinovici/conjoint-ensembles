'''
Functions in this file:

    pathology(beta, kind=None)

    generate_simulated_design(nresp, nscns, nalts, nlvls, ncovs)

    compute_beta_response(data_dict, pathology_type=None)

    generate_simulated_data(pathology_type=None)

    fit_model(data_dict, model_name)

    get_loo_list(fit)

'''


from . import psis
import pystan
import pickle
import numpy as np
from matplotlib import colors
import matplotlib.pyplot as plt
from scipy.optimize import minimize, LinearConstraint


# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

    
def pathology(beta, kind="none", prob=[.5, .5]):
    if kind == 'none':
        pass
    elif kind == 'ANA':
        beta *= np.random.choice([1, 0], size=len(beta), p=prob)
    elif kind == 'screening':
        beta[-3:] -= 100
    elif kind == 'screening_inf':
        beta[-1] = -np.inf
    elif kind == 'screening_random':
        if int(np.random.choice([1,0], p=prob)):
            i = np.random.randint(len(beta))
            beta[i] = -10000
    elif kind == 'ANA_systematic':
        pathology_vector = np.ones_like(beta)
        pathology_vector[int(len(beta)//2):] = 0
        beta *= pathology_vector
    elif kind == 'ANA_random':
        if int(np.random.choice([1, 0], p=prob)):
            pathology_vector = np.ones_like(beta)
            pathology_vector[:int(len(beta)//2)] = 0
            beta *= pathology_vector
    return beta


def generate_simulated_design(nresp=100, nscns=10, nalts=4, nlvls=12, ncovs=1):
    
    # X is the experimental design
    X = np.zeros((nresp, nscns, nalts, nlvls))
    # Z is the covariates
    Z = np.zeros((ncovs, nresp))
    
    for resp in range(nresp):
        z_resp = 1
        if ncovs > 1:
            raise NotImplementedError
    
        for scn in range(nscns):
            X_scn = np.random.choice([0,1], p =[.5,.5], size=nalts*nlvls).reshape(nalts,nlvls)
            X[resp, scn] += X_scn
    
        Z[:, resp] += z_resp
    
    # dictionary to store the simulated data and generation parameters
    data_dict = {'X':X,
                 'Z':Z.T,
                 'A':nalts,
                 'R':nresp,
                 'C':ncovs,
                 'T':nscns,
                 'L':nlvls}
    return data_dict


def compute_beta_response(data_dict, pathology_type=None):

    # beta means
    Gamma = np.random.uniform(-3, 4, size=data_dict['C'] * data_dict['L'])
    # beta variance-covariance
    Vbeta = np.diag(np.ones(data_dict['L'])) + .5 * np.ones((data_dict['L'], data_dict['L']))

    # Y is the response
    Y = np.zeros((data_dict['R'], data_dict['T']))
    # Beta is the respondent coefficients (part-worths/utilities)
    Beta = np.zeros((data_dict['L'], data_dict['R']))
    
    for resp in range(data_dict['R']):
        z_resp = 1
        if data_dict['C'] > 1:
            raise NotImplementedError
    
        beta = np.random.multivariate_normal(Gamma, Vbeta)
        if pathology_type:
            beta = pathology(beta, kind=pathology_type)
    
        for scn in range(data_dict['T']):
            X_scn = data_dict['X'][resp, scn]

            U_scn = X_scn.dot(beta) - np.log(-np.log(np.random.uniform(size=data_dict['C'])))
            Y[resp, scn] += np.argmax(U_scn) + 1
    
        Beta[:, resp] += beta

    data_dict['B'] = Beta
    data_dict['Y'] = Y.astype(int)

    if pathology_type == 'ANA':
        data_dict['w'] = np.random.binomial(1, .5, size=data_dict['T'])

    return data_dict


def generate_simulated_data(pathology_type="none", use_stan=False):
    if use_stan:
        sm = get_model(model_name='generate_data')
        data_dict = {'A':4, 'L':12, 'T':10, 'R':100, 'C':1}
        data = sm.sampling(data=data_dict,
                           iter=100,
                           warmup=0,
                           chains=1,
                           refresh=100,
                           seed=1750532,
                           algorithm="Fixed_param")
        # pystan fit objects can take a long time to unpack...
        data_dict.update(data.extract(pars=['X','Y','Z','B']))
        for v in ['X', 'Y', 'Z', 'B']:
            data_dict[v] = data_dict[v][-1]
        data_dict['B'] = data_dict['B'].T
        data_dict['Y'] = data_dict['Y'].astype(int)
    else:
        data_dict = generate_simulated_design()
        data_dict = compute_beta_response(data_dict, pathology_type=pathology_type)
    return data_dict


def get_model(model_name='mnl_vanilla'):

    with open('./STAN/{0}.stan'.format(model_name), 'r') as f:
        stan_model = f.read()
    
    try:
        sm = pickle.load(open('./STAN/{0}.pkl'.format(model_name), 'rb'))
    
    except:
        sm = pystan.StanModel(model_code=stan_model)
        with open('./STAN/{0}.pkl'.format(model_name), 'wb') as f:
            pickle.dump(sm, f)
    
    return sm


def plot_ppc(data_dict, fit):
    # define variables
    Y = data_dict['Y'].flatten()
    Y_ppc = fit.extract(pars=['Y_ppc'])['Y_ppc']
    B = data_dict['A']+2
    
    fig = plt.figure(figsize=(8,8))
    ax = plt.gca()
    bins = [b - 0.5 for b in range(B + 1)]

    idxs = [ idx for idx in range(B) for r in range(2) ]
    xs = [ idx + delta for idx in range(B) for delta in [-0.5, 0.5]]

    # make a histogram for each sample of the markov iteration
    counts = [np.histogram(Y_ppc[n].flatten(), bins=bins)[0] for n in range(Y_ppc.shape[0])]
    probs = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    # find the percentiles for each sample-histogram
    creds = [np.percentile([count[b] for count in counts], probs) for b in range(B)]
    pad_creds = [ creds[idx] for idx in idxs ]

    ax.fill_between(xs, [c[0] for c in pad_creds], [c[8] for c in pad_creds], color='y', lw=0, alpha=.1)
    ax.fill_between(xs, [c[1] for c in pad_creds], [c[7] for c in pad_creds], color='orange', lw=0, alpha=.1)
    ax.fill_between(xs, [c[2] for c in pad_creds], [c[6] for c in pad_creds], color='orange', lw=0, alpha=.1)
    ax.fill_between(xs, [c[3] for c in pad_creds], [c[5] for c in pad_creds], color='red', lw=0, alpha=.1)

    ax.plot(xs, [c[4] for c in pad_creds], color='r', alpha=.3)
    ax.hist(Y, bins=bins, histtype='step', color='#4b0082', alpha=.8)
    plt.show()

    
def plot_betas(B, plot_title='model coefficients'):
    max_beta = np.absolute(B).max()
    # Plot the betas both generated and estimated
    plt.figure(figsize=(16,8))

    plt.subplot(311)
    plt.imshow(B, cmap='RdGy_r', norm=MidpointNormalize(midpoint=0, vmin=-max_beta, vmax=max_beta))
    plt.title(plot_title)

    plt.subplot(312)
    plt.plot(np.arange(12), B.mean(axis=1), color='r')
    plt.title("Feature Level Avg")

    plt.subplot(313)
    y = B.mean(axis=0)
    plt.plot(np.arange(len(y)), y, color='r')
    plt.title("Respondent Avg")

    plt.show()

    
def plot_respondent(r, data_dict, fit):
    B = fit.extract(pars=['B'])['B']
    plt.figure(figsize=(12,15))
    for l in range(data_dict['L']):
        ax = plt.subplot(4,3,l+1)
        ax.hist(B[:,r,l], color='r', alpha=.5)
        ax.axvline(data_dict['B'][l,r], color='k')
        ax.set_title("Beta {0}".format(l+1))
    plt.show()

    plt.figure(figsize=(8,15))
    for t in range(data_dict['T']):
        ax = plt.subplot(5,2,t+1)
        ax.hist(Y_ppc[:,r,t], color='r', alpha=.5)
        ax.axvline(data_dict['Y'][r, t], color='k')
        ax.set_title("Task {0}".format(t+1))
    plt.show()


#def get_loos(fit):
#
#    log_lik = fit.extract(pars='log_lik')['log_lik']
#
#    LL = np.zeros((log_lik.shape[0], log_lik.shape[1]*log_lik.shape[2]))
#    for i in range(log_lik.shape[0]):
#        LL[i] = log_lik[i].flatten()
#
#    return psis.psisloo(LL)
#
#
#def stacking_weights(LL_list):
#    lpd_point = np.vstack(LL_list).T
#    N = lpd_point.shape[0]
#    K = lpd_point.shape[1]
#    exp_lpd_point = np.exp(lpd_point)
#
#    # neg_log_score_loo
#    def f(w):
#        w_full = np.hstack((w, 1-np.sum(w)))
#        S = 0
#        for i in range(N):
#            S += np.log(np.exp(lpd_point[i, :]).dot(w_full))
#        return -S
#
#    # grad_neg_log_score_loo
#    def grad_f(w):
#        w_full = np.hstack((w, 1-np.sum(w)))
#        grad = np.zeros(K-1)
#        for k in range(K-1):
#            for i in range(N):
#                grad[k] += (exp_lpd_point[i,k] - exp_lpd_point[i,-1]) / (exp_lpd_point[i, :].dot(w_full))
#        return -grad
#
#    ui = np.vstack((-np.ones(K-1), np.diag(np.ones(K-1))))
#    ci = np.zeros(K)
#    ci[0] = -1
#    x0 = (1/K)*np.ones(K-1)
#    lincon = ({'type': 
#    out = minimize(f, x0, method='COBYLA', jac=grad_f, constraints=[lincon])
#    return out
#
