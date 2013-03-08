import os
import itertools

import scipy as sp
import copy

from vb_mf import normalize_trans


def clique_init_args(args):
    """initialize args with cliqued parameters"""
    I,T,K,L = args.I, args.T, args.K, args.L
    args.clq_Q = sp.zeros((T, K**I))
    args.clq_Q_pairs = sp.zeros((T, K**I, K**I))
    args.clq_init = sp.ones((K**I))
    args.clq_trans = sp.ones((K**I, K**I))
    # emission probabilities clamped to the current t's observations
    args.clq_emit = sp.ones((K**I, T))
    args.clq_emit_no_t = sp.ones((K**I, L*2))

def clique_update_q(args):
    """update Q distribution by cliqueing the values"""
    args.Q_prev = copy.deepcopy(args.Q)

    print '# recliqueing parameters'
    clique_init_probs_from_params(args.clq_init, args.beta, args.gamma, args.I)
    clique_transmission_from_params(args.clq_trans, args.alpha, args.theta, args.I)
    clique_emit_from_params(args.clq_emit, args.emit_probs, args.I, args.X)
    
    print '# inferring clique hidden states'
    args.clq_loglh = infer_hidden_marginals(args.clq_Q, args.clq_Q_pairs, args.clq_init, args.clq_trans, args.clq_emit)
    print 'log likelihood is', args.clq_loglh
    
    print '# uncliqueing Q'
    clique_marginals_to_Q(args.clq_Q, args.Q)

def clique_update_params(args):
    """update parameters after cliqued hmm"""
    print '# updating params from Q'
    params_from_clique_marginals(args.theta, args.alpha, args.beta, args.gamma, args.emit_probs, args.clq_Q, args.clq_Q_pairs, args.X, args)


def clique_init_probs_from_params(clq_init, beta, gamma, I):
    """Convert beta (K*K) and gamma (K) matrices into a K**I init_probs distribution"""
    K = gamma.shape[0]
    combinations = list(enumerate(itertools.product(range(K), repeat=I)))
    clq_init[:] = 1.
    vp = 0
    for k_from, from_val in combinations:
        clq_init[k_from] *= gamma[from_val[0]]
        for index in xrange(1, I):
            clq_init[k_from] *= beta[from_val[vp], from_val[index]]

def clique_transmission_from_params(clq_trans, alpha, theta, I):
    """Convert an alpha (K*K) matrix into a cliqued transition matrix (K^I * K^I)"""
    K = alpha.shape[0]
    clq_trans[:] = 1.
    vp = 0
    # each species can be in any of K states; moving from one of these combinations to one of these combinations
    combinations = list(enumerate(itertools.product(range(K), repeat=I)))
    for ((k_from, from_val), (k_to, to_val)) in itertools.product(combinations, repeat=2):
        clq_trans[k_from, k_to] *= alpha[from_val[0], to_val[0]]
        for index in xrange(1, I):
            clq_trans[k_from, k_to] *= theta[to_val[vp], from_val[index], to_val[index]]

def clique_emit_from_params(clq_emit, emit_probs, I, X):
    """create a T x K**I matrix from a K*L emission matrix and also bind to the values of X"""
    I,T,L = X.shape
    K,L = emit_probs.shape
    combinations = list(enumerate(itertools.product(range(K), repeat=I)))
    clq_emit[:] = 1.
    # calculate the probability of observing X
    for t in xrange(T):
        for k_from, from_val in combinations:
            for i, k in enumerate(from_val):
                for l in xrange(L):
                    #clq_emit[k_from,t] *= emit_probs[k, l]
                    if X[i,t,l]:
                        clq_emit[k_from,t] *= emit_probs[k, l]
                    else:
                        clq_emit[k_from,t] *= 1. - emit_probs[k, l]

def clique_emit_1d_from_params(clq_emit_no_t, emit_probs, I, X):
    """create a K**I x L matrix from a K*L emission matrix without binding the values of X"""
    I,T,K = X.shape
    K,L = emit_probs.shape
    combinations = list(enumerate(itertools.product(range(K), repeat=I)))
    clq_emit_no_t[:] = 1.
    # calculate the probability of observing X
    for k_from, from_val in combinations:
        for i, k in enumerate(from_val):
            for l in xrange(L):
                clq_emit_no_t[k_from,l+1] *= emit_probs[k, l]
                clq_emit_no_t[k_from,l] *= 1. - emit_probs[k, l]


def clique_likelihood(args):
    """calculate the data likelihood for the current parameters"""
    return args.clq_loglh
    #print 'nothin'
    #Q = sp.zeros((args.T, args.K ** args.I))
    #T,K = Q.shape
    #a_s = sp.zeros((T,K)) 
    #loglh = 0.
    #
    #init_probs, transmat, emit_probs_mat = args.clq_init, args.clq_trans, args.clq_emit
    #
    #a_t = init_probs * emit_probs_mat[:, 0]
    #s_t = [a_t.sum()]
    #a_t /= s_t[0]
    #a_s[0,:] = a_t
    #
    ## forward algorithm
    #for t in range(1,T):
    #    a_t = emit_probs_mat[:,t] * sp.dot(a_t.T, transmat)
    #    s_t.append(a_t.sum())
    #    a_t /= s_t[t]
    #    a_s[t,:] = a_t
    #
    #loglh = sp.log(sp.array(s_t)).sum()
    #return loglh
    

def infer_hidden_marginals(Q, Q_pairs, init_probs, transmat, emit_probs_mat):
    """Do forward-backward algorithm to infer hidden nodes' marginal distributions"""
    Q[:] = 0.
    Q_pairs[:] = 0.
    T,K = Q.shape
    a_s = sp.zeros((T,K)) 
    b_s = sp.zeros((T,K))
    loglh = 0.
    
    #emit_probs_mat = sp.exp(log_emit_probs(emit_probs, X))
    a_t = init_probs * emit_probs_mat[:, 0]
    s_t = [a_t.sum()]
    a_t /= s_t[0]
    b_t = sp.ones((K,))

    a_s[0,:] = a_t
    b_s[T-1,:] = b_t

    # forward algorithm
    for t in range(1,T):
        a_t = emit_probs_mat[:,t] * sp.dot(a_t.T, transmat)
        s_t.append(a_t.sum())
        a_t /= s_t[t]
        a_s[t,:] = a_t

    # backward algorithm        
    for t in range(T-2,-1,-1):
        b_t = sp.dot(transmat, emit_probs_mat[:,t+1]*b_t)
        b_t /= s_t[t+1]  # previously t
        b_s[t,:] = b_t
    
    loglh = sp.log(sp.array(s_t)).sum()
    
    for t in range(T):
        Q[t,:] = a_s[t,:]* b_s[t,:]
        tmp2 = sp.dot(sp.dot(sp.diag(a_s[t-1,:]), transmat), sp.diag(emit_probs_mat[:,t]* b_s[t,:]))
        Q_pairs[t, :, :]  = tmp2/tmp2.sum()
    
    return loglh

def params_from_clique_marginals(theta, alpha, beta, gamma, emit_probs, clq_Q, clq_Q_pairs, X, args):
    """Recompute parameters using marginal probabilities"""
    I,T,L = X.shape
    K = gamma.shape[0]
    vp = 0
    pc = 1e-10 # pseudocount
    theta[:] = pc; alpha[:] = pc; beta[:] = pc; gamma[:] = pc; emit_probs[:] = pc
    #theta[:] = 0; alpha[:] = 0; beta[:] = 0; gamma[:] = 0; emit_probs[:] = 0
    global combinations
    combinations = list(enumerate(itertools.product(range(K), repeat=I)))
    #e_sum = sp.ones(emit_probs.shape[0]) * pc
    e_sum = sp.ones_like(emit_probs) * pc
    #e_sum = sp.ones(emit_probs.shape[1]) * 0
    t = 0
    for k_to, to_val in combinations:
        for i in xrange(I):
            for l in xrange(L):
                if X[i,t,l]:
                    emit_probs[to_val[i], l] += clq_Q[t, k_to]
                #e_sum[to_val[i]] += clq_Q[t, k_to]
                e_sum[to_val[i], l] += clq_Q[t, k_to]
        gamma[to_val[vp]] += clq_Q[t,k_to]
        for i in xrange(1, I):
            beta[to_val[vp], to_val[i]] += clq_Q[t,k_to]
    
    #import ipdb; ipdb.set_trace()
    for t in xrange(1, T):
        for k_to, to_val in combinations:
            for i in xrange(I):
                for l in xrange(L):
                    if X[i,t,l]:
                        #import ipdb; ipdb.set_trace()
                        emit_probs[to_val[i], l] += clq_Q[t, k_to]
                    #e_sum[to_val[i]] += clq_Q[t, k_to]
                    e_sum[to_val[i], l] += clq_Q[t, k_to]
            for k_from, from_val in combinations:    
                alpha[from_val[vp], to_val[vp]] += clq_Q_pairs[t,k_from, k_to]
                for i in xrange(1, I):
                    theta[to_val[vp], from_val[i], to_val[i]] += clq_Q_pairs[t,k_from, k_to]
    #import ipdb; ipdb.set_trace()
    
    #theta, alpha, beta, gamma = theta+pc*theta.max(), alpha+pc*alpha.max(),beta+pc*beta.max(),gamma+pc*gamma.max()
    normalize_trans(theta, alpha, beta, gamma)
    #emit_probs[:] = sp.dot(sp.diag(1./e_sum), emit_probs)
    #emit_probs[:] = sp.dot(emit_probs, sp.diag(1./e_sum * L))
    emit_probs[:] = emit_probs / e_sum
    #emit_probs[:] = sp.dot(emit_probs + pc*emit_probs.max(), sp.diag(1./(e_sum + pc*emit_probs.max())))
    args.emit_sum = e_sum


def clique_marginals_to_Q(clq_Q, Q):
    """Convert clique marginals (T*K**I) to Q marginals (I*T*K)"""
    I,T,K = Q.shape
    Q[:] = 0.
    combinations = list(enumerate(itertools.product(range(K), repeat=I)))
    for t in xrange(T):
        for k_to, to_val in combinations:
            for i in xrange(I):
                Q[i,t,to_val[i]] += clq_Q[t, k_to]
    
    
if __name__ == '__main__':
    main()
