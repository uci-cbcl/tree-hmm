#!python
#cython: boundscheck=False, wraparound=False, cdivision=True
# can also add profile=True

import os
import itertools
import scipy as sp
import numpy as np
cimport numpy as np
import copy

from vb_mf import normalize_trans
DTYPE = np.int

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
    clique_init_probs_from_params(args.clq_init, args.beta, args.gamma, args.I, args)
    clique_transmission_from_params(args.clq_trans, args.alpha, args.theta, args.I, args.vert_parent)
    clique_emit_from_params(args.clq_emit, args.emit_probs, args.I, args.X, args.mark_avail)

    print '# inferring clique hidden states'
    args.clq_loglh = infer_hidden_marginals(args.clq_Q, args.clq_Q_pairs, args.clq_init, args.clq_trans, args.clq_emit)
    print 'log likelihood is', args.clq_loglh

    print '# uncliqueing Q'
    clique_marginals_to_Q(args.clq_Q, args.Q)

def clique_update_params(args, renormalize=True):
    """update parameters after cliqued hmm"""
    print '# updating params from Q'
    params_from_clique_marginals(args.theta, args.alpha, args.beta, args.gamma, args.emit_probs, args.emit_sum, args.clq_Q, args.clq_Q_pairs, args.X, args.mark_avail, args.vert_parent, renormalize)


def clique_init_probs_from_params(clq_init, beta, gamma, I, args):
    """Convert beta (K*K) and gamma (K) matrices into a K**I init_probs distribution"""
    K = gamma.shape[0]
    combinations = list(enumerate(itertools.product(range(K), repeat=I)))
    clq_init[:] = 1.
    vp = args.vert_parent
    for k_from, from_val in combinations:
        clq_init[k_from] *= gamma[from_val[0]]
        for index in xrange(1, I):
            clq_init[k_from] *= beta[from_val[vp[index]], from_val[index]]

cpdef clique_transmission_from_params(np.ndarray[np.float64_t, ndim=2] clq_trans,
                                    np.ndarray[np.float64_t, ndim=2] alpha, 
                                    np.ndarray[np.float64_t, ndim=3] theta, np.int_t I,
                                    np.ndarray[np.int8_t, ndim=1] vp):
    """Convert an alpha (K*K) matrix into a cliqued transition matrix (K^I * K^I)"""
    cdef:
        Py_ssize_t K=alpha.shape[0]
        Py_ssize_t i, i2, t, l, k, k_from, k_to, tmp, tmp2, index
        np.ndarray[np.int_t, ndim=1] from_val, to_val
        
    from_val = np.ones(I, dtype=np.int)
    to_val = np.ones(I, dtype=np.int)
        
    clq_trans[:] = 1.
    #vp = args.vert_parent
    # each species can be in any of K states; moving from one of these combinations to one of these combinations
    #combinations = list(enumerate(itertools.product(range(K), repeat=I)))
    #for ((k_from, from_val), (k_to, to_val)) in itertools.product(combinations, repeat=2):
    
    for k_from in xrange(K**I):
        tmp = k_from
        for i in xrange(I):
            from_val[i] = tmp//(K**(I-i-1))
            tmp = tmp % K**(I-i-1)
        
        for k_to in xrange(K**I):
            tmp2 = k_to
            for i2 in xrange(I):
                to_val[i2] = tmp2//(K**(I-i2-1))
                tmp2 = tmp2 % K**(I-i2-1)
        
            clq_trans[k_from, k_to] *= alpha[from_val[0], to_val[0]]
            for index in xrange(1, I):
                clq_trans[k_from, k_to] *= theta[to_val[vp[index]], from_val[index], to_val[index]]

cpdef clique_emit_from_params(np.ndarray[np.float64_t, ndim=2] clq_emit,
                    np.ndarray[np.float64_t, ndim=2] emit_probs, np.int_t I,
                    np.ndarray[np.int8_t, ndim=3] X,
                    np.ndarray[np.int8_t, ndim=2] mark_avail):
    """create a T x K**I matrix from a K*L emission matrix and also bind to the values of X"""
    
    cdef:
        Py_ssize_t T=X.shape[1], L=X.shape[2], K=emit_probs.shape[0]
        Py_ssize_t i, t, l, k, k_from, tmp
        np.ndarray[np.int_t, ndim=1] from_val
        
    from_val = np.ones(I, dtype=np.int)
#    I=X.shape[0]
#    combinations = list(enumerate(itertools.product(range(K), repeat=I)))
    clq_emit[:] = 1.
    # calculate the probability of observing X
    for t in xrange(T):
        #for k_from, from_val in combinations:
        for k_from in xrange(K**I):
            tmp = k_from
            for i in xrange(I):
                from_val[i] = tmp//(K**(I-i-1))
                tmp = tmp% K**(I-i-1)
            
            #for i, k in enumerate(from_val):
            for i in xrange(I):
                k = from_val[i]
                for l in xrange(L):
                    #clq_emit[k_from,t] *= emit_probs[k, l]
                    if mark_avail[i,l]:
                        if X[i,t,l]:
                            clq_emit[k_from,t] *= emit_probs[k, l]
                        else:
                            clq_emit[k_from,t] *= 1. - emit_probs[k, l]

cdef clique_emit_1d_from_params(np.ndarray[np.float64_t, ndim=2] clq_emit_no_t,
                            np.ndarray[np.float64_t, ndim=2] emit_probs, np.int_t I,
                            np.ndarray[np.int8_t, ndim=3] X):
    """create a K**I x L matrix from a K*L emission matrix without binding the values of X"""
    cdef:
        Py_ssize_t T=X.shape[1], L=X.shape[2], K=emit_probs.shape[0]
        Py_ssize_t i, t, l, k, k_from, tmp
        from_val = np.ones(I, dtype=np.int)
        
    I=X.shape[0]
    #combinations = list(enumerate(itertools.product(range(K), repeat=I)))
    clq_emit_no_t[:] = 1.
    # calculate the probability of observing X
    for k_from in xrange(K**I):
        tmp = k_from
        for i in xrange(I):
           from_val[i] = tmp//(K**(I-i-1))
           tmp = tmp% K**(I-i-1)
        for i in xrange(I):
            k = from_val[i]
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
    if sp.iscomplex(emit_probs_mat).any():
        raise Exception("Complex")
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

cpdef params_from_clique_marginals(np.ndarray[np.float64_t, ndim=3] theta,
                        np.ndarray[np.float64_t, ndim=2] alpha,
                        np.ndarray[np.float64_t, ndim=2] beta,
                        np.ndarray[np.float64_t, ndim=1] gamma,
                        np.ndarray[np.float64_t, ndim=2] emit_probs,
                        np.ndarray[np.float64_t, ndim=2] e_sum,
                        np.ndarray[np.float64_t, ndim=2] clq_Q,
                        np.ndarray[np.float64_t, ndim=3] clq_Q_pairs,
                        np.ndarray[np.int8_t, ndim=3] X,
                        np.ndarray[np.int8_t, ndim=2] mark_avail,
                        np.ndarray[np.int8_t, ndim=1] vert_parent,
                        np.int8_t renormalize):
    """Recompute parameters using marginal probabilities"""
    cdef:
        Py_ssize_t I=X.shape[0], T=X.shape[1], L=X.shape[2], K=gamma.shape[0]
        Py_ssize_t i, t, l, k, k_to, k_from, tmp
        np.float64_t pc = 1e-10
        np.ndarray[np.int8_t, ndim=1] vp
        np.ndarray[np.int_t, ndim=1] to_val, from_val
        
        #np.ndarray[np.float64_t, ndim=2] e_sum = np.ones_like(emit_probs) * pc
       
    vp = vert_parent
    to_val = np.ones(I, dtype=np.int)
    from_val = np.ones(I, dtype=np.int)
    theta[:] = pc; alpha[:] = pc; beta[:] = pc; gamma[:] = pc; emit_probs[:] = pc; e_sum[:]=pc

    #theta[:] = 0; alpha[:] = 0; beta[:] = 0; gamma[:] = 0; emit_probs[:] = 0
    #global combinations
    #combinations = list(enumerate(itertools.product(range(K), repeat=I)))
    #e_sum = sp.ones(emit_probs.shape[0]) * pc
    #e_sum = sp.ones_like(emit_probs) * pc
    
    t = 0
    for k_to in xrange(K**I):
        tmp = k_to
        for i in xrange(I):
            to_val[i] = tmp//(K**(I-i-1))
            tmp = tmp%K**(I-i-1)
        #print to_val
        for i in xrange(I):
            for l in xrange(L):
                if mark_avail[i,l]:
                    if X[i,t,l]:
                        emit_probs[to_val[i], l] += clq_Q[t, k_to]
                    #e_sum[to_val[i]] += clq_Q[t, k_to]
                    e_sum[to_val[i], l] += clq_Q[t, k_to]
        gamma[to_val[0]] += clq_Q[t,k_to]
        for i in xrange(1, I):
            beta[to_val[vp[i]], to_val[i]] += clq_Q[t,k_to]

    #import ipdb; ipdb.set_trace()
    for t in xrange(1, T):
        for k_to in xrange(K**I):
            tmp = k_to
            for i in xrange(I):
                to_val[i] = tmp//(K**(I-i-1))
                tmp = tmp% K**(I-i-1)

            for i in xrange(I):
                for l in xrange(L):
                    if mark_avail[i,l]:
                        if X[i,t,l]:
                            #import ipdb; ipdb.set_trace()
                            emit_probs[to_val[i], l] += clq_Q[t, k_to]
                        #e_sum[to_val[i]] += clq_Q[t, k_to]
                        e_sum[to_val[i], l] += clq_Q[t, k_to]
            for k_from in xrange(K**I):
                tmp = k_from
                for i in xrange(I):
                    from_val[i] = tmp//(K**(I-i-1))
                    tmp = tmp% K**(I-i-1)
                
                alpha[from_val[0], to_val[0]] += clq_Q_pairs[t,k_from, k_to]
                for i in xrange(1, I):
                    theta[to_val[vp[i]], from_val[i], to_val[i]] += clq_Q_pairs[t,k_from, k_to]

    if renormalize:
        normalize_trans(theta, alpha, beta, gamma)
        #emit_probs[:] = np.dot(sp.diag(1./e_sum), emit_probs)
        #emit_probs[:] = np.dot(emit_probs, sp.diag(1./e_sum * L))
        for k in xrange(K):
            for l in xrange(L):
                emit_probs[k,l] = emit_probs[k,l] / e_sum[k,l]
    #emit_probs[:] = np.dot(emit_probs + pc*emit_probs.max(), np.diag(1./(e_sum + pc*emit_probs.max())))
    #args.emit_sum = e_sum
    
def clique_marginals_to_Q(clq_Q, Q):
    """Convert clique marginals (T*K**I) to Q marginals (I*T*K)"""
    I,T,K = Q.shape
    Q[:] = 0.
    combinations = list(enumerate(itertools.product(range(K), repeat=I)))
    for t in xrange(T):
        for k_to, to_val in combinations:
            for i in xrange(I):
                Q[i,t,to_val[i]] += clq_Q[t, k_to]


#if __name__ == '__main__':
#    main()
