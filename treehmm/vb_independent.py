#!python

import numpy as np
import time
import copy
from numpy import array, random, diag

from vb_mf import normalize_trans, normalize_emit, make_log_obs_matrix, make_log_obs_matrix_gaussian
from treehmm.static import float_type

min_val = float_type('1e-150')


def independent_update_qs(args):
    theta, alpha, beta, gamma, X, log_obs_mat, Q, Q_pairs = args.theta, args.alpha, args.beta, args.gamma, args.X, args.log_obs_mat, args.Q, args.Q_pairs
    I, T, L = X.shape
    K = alpha.shape[0]
    a_s = np.zeros((T, K), dtype=float_type)
    b_s = np.zeros((T, K), dtype=float_type)
    Q[:] = np.zeros((I, T, K), dtype=float_type)
    Q_pairs[:] = np.zeros((I, T, K, K), dtype=float_type)
    loglh = np.zeros(I, dtype=float_type)
    print 'initializing Q',

    for i in range(I):
        emit_probs_mat = np.exp(log_obs_mat[i, :, :]).T
        if np.any(emit_probs_mat < min_val):
            print 'applying minimum probability to emit probs'
            emit_probs_mat[emit_probs_mat < min_val] = min_val

        transmat = alpha
        a_t = gamma * emit_probs_mat[:, 0]
        s_t = [a_t.sum()]
        a_t /= s_t[0]
        b_t = np.ones((K, ))

        a_s[0, :] = a_t
        b_s[T - 1, :] = b_t

        # forward algorithm
        for t in range(1, T):
            a_t = emit_probs_mat[:, t] * np.dot(a_t.T, transmat)
            s_t.append(a_t.sum())
            a_t /= s_t[t]
            a_s[t, :] = a_t

        #back-ward algorithm
        for t in range(T - 2, -1, -1):
            b_t = np.dot(transmat, emit_probs_mat[:, t+1] * b_t)
            b_t /= s_t[t + 1]  # previously t
            b_s[t,:] = b_t

        loglh[i] = np.log(array(s_t)).sum()

        for t in range(1,T):
            tmp1 = a_s[t, :] * b_s[t, :]
            Q[i, t, :] = tmp1/tmp1.sum()
            tmp2 = np.dot(np.dot(diag(a_s[t-1,:]), transmat), diag(emit_probs_mat[:,t]* b_s[t,:]))
            Q_pairs[i, t, :, :]  = tmp2/tmp2.sum()

        if np.any(Q < min_val):
            print 'fixing Q... values too low'
            Q[Q < min_val] = min_val
        print 'done'


def independent_update_params(args, renormalize=True):
    X = args.X
    Q, Q_pairs, theta, alpha, beta, gamma, vert_parent, vert_children, log_obs_mat, pseudocount = (
                                                   args.Q, args.Q_pairs, args.theta,
                                                   args.alpha, args.beta,
                                                   args.gamma, args.vert_parent, args.vert_children, args.log_obs_mat, args.pseudocount)
    I, T, K = Q.shape
    L = X.shape[2]

    if args.continuous_observations:
        new_means = np.zeros_like(args.means)
        new_variances = np.zeros_like(args.variances)
        total_q = np.zeros_like(args.variances)
    else:
        emit_probs = args.emit_probs
        emit_probs[:] = pseudocount

    theta[:] = pseudocount
    alpha[:] = pseudocount
    beta[:] = pseudocount
    gamma[:] = pseudocount


    for i in xrange(I):
        vp = vert_parent[i]
        for t in xrange(T):
            for k in xrange(K):
                if i==0 and t==0:
                    gamma[k] += Q[i, t, k]
                else:
                    for v in xrange(K):
                        if t == 0:
                            beta[v,k] += Q[vp,t,v] * Q[i,t,k]
                        else:
                            alpha[v,k] += Q_pairs[i,t,v,k]
                if not args.continuous_observations:
                    for l in xrange(L):
                        if X[i,t,l]:
                            emit_probs[k, l] += Q[i, t, k]
    if args.continuous_observations:
        for i in xrange(I):
            for t in xrange(T):
                for k in xrange(K):
                    for l in xrange(L):
                        new_means[k,l] += Q[i, t, k] * X[i,t,l]  # expectation of X wrt Q
                        total_q[k,l] += Q[i,t,k]

        args.means[:] = new_means = new_means / total_q + 1e-50

        np.seterr(under='ignore')
        for i in xrange(I):
            for t in xrange(T):
                for k in xrange(K):
                    for l in xrange(L):
                        new_variances[k,l] += Q[i, t, k] * (X[i,t,l] - new_means[k,l]) * (X[i,t,l] - new_means[k,l])
        np.seterr(under='print')
        args.variances[:] = new_variances / total_q  # 1 / N_k
        args.variances += pseudocount
    else:
        normalize_emit(Q, emit_probs, pseudocount, args, renormalize)

    if renormalize:
        theta += theta.max() * (pseudocount * 1e-20)
        alpha += alpha.max() * (pseudocount * 1e-20)
        beta += beta.max() * (pseudocount * 1e-20)
        gamma += gamma.max() * (pseudocount * 1e-20)
        normalize_trans(theta, alpha, beta, gamma)

    if args.continuous_observations:
        make_log_obs_matrix_gaussian(args)
    else:
        make_log_obs_matrix(args)


def independent_free_energy(args):
    """Calculate the free energy for Q"""
    I, T, L = args.X.shape
    K = args.alpha.shape[0]
    a_s = np.zeros((T,K), dtype=float_type)
    b_s = np.zeros((T,K), dtype=float_type)
    loglh = np.zeros(I, dtype=float_type)
    transmat = args.alpha

    for i in range(I):
        emit_probs_mat = np.exp(args.log_obs_mat[i,:,:]).T
        if np.any(emit_probs_mat < min_val):
            print 'applying minimum probability to emit probs'
            emit_probs_mat[emit_probs_mat < min_val] = min_val

        a_t = args.gamma * emit_probs_mat[:, 0]
        s_t = [a_t.sum()]
        a_t /= s_t[0]
        b_t = np.ones((K,))

        a_s[0,:] = a_t
        b_s[T-1,:] = b_t

        # forward algorithm
        for t in range(1,T):
            a_t = emit_probs_mat[:,t] * np.dot(a_t.T, transmat)
            s_t.append(a_t.sum())
            a_t /= s_t[t]
            a_s[t,:] = a_t

        loglh[i] = np.log(array(s_t)).sum()
    return loglh.sum()
