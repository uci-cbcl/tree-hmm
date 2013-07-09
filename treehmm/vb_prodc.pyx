#!python
#cython: boundscheck=False, wraparound=False, cdivision=True
# can also add profile=True

"""Inference using product of chains"""

cimport cython
from cython.parallel import prange
import numpy as np
import time
import copy
from numpy import array, random, diag
cimport numpy as np
from libc.math cimport exp, log

from vb_mf import normalize_trans, normalize_emit, make_log_obs_matrix, make_log_obs_matrix_gaussian

#ctypedef np.float128_t float_type
#ctypedef long double float_type
ctypedef double float_type

cdef float_type min_val = np.double('1e-150')
#min_val **= 4  # 1e-400
min_val **= 1  # 1e-100
#cdef float_type min_val_log_space = np.log(min_val)  # -921.03


def prodc_initialize_qs(theta, alpha, beta, gamma, X, log_obs_mat):
    I, T, L = X.shape
    #I = 1
    K = alpha.shape[0]
    a_s = np.zeros((T,K), dtype=np.double)
    b_s = np.zeros((T,K), dtype=np.double)
    Q = np.zeros((I,T,K), dtype=np.double)
    Q_pairs = np.zeros((I,T,K,K), dtype=np.double)
    loglh = np.zeros(I, dtype=np.double)
    print 'initializing Q',

    for i in range(I):
        #emit_probs_mat = np.exp(log_emit_probs_i(emit_probs, X, i))
        emit_probs_mat = np.exp(log_obs_mat[i,:,:]).T
        if np.any(emit_probs_mat < min_val):
            print 'applying minimum probability to emit probs'
            emit_probs_mat[emit_probs_mat < min_val] = min_val
        if i == 0:
            transmat = alpha
        else:
            # TODO:  This should be i, not 0, right?
            transmat = theta[0,:,:]

        a_t = gamma * emit_probs_mat[:, 0]
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
#            a_t = array(random.random((10,))) * np.dot(a_t.T, transmat)
            a_s[t,:] = a_t

        #back-ward algorithm
        for t in range(T-2,-1,-1):
            b_t = np.dot(transmat, emit_probs_mat[:,t+1]*b_t)
            b_t /= s_t[t+1]  # previously t
#            b_t = np.dot(transmat, array(random.random((10,))) * b_t)
            #print t, tmp.shape, transmat.shape, b_t.shape, b_s.shape
            b_s[t,:] = b_t

        loglh[i] = np.log(array(s_t)).sum()
#        print a_s.shape, b_s.shape, loglh.shape
        #tmp1 = a_s[0,:]*b_s[0,:]
        #Q[i,0,:] = tmp1/tmp1.sum()
        Q[i,0,:] = a_s[0,:]*b_s[0,:]

        for t in range(1,T):
#            breakpoint()
            #tmp1 = a_s[t,:]* b_s[t,:]
            #Q[i,t,:] = tmp1/tmp1.sum()
            Q[i,t,:] = a_s[t,:]* b_s[t,:]
#            breakpoint()
            tmp2 = np.dot(np.dot(diag(a_s[t-1,:]), transmat), diag(emit_probs_mat[:,t]* b_s[t,:]))
            Q_pairs[i, t, :, :]  = tmp2/tmp2.sum()

    return Q, Q_pairs

def prodc_update_q(args):
    cdef int i = 0, I = args.I
    args.Q_prev = copy.deepcopy(args.Q)
    #for k in range(args.K):
    #    print 'means[%s,:] = ' % k, args.means[k,:]
    #    print 'variances[%s,:] = ' % k, args.variances[k,:]
    for i in xrange(I):
        #print 'updating Q distribution for i=', i

        #prodc_update_qs_i(i, args.theta, args.alpha, args.beta, args.gamma,
        #                  args.emit_probs, args.X, args.Q, args.Q_pairs,
        #                  args.vert_parent, args.vert_children)
        prodc_update_qs_i_new(i, args.theta, args.alpha, args.beta, args.gamma,
                          args.Q, args.Q_pairs,
                          args.vert_parent, args.vert_children, args.log_obs_mat, args)





def prodc_update_qs_i_new(int si, np.ndarray[float_type, ndim=3] theta,
                        np.ndarray[float_type, ndim=2] alpha,
                        np.ndarray[float_type, ndim=2] beta,
                        np.ndarray[float_type, ndim=1] gamma,
                        np.ndarray[float_type, ndim=3] Q,
                        np.ndarray[float_type, ndim=4] Q_pairs,
                        np.ndarray[np.int8_t, ndim=1] vert_parent,
                        vert_children,
                        np.ndarray[float_type, ndim=3] log_obs_mat, args):
    cdef:
        Py_ssize_t I = Q.shape[0], T = Q.shape[1], K = Q.shape[2]
        Py_ssize_t i,t,v,h,k, ch_i, vp, len_v_chs
        np.ndarray[float_type, ndim=1] log_gamma
        np.ndarray[float_type, ndim=2] log_alpha, log_beta
        np.ndarray[float_type, ndim=3] log_theta, transmat
        np.ndarray[float_type, ndim=1] g_t, log_g_t, f1, log_f1, b_t, a_t, s_t, prev_b_t, prev_a_t
        #np.ndarray[float_type, ndim=1] loglh
        np.ndarray[float_type, ndim=2] a_s, b_s, emit_probs_mat, log_f, f_t
        float_type tmpsum_q, tmpsum_qp, total_f1, g_t_max

    #cdef np.ndarray[float_type, ndim=3] X = args.X
    X = args.X.astype(np.double)


    # first calculate the transition matrices (all different for different t) for chain si
    print 'prodc_update_qs'
    a_s = np.zeros((T,K), dtype=np.double)
    b_s = np.zeros((T,K), dtype=np.double)
    transmat = np.zeros((T, K, K), dtype=np.double)
    #loglh = np.zeros(I)
    emit_probs_mat = np.exp(log_obs_mat[si,:,:]).T
    if np.any(emit_probs_mat < min_val):
        print 'applying minimum probability to emit probs'
        emit_probs_mat[emit_probs_mat < min_val] = min_val
    #emit_probs_mat = np.exp(log_emit_probs_i(emit_probs, X, si))
    #start from the last node
    np.seterr(under='ignore')
    log_theta = np.log(theta)
    log_alpha = np.log(alpha)
    log_beta = np.log(beta)
    log_gamma = np.log(gamma)
    for mat in [log_theta, log_alpha, log_beta, log_gamma]:
        mat[mat < -min_val] = -min_val
    np.seterr(under='print')

    g_t = np.ones(K, dtype=np.double) # initialize for last node
    log_g_t = np.log(g_t)
    tmpsum_q = 0.
    tmpsum_qp = 0.

    # TODO finish
    b_t = np.ones(K, dtype=np.double)
    b_s[T-1,:] = b_t
    a_t = np.zeros(K, dtype=np.double)
    s_t = np.zeros(T, dtype=np.double)
    f1 = np.zeros(K, dtype=np.double)
    log_f1 = np.zeros(K, dtype=np.double)
    f_t = np.zeros((K,K), dtype=np.double)
    log_f = np.zeros((K,K), dtype=np.double)
    total_f1 = 0.
    g_t_max = 0.
    prev_b_t = np.ones(K, dtype=np.double)
    prev_a_t = np.zeros(K, dtype=np.double)


    vp = vert_parent[si]
    # update transition along nodes
    for t in xrange(T-1,0,-1):
        #if si == 0:
        #    print '****i= %s t= %s' % (si, t)
        g_t_max = 0.
        for v in xrange(K):
            g_t[v] = 0.
        for k in xrange(K):
            #evidence term from marginalization of children
            q_sum_k_ch_i = 0.
            for ch_i in vert_children[si]:
                for v in xrange(K):
                    for h in xrange(K):
                        q_sum_k_ch_i += Q_pairs[ch_i, t, h, v] * log_theta[k, h, v]
            # all together
            for v in xrange(K):
                if si == 0:
                    log_f[v,k] = log_alpha[v,k] + log_g_t[k] + q_sum_k_ch_i
                else:
                    log_f[v,k] = log_g_t[k] + q_sum_k_ch_i
                    for h in xrange(K):
                        log_f[v,k] += Q[vp,t,h] * log_theta[h,v,k]
                #f_t[v,k] = exp(log_f[v,k])
                #g_t[v] += f_t[v,k]
        #print 'here', t
        #print 'log_f'
        for k in xrange(K):
            for v in xrange(K):
                #print log_f[k,v],
                f_t[v,k] = exp(log_f[v,k])
                g_t[v] += f_t[v,k]
        #if si == 0:
        #    print 'g_t', g_t
        #    print 'f_t', f_t
        #time.sleep(1)
        for k in xrange(K):
            if g_t_max < g_t[k]:
                g_t_max = g_t[k]
            for v in xrange(K):
                #transmat[t,v,k] = f_t[v,k] / g_t[k]
                transmat[t,v,k] = f_t[v,k] / g_t[v]
        #print 'g_t',
        #print g_t_max
        #print g_t
        #g_t /= g_t.max()
        for k in xrange(K):
            g_t[k] /= g_t_max
            #print g_t[k]
            log_g_t[k] = log(g_t[k])
        #print t, g_t

    # first node
    #print 'here2'
    for k in xrange(K):
        if si == 0:
            log_f1[k] = log_gamma[k] + log_g_t[k]
        else:
            log_f1[k] = 0.
            for v in xrange(K):
                log_f1[k] += Q[vp,0,v] * log_beta[v,k] + log_g_t[k]
                for ch_i in vert_children[si]:
                    log_f1[k] += log_beta[k,v] * Q[ch_i,0,v]
        f1[k] = exp(log_f1[k])
        total_f1 += f1[k]
    #print 'here3'
    for k in xrange(K):
        f1[k] /= total_f1

    #print 'f1', f1
    #print 'transmat', transmat

    # update Q, Q_pairs, lh
    for k in xrange(K):
        a_t[k] = f1[k] * emit_probs_mat[k,0]
        s_t[0] += a_t[k]
    #print s_t
    for k in xrange(K):
        a_t[k] /= s_t[0]
        a_s[0,k] = a_t[k]

    # forward algorithm
    #for t in xrange(1,T):
    #    for k in xrange(K):
    #        prev_a_t[k] = a_t[k]
    #    for k in xrange(K):
    #        a_t[k] = 0.
    #        for v in xrange(K):
    #            a_t[k] += prev_a_t[v] * transmat[t,v,k]
    #        a_t[k] *= emit_probs_mat[k, t]
    #        s_t[t] += a_t[k]
    #    for k in xrange(K):
    #        a_t[k] /= s_t[t]
    #        a_s[t,k] = a_t[k]
    for t in xrange(1,T):
#        breakpoint()        *#*#*#*#
        a_t = emit_probs_mat[:, t] * (np.dot(a_t.T, transmat[t,:,:]))
        s_t[t] = a_t.sum()
        a_t /= s_t[t]
        a_s[t,:] = a_t

        #print t, a_s[t,:]
    #print 'forward done, waiting...'
    #time.sleep(3)
    #print a_s

    #backward algorithm
    for t in xrange(T-2,-1,-1):
        #for k in xrange(K):
        #    prev_b_t[k] = b_t[k]
        #for k in xrange(K):
        #    b_t[k] = 0.
        #    for v in xrange(K):
        #        #b_t[k] += transmat[t+1,k,v] * emit_probs_mat[v,t+1] * prev_b_t[k]
        #        b_t[k] += transmat[t+1,k,v] * emit_probs_mat[v,t+1] * prev_b_t[v]
        b_t = np.dot(transmat[t+1,:,:], emit_probs_mat[:,t+1]*b_t)
        for k in xrange(K):
            b_t[k] /= s_t[t+1]
            b_s[t,k] = b_t[k]
    ##backward algorithm
    #for t in range(T-2,-1,-1):
    #    b_t = sp.dot(transmat[t+1,:,:], emit_probs_mat[:,t+1]*b_t)
    #    b_t /= s_t[t+1]
    #    b_s[t,:] = b_t
    #print 'b_s', b_s

        #print t, b_t, s_t[t]
        #print b_s[t,:]

    #global vb_prodc_s_t
    #vb_prodc_s_t = s_t
    #loglh[si] = np.log(s_t).sum()
    #print 'a_s', a_s
    #print 'b_s', b_s
    #print b_s

    #print 'likelihood', loglh[si]

    #tmp1 = a_s[0,:] * b_s[0,:]
    #Q[si,0,:] = tmp1/tmp1.sum()
    #for t in xrange(1,T):
    #    tmpsum_q = 0.
    #    tmpsum_qp = 0.
    #    for k in xrange(K):
    #        Q[si,t,k] = a_s[t,k] * b_s[t,k]
    #        tmpsum_q += Q[si,t,k]
    #    for k in xrange(K):
    #        Q[si,t,k] /= tmpsum_q
    #        for v in xrange(K):
    #            # TODO: make sure the two dots are right
    #            #Q_pairs[si, t, k, v] += a_s[t-1,v] * transmat[t,v,k] * emit_probs_mat[k,t] * b_s[t,k]
    #            Q_pairs[si, t, k, v] += a_s[t-1,v] * transmat[t,v,k] * emit_probs_mat[k,t] * b_s[t,k]
    #            tmpsum_qp += Q_pairs[si, t, k, v]
    #    for k in xrange(K):
    #        for v in xrange(K):
    #            Q_pairs[si, t, k, v] /= tmpsum_qp
    tmp1 = a_s[0,:] * b_s[0,:]
    Q[si,0,:] = tmp1/tmp1.sum()
    for t in range(1,T):
        tmp1 = a_s[t,:]* b_s[t,:]
        Q[si, t,:] = tmp1/tmp1.sum()
        tmp2 = np.dot(np.dot(diag(a_s[t-1,:]), transmat[t]), diag(emit_probs_mat[:,t]* b_s[t,:]))
        Q_pairs[si, t]  = tmp2/tmp2.sum()

    if np.any(Q < min_val):
        print 'fixing Q... values too low'
        Q[Q < min_val] = min_val
    if np.any(Q_pairs < min_val):
        print 'fixing Q_pairs... values too low'
        Q_pairs[Q_pairs < min_val] = min_val
    print 'done'




def prodc_update_params(args, renormalize=True):
    cdef:
        #np.ndarray[np.int8_t, ndim=3] X
        np.ndarray[float_type, ndim=3] Q
        np.ndarray[float_type, ndim=4] Q_pairs
        np.ndarray[float_type, ndim=3] theta
        np.ndarray[float_type, ndim=2] alpha, beta, emit_probs
        np.ndarray[float_type, ndim=1] gamma
        np.ndarray[np.int8_t, ndim=1] vert_parent
        np.ndarray[float_type, ndim=3] log_obs_mat
        float_type pseudocount
    X = args.X
    Q, Q_pairs, theta, alpha, beta, gamma, vert_parent, vert_children, log_obs_mat, pseudocount = (args.Q, args.Q_pairs, args.theta,
                                                   args.alpha, args.beta,
                                                   args.gamma, args.vert_parent, args.vert_children, args.log_obs_mat, args.pseudocount)
    mark_avail = args.mark_avail
    cdef int I = Q.shape[0], T = Q.shape[1], K = Q.shape[2]
    cdef int L = X.shape[2]
    cdef Py_ssize_t i,t,v,h,k,vp,l
    if args.continuous_observations:
        new_means = np.zeros_like(args.means)
        new_variances = np.zeros_like(args.variances)
        total_q = np.zeros_like(args.variances)
    else:
        emit_probs = args.emit_probs
        emit_probs[:] = pseudocount
    #print 'mf_update_params'
    theta[:] = pseudocount
    alpha[:] = pseudocount
    beta[:] = pseudocount
    gamma[:] = pseudocount
    emit_probs[:] = pseudocount
    for i in xrange(I):
    #for i in prange(I, nogil=True):
        vp = vert_parent[i]
        for t in xrange(T):
            for k in xrange(K):
                if i==0 and t==0:
                    gamma[k] += Q[i, t, k]
                else:
                    for v in xrange(K):
                        if t == 0:
                            beta[v,k] += Q[vp,t,v] * Q[i,t,k]
                        elif i == 0:
                            alpha[v,k] += Q_pairs[i,t,v,k]
                        else:
                            for h in xrange(K):
                                theta[v,h,k] += Q[vp,t,v] * Q_pairs[i,t,h,k]
                if not args.continuous_observations:
                    real_i = args.real_species_i if args.real_species_i is not None else i
                    for l in xrange(L):
                        if mark_avail[real_i,l] and X[i,t,l]:
                            emit_probs[k, l] += Q[i, t, k]
    #Q[:,:,:] = .0001
    #Q[:,:,0] = .9999
    if args.continuous_observations:
        for i in xrange(I):
            for t in xrange(T):
                for k in xrange(K):
                    for l in xrange(L):
                        new_means[k,l] += Q[i, t, k] * X[i,t,l]  # expectation of X wrt Q
                        total_q[k,l] += Q[i,t,k]
        print 'is this it 1'
        args.means[:] = new_means = new_means / total_q + 1e-50
        print 'done'
        #print 'updated means'
        #for k in range(args.K):
        #    print 'means[%s,:] = ' % k, args.means[k,:]
        np.seterr(under='ignore')
        for i in xrange(I):
            for t in xrange(T):
                for k in xrange(K):
                    for l in xrange(L):
                        #print 'X[%s,%s,%s] = %s' % (i,t,l, X[i,t,l]), 'new_means[%s,%s] = %s', new_means[k,l], 'Q = ', Q[i,t,k]
                        new_variances[k,l] += Q[i, t, k] * (X[i,t,l] - new_means[k,l]) * (X[i,t,l] - new_means[k,l])
        np.seterr(under='print')
        #print 'var before renormalize:', new_variances
        #print 'total_q', total_q
        #if np.any(Q < 0 | Q > 1):
        #    raise RuntimeError("Bug here!")
        ##args.variances[:] = new_variances / (I * T) + pseudocount # 1 / N
        print 'is this it 2'
        args.variances[:] = new_variances / total_q  # 1 / N_k
        args.variances += pseudocount
        print 'done'
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

#cpdef float_type prodc_free_energy(args):
def prodc_free_energy(args):
    """Calculate the free energy for Q"""
    #cdef np.int8_t[:,:,:] X
    cdef np.ndarray[float_type, ndim=3] Q, theta
    cdef float_type[:,:,:,:] Q_pairs
    cdef float_type[:,:] alpha, beta
    cdef float_type[:] gamma
    cdef np.int8_t[:] vert_parent
    cdef float_type[:,:,:] log_obs_mat
    X, Q, Q_pairs, theta, alpha, beta, gamma, vert_parent, vert_children, log_obs_mat = (args.X, args.Q, args.Q_pairs, args.theta,
                                                   args.alpha, args.beta,
                                                   args.gamma, args.vert_parent, args.vert_children, args.log_obs_mat)
    cdef Py_ssize_t I = Q.shape[0], T = Q.shape[1], K = Q.shape[2]
    cdef Py_ssize_t i,t,v,h,k, ch_i, vp, len_v_chs
    cdef float_type[:] log_gamma
    cdef float_type[:,:] log_alpha, log_beta
    cdef float_type[:,:,:] log_theta
    cdef float_type val = 0
    #print 'mf_free_energy'
    log_theta = np.log(theta)
    log_alpha = np.log(alpha)
    log_beta = np.log(beta)
    log_gamma = np.log(gamma)

    # Q * logQ
    cdef float_type total_free = 0., q_cond = 0.
    np.seterr(under='ignore')
    for i in xrange(I):
    #for i in prange(I, nogil=True):
        for k in xrange(K):
            total_free += Q[i,0,k] * log(Q[i,0,k])  # first column's entropy
            for t in xrange(1,T):
                for v in xrange(K):
                    q_cond = Q_pairs[i,t,k,v] / Q[i,t-1,k]
                    val = Q_pairs[i,t,k,v] * log(q_cond)
                    total_free += val if np.isfinite(val) else 0.
    np.seterr(under='print')
    #total_free = 0
    #for i in xrange(I):
    #    #calculate Q* log(Q)
    #    total_free += np.dot(Q[i, 0,:], np.log(Q[i, 0,:]))
    #    for t in xrange(1, T):
    #        q_cond = np.dot(diag(1./Q[i,t-1,:]), Q_pairs[i, t])
    #        #if (q_cond - 1. > epsilon).any(): #abs(q_cond.sum(axis=1) -1.).any() > epsilon:
    #        #    breakpoint()
    #        #    print q_cond
    #
    #        total_free += (Q_pairs[i, t] * np.log(q_cond)).sum()

    # calculate - Q* log(P)
    for i in xrange(I):
    #for i in prange(I, nogil=True):
        vp = vert_parent[i]
        #for t in prange(T, nogil=True):
        for t in xrange(T):
            for k in xrange(K):
                total_free -= Q[i,t,k] * log_obs_mat[i,t,k]
                if i == 0 and t == 0:
                    total_free -= Q[i,t,k] * log_gamma[k]
                else:
                    for v in xrange(K):
                        if i == 0:
                            total_free -= Q_pairs[i,t,v,k] * log_alpha[v,k]
                        elif t == 0:
                            total_free -= Q[vp,t,v] * Q[i,t,k] * log_beta[v,k]
                        else:
                            for h in xrange(K):
                                total_free -= Q[vp,t,v] * Q_pairs[i,t,h,k] * log_theta[v,h,k]
    #prodc_free_energy_old(args)
    return total_free
