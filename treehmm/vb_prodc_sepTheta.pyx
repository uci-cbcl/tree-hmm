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

from vb_mf import normalize_trans, normalize_emit, make_log_obs_matrix

#implementation of seperate theta matrix for each species
# theta:
#previously K*K*K
# now      (I-1)*K*K*K

def prodc_initialize_qs(theta, alpha, beta, gamma, emit_probs, X, log_obs_mat):
    I, T, L = X.shape
    #I = 1
    K = alpha.shape[0]
    a_s = np.zeros((T,K)) 
    b_s = np.zeros((T,K))
    Q = np.zeros((I,T,K), dtype=np.float64) 
    Q_pairs = np.zeros((I,T,K,K), dtype=np.float64)
    loglh = np.zeros(I)
    
    for i in range(I):
        #emit_probs_mat = np.exp(log_emit_probs_i(emit_probs, X, i))
        emit_probs_mat = np.exp(log_obs_mat[i,:,:]).T
        if i == 0:
            transmat = alpha
        else:
            # TODO:  This should be i, not 0, right?
            transmat = theta[i-1,0,:,:]

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

        #backward algorithm        
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
    for i in xrange(I):
        #prodc_update_qs_i(i, args.theta, args.alpha, args.beta, args.gamma,
        #                  args.emit_probs, args.X, args.Q, args.Q_pairs,
        #                  args.vert_parent, args.vert_children)
        prodc_update_qs_i_new(i, args.theta, args.alpha, args.beta, args.gamma,
                          args.emit_probs, args.X, args.Q, args.Q_pairs,
                          args.vert_parent, args.vert_children, args.log_obs_mat)




cpdef prodc_update_qs_i_new(int si, np.ndarray[np.float64_t, ndim=4] theta,
                        np.ndarray[np.float64_t, ndim=2] alpha,
                        np.ndarray[np.float64_t, ndim=2] beta,
                        np.ndarray[np.float64_t, ndim=1] gamma,
                        np.ndarray[np.float64_t, ndim=2] emit_probs,
                        np.ndarray[np.int8_t, ndim=3] X,
                        np.ndarray[np.float64_t, ndim=3] Q,
                        np.ndarray[np.float64_t, ndim=4] Q_pairs,
                        np.ndarray[np.int8_t, ndim=1] vert_parent,
                        vert_children,
                        np.ndarray[np.float64_t, ndim=3] log_obs_mat):
    cdef:
        Py_ssize_t I = Q.shape[0], T = Q.shape[1], K = Q.shape[2]
        Py_ssize_t i,t,v,h,k, ch_i, vp, len_v_chs
        np.ndarray[np.float64_t, ndim=1] log_gamma
        np.ndarray[np.float64_t, ndim=2] log_alpha, log_beta
        np.ndarray[np.float64_t, ndim=3] transmat
        np.ndarray[np.float64_t, ndim=4] log_theta
        np.ndarray[np.float64_t, ndim=1] g_t, log_g_t, f1, log_f1, b_t, a_t, s_t, prev_b_t, prev_a_t
        #np.ndarray[np.float64_t, ndim=1] loglh
        np.ndarray[np.float64_t, ndim=2] a_s, b_s, emit_probs_mat, log_f, f_t
        np.float64_t tmpsum_q, tmpsum_qp, total_f1, g_t_max
    
    # first calculate the transition matrices (all different for different t) for chain si
    #print 'prodc_update_qs'
    a_s = np.zeros((T,K), dtype=np.float64)
    b_s = np.zeros((T,K), dtype=np.float64)
    transmat = np.zeros((T, K, K), dtype=np.float64)
    #loglh = np.zeros(I)
    emit_probs_mat = np.exp(log_obs_mat[si,:,:]).T
    #emit_probs_mat = np.exp(log_emit_probs_i(emit_probs, X, si))
    #start from the last node
    log_theta = np.log(theta)
    log_alpha = np.log(alpha)
    log_beta = np.log(beta)
    log_gamma = np.log(gamma)
    
    g_t = np.ones(K, dtype=np.float64) # initialize for last node
    log_g_t = np.log(g_t)
    tmpsum_q = 0.
    tmpsum_qp = 0.
    
    # TODO finish
    b_t = np.ones(K)
    b_s[T-1,:] = b_t
    a_t = np.zeros(K, dtype=np.float64)
    s_t = np.zeros(T, dtype=np.float64)
    f1 = np.zeros(K, dtype=np.float64)
    log_f1 = np.zeros(K, dtype=np.float64)
    f_t = np.zeros((K,K), dtype=np.float64)
    log_f = np.zeros((K,K), dtype=np.float64)
    total_f1 = 0.
    g_t_max = 0.
    prev_b_t = np.ones(K, dtype=np.float64)
    prev_a_t = np.zeros(K, dtype=np.float64)


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
                        q_sum_k_ch_i += Q_pairs[ch_i, t, h, v] * log_theta[ch_i-1, k, h, v]
            # all together
            for v in xrange(K):
                if si == 0:
                    log_f[v,k] = log_alpha[v,k] + log_g_t[k] + q_sum_k_ch_i
                else:
                    log_f[v,k] = log_g_t[k] + q_sum_k_ch_i
                    for h in xrange(K):
                        log_f[v,k] += Q[vp,t,h] * log_theta[si-1,h,v,k]
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
        Q_pairs[si, t] = tmp2/tmp2.sum()        


#def prodc_update_qs_i(si, theta, alpha, beta, gamma, emit_probs, X, Q, Q_pairs, vert_parent, vert_children):
#    # first calculate the transition matrices (all different for different t) for chain si
#    I, T, L = X.shape
#    #I = 1
#    K = alpha.shape[0]
#    #print 'prodc_update_qs'
#    a_s = np.zeros((T,K)) 
#    b_s = np.zeros((T,K))
#    transmat = np.zeros((T, K, K))
#    loglh = np.zeros(I)
#    emit_probs_mat = np.exp(log_emit_probs_i(emit_probs, X, si))
#    #start from the last node
#    
#    g_t = np.ones((K,)) # initialize for last node
#    if si == 0:
#        for t in range(T-1,0,-1):
#            log_f = np.log(alpha)
#            
#            # possibly need to check if this is leaf node
#            for k in range(K):
#                log_f[:,k] += np.log(g_t[k])
#                for ch_i in vert_children[si]:
#                    log_f[:,k] += (Q_pairs[ch_i,t,:,:] * np.log(theta[k,:,:])).sum() # a number
#            
#            #breakpoint()        
#            f_t = np.exp(log_f)
#            g_t = f_t.sum(axis=1)
#            transmat[t,:,:] = np.dot(diag(1./g_t), f_t)
#            g_t /= max(g_t)
#            #if abs(transmat[t,1].sum()-1) > 1E-6:
#                #breakpoint()
#                
#        # first node
#        log_f1 = np.log(gamma)+np.log(g_t)
#        for ch_i in vert_children[si]:          
#            log_f1 += np.dot(np.log(beta), Q[ch_i,0,:])
#            
#        tmp = np.exp(log_f1)
#        f1 = tmp/tmp.sum()
#        
#        
#    else:
#        vp = vert_parent[si]
#        for t in range(T-1,0,-1):
##            log_f = np.dot(Q[vp, t, :], np.log(theta))
#            log_f = np.zeros((K,K))
#            # safe way
#            for i in range(K):
#                for j in range(K):
#                    log_f[i,j] += np.dot(Q[vp,t,:], np.log(theta[:,i,j]))
#
##            breakpoint()
#            
#            # possibly need to check if this is leaf node
#            for k in range(K):
#                log_f[:,k] += np.log(g_t[k])
#                for ch_i in vert_children[si]:
#                    log_f[:,k] += (Q_pairs[ch_i,t,:,:] * np.log(theta[j,:,:])).sum()
#                    
#            f_t = np.exp(log_f)
#            g_t = f_t.sum(axis=1)             
#            transmat[t,:,:] = np.dot(diag(1./g_t), f_t)
#            g_t /= max(g_t)
#            if abs(transmat[t,1].sum()-1) > 1E-6:
#                #breakpoint()
#                pass
#
#        # first node
#        log_f1 = np.dot(Q[vp,0,:], np.log(beta)) + np.log(g_t)  # vector instead of a matrix
#        #breakpoint()
#        for ch_i in vert_children[si]:          
#            log_f1 += np.dot(np.log(beta), Q[ch_i,0,:])
#            
#        tmp = np.exp(log_f1)
#        f1 = tmp/tmp.sum()
#
#    # update Q, Q_pairs, lh            
#    a_t = f1 * emit_probs_mat[:, 0]
#    s_t = [a_t.sum()]
#    a_t /= s_t[0]
#    b_t = np.ones((K,))
#    a_s[0,:] = a_t
#    b_s[T-1,:] = b_t
#
#    # forward algorithm
#    for t in range(1,T):
##        breakpoint()        *#*#*#*#     
#        a_t = emit_probs_mat[:, t] * (np.dot(a_t.T, transmat[t,:,:]))
#        s_t.append(a_t.sum())
#        a_t /= s_t[t]
#        a_s[t,:] = a_t
#
#    #backward algorithm        
#    for t in range(T-2,-1,-1):
#        b_t = np.dot(transmat[t+1,:,:], emit_probs_mat[:,t+1]*b_t)
#        b_t /= s_t[t+1]
#        b_s[t,:] = b_t
#        
#    loglh[si] = np.log(array(s_t)).sum()
##    print a_s.shape, b_s.shape, loglh.shape
#    tmp1 = a_s[0,:] * b_s[0,:]
#    Q[si,0,:] = tmp1/tmp1.sum()
#    for t in range(1,T):
#        tmp1 = a_s[t,:]* b_s[t,:]
#        Q[si, t,:] = tmp1/tmp1.sum()
#        tmp2 = np.dot(np.dot(diag(a_s[t-1,:]), transmat[t]), diag(emit_probs_mat[:,t]* b_s[t,:]))
#        Q_pairs[si, t]  = tmp2/tmp2.sum()        
#
#
#
#def log_emit_probs_i(emit_probs, X, i):
##"""Get the emission probability for the given X[i,t]"""
#    K = emit_probs.shape[0]
#    I, T, L = X.shape
#    #I = 1
#    a = np.zeros((K, T))
#    for t in range(T):
#        for l in range(L):
#            if X[i,t,l]:
#                a[:,t] += np.log(emit_probs[:,l])
#            else:
#                a[:,t] += np.log(1-emit_probs[:,l])
#            
#    return a



def prodc_update_params(args, renormalize=True):
    cdef:
        np.ndarray[np.int8_t, ndim=3] X
        np.ndarray[np.float64_t, ndim=3] Q
        np.ndarray[np.float64_t, ndim=4] Q_pairs
        np.ndarray[np.float64_t, ndim=4] theta
        np.ndarray[np.float64_t, ndim=2] alpha, beta, emit_probs
        np.ndarray[np.float64_t, ndim=1] gamma
        np.ndarray[np.int8_t, ndim=1] vert_parent
        np.ndarray[np.float64_t, ndim=3] log_obs_mat
        np.float64_t pseudocount
    X = args.X
    Q, Q_pairs, theta, alpha, beta, gamma, emit_probs, vert_parent, vert_children, log_obs_mat, pseudocount, mark_avail = (args.Q, args.Q_pairs, args.theta, args.alpha, args.beta,
                                                   args.gamma, args.emit_probs, args.vert_parent, args.vert_children, args.log_obs_mat, args.pseudocount, args.mark_avail)
#    print 'got here'
    cdef int I = Q.shape[0], T = Q.shape[1], K = Q.shape[2]
    cdef int L = X.shape[2]
    cdef Py_ssize_t i,t,v,h,k,vp,l
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
                                theta[i-1,v,h,k] += Q[vp,t,v] * Q_pairs[i,t,h,k]
                for l in xrange(L):
                    real_i = args.real_species_i if args.real_species_i is not None else i
                    if mark_avail[real_i,l] and X[i,t,l]:
                        emit_probs[k, l] += Q[i, t, k]
    if renormalize:
        normalize_trans(theta, alpha, beta, gamma)
    normalize_emit(Q, emit_probs, pseudocount, args, renormalize)
    
    make_log_obs_matrix(args)

cpdef np.float64_t prodc_free_energy(args):
    """Calculate the free energy for Q"""
    cdef np.int8_t[:,:,:] X
    cdef np.ndarray[np.float64_t, ndim=3] Q
    cdef np.float64_t[:,:,:,:] Q_pairs, theta
    cdef np.float64_t[:,:] alpha, beta, emit
    cdef np.float64_t[:] gamma
    cdef np.int8_t[:] vert_parent
    cdef np.float64_t[:,:,:] log_obs_mat
    X, Q, Q_pairs, theta, alpha, beta, gamma, emit, vert_parent, vert_children, log_obs_mat = (args.X, args.Q, args.Q_pairs, args.theta,
                                                   args.alpha, args.beta,
                                                   args.gamma, args.emit_probs, args.vert_parent, args.vert_children, args.log_obs_mat)
    cdef Py_ssize_t I = Q.shape[0], T = Q.shape[1], K = Q.shape[2]
    cdef Py_ssize_t i,t,v,h,k, ch_i, vp, len_v_chs
    cdef np.float64_t[:] log_gamma
    cdef np.float64_t[:,:] log_alpha, log_beta
    cdef np.float64_t[:,:,:,:] log_theta
    #print 'mf_free_energy'
    log_theta = np.log(theta)
    log_alpha = np.log(alpha)
    log_beta = np.log(beta)
    log_gamma = np.log(gamma)
    
    # Q * logQ
    cdef np.float64_t total_free = 0., q_cond = 0.
    for i in xrange(I):
    #for i in prange(I, nogil=True):
        for k in xrange(K):
            total_free += Q[i,0,k] * log(Q[i,0,k])  # first column's entropy
            for t in xrange(1,T):
                for v in xrange(K):
                    q_cond = Q_pairs[i,t,k,v] / Q[i,t-1,k]
                    total_free += Q_pairs[i,t,k,v] * log(q_cond)
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
                                total_free -= Q[vp,t,v] * Q_pairs[i,t,h,k] * log_theta[i-1,v,h,k]
    #prodc_free_energy_old(args)
    return total_free



    
#def prodc_free_energy_old(args):
#    """calculate the free energy for the current Q and parameters"""
#    theta, alpha, beta, gamma, emit_probs, X, Q, Q_pairs = (args.theta,
#                        args.alpha, args.beta, args.gamma, args.emit_probs,
#                        args.X, args.Q, args.Q_pairs)
#    vert_parent = args.vert_parent
#    log_obs_mat = args.log_obs_mat
#    I, T, L = X.shape
#    #I = 1
#    K = alpha.shape[0]
#    free_e = 0.
#    log_theta, log_alpha, log_beta, log_gamma = np.log(theta), np.log(alpha), np.log(beta), np.log(gamma)
#    #log_emit_probs_mat = np.zeros((K,T))
#    for i in range(I):
#        # calculate emission matrix K*T
#        #log_emit_probs_mat[:] = log_emit_probs_i(emit_probs, X, i)
#        #log_emit_probs_mat[:] = np.exp(log_obs_mat[i,:,:])
#        log_emit_probs_mat = args.log_obs_mat[i,:,:].T
#        
#        #calculate Q* log(Q)
#        free_e += np.dot(Q[i, 0,:], np.log(Q[i, 0,:]))
#        for t in xrange(1, T):
#            q_cond = np.dot(diag(1./Q[i,t-1,:]), Q_pairs[i, t])
#            #if (q_cond - 1. > epsilon).any(): #abs(q_cond.sum(axis=1) -1.).any() > epsilon:
#            #    breakpoint()
#            #    print q_cond
#            
#            free_e += (Q_pairs[i, t] * np.log(q_cond)).sum()
#
#        # calculate - Q* log(P)
#        if i == 0:
#            for t in range(T):
#                free_e -= (Q[i,t,:] * log_emit_probs_mat[:, t] ).sum()
#                if t == 0:
#                    free_e -= (Q[0,0,:] * log_gamma).sum()
#                else:
#                    free_e -= (Q_pairs[i, t] * log_alpha).sum()
#            #breakpoint()
#        else:
#            vp = vert_parent[i]
#            for t in xrange(T):
#                free_e -= (Q[i,t,:] * log_emit_probs_mat[:, t] ).sum()
#                if t == 0:
#                    free_e -= np.dot(np.dot(diag(Q[vp,0,:]), log_beta), diag(Q[i,0,:])).sum()
#                else:
#                    tmp = []
#                    for k in range(K):
#                        tmp.append((Q_pairs[i, t] * log_theta[k]).sum())
#                    free_e -= np.dot(Q[vp, t], tmp)
#            #breakpoint()
#    print 'final part:', free_e
#    return free_e
#
