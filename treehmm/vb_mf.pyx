#!python
#cython: boundscheck=False, wraparound=False
# can also add profile=True

cimport cython
from cython.parallel import prange
import numpy as np
import sys
import copy
cimport numpy as np
from libc.math cimport exp, log

from scipy.stats import norm
import time

#ctypedef np.float128_t float_type
#ctypedef long double float_type
ctypedef double float_type



@cython.profile(False)
cpdef inline float_type log_obs(Py_ssize_t i, Py_ssize_t t, Py_ssize_t k, float_type[:,:] emit_probs, np.int8_t[:,:,:] X, np.int8_t[:,:] mark_avail, Py_ssize_t real_i) nogil:
    """Get the emission probability for the given X[i,t,k]"""
    cdef float_type total = 0.
    cdef Py_ssize_t l
    for l in xrange(X.shape[2]):
        if mark_avail[real_i,l]:
            if X[i,t,l]:
                total += log(emit_probs[k,l])
            else:
                total += log(1. - emit_probs[k,l])
    return total


_norm_pdf_logC = np.log(np.sqrt(2 * np.pi)).astype(np.double)
def _norm_logpdf(x, loc, scale):
    x, loc, scale = map(np.asarray, (x, loc, scale))
    x = (x - loc) * 1.0/scale
    return -x**2 / 2.0 - _norm_pdf_logC - np.log(scale)



@cython.profile(False)
@cython.cdivision(True)
cpdef inline float_type log_obs_gaussian(Py_ssize_t i, Py_ssize_t t, Py_ssize_t k, float_type[:,:] means, float_type[:,:] variances, float_type[:,:,:] X):
    """Get the emission probability for the given X[i,t,:] when the parent is in state k"""
    #val = norm.pdf(X[i,t,:], loc=means[k,:], scale=np.sqrt(variances[k,:]))
    #val[val <= 5e-5] = 5e-5
    ##if (np.any(norm.pdf(X[i,t,:], loc=means[k,:], scale=np.sqrt(variances[k,:])) <= 0) or
    ##        not np.all(np.isfinite(val))):
    ##    print 'bad number in log_obs_gaussian', i,t,k, '\nmeans = ', means[k,:], '\nvariances = ', variances[k,:], '\npdf = ', norm.pdf(X[i,t,:], loc=means[k,:], scale=np.sqrt(variances[k,:])), '\nlogpdf = ', np.log(norm.pdf(X[i,t,:], loc=means[k,:], scale=np.sqrt(variances[k,:])))
    ##    time.sleep(.01)
    #return np.log(val).sum()

    #val = norm.logpdf(X[i,t,:], loc=means[k,:], scale=np.sqrt(variances[k,:], dtype=np.double), dtype=np.double)
    val = _norm_logpdf(X[i,t,:], loc=means[k,:], scale=np.sqrt(variances[k,:], dtype=np.double))

    #if np.any(val < -50):
    #    print '********* too low', list(val), i, t, k, list(X[i,t,:]), list(means[k,:]), list(variances[k,:])
    #    val[val < -50] = -50
    #if np.any(val == 0.):
    #    print '********* underflow?', list(val), i, t, k, list(X[i,t,:]), list(means[k,:]), list(variances[k,:])
    #    val[val == 0.] = -50
    return val.sum()
cpdef make_log_obs_matrix(args):
    """Update p(X^i_t|Z^i_t=k, emit_probs) (I,T,K) from current emit_probs"""
    cdef np.ndarray[float_type, ndim=3] log_obs_mat = args.log_obs_mat
    cdef np.int8_t[:,:,:] X = args.X
    cdef float_type[:,:] emit = args.emit_probs
    cdef np.int8_t[:,:] mark_avail = args.mark_avail
    cdef Py_ssize_t I = X.shape[0], T = X.shape[1], K = emit.shape[0]
    cdef Py_ssize_t i,t,k
    #log_obs_mat[...] = np.zeros((I,T,K))
    log_obs_mat[:] = 0.
    #cdef float_type[:,:,:] obs_mat_view = log_obs_mat
    #print 'making log_obs matrix'
    #for i in prange(I, nogil=True):
    for i in xrange(I):
        real_i = args.real_species_i if args.real_species_i is not None else i
        for t in xrange(T):
            for k in xrange(K):
                #obs_mat_view[i,t,k] = log_obs(i,t,k,emit,X)
                log_obs_mat[i,t,k] = log_obs(i,t,k,emit,X, mark_avail, real_i)

@cython.cdivision(True)
cpdef make_log_obs_matrix_gaussian(args):
    """Update p(X^i_t|Z^i_t=k, emit_probs) (I,T,K) from current emit_probs"""
    cdef np.ndarray[float_type, ndim=3] log_obs_mat = args.log_obs_mat
    cdef float_type[:,:,:] X = args.X
    cdef float_type[:,:] means = args.means
    cdef float_type[:,:] variances = args.variances
    cdef Py_ssize_t I = X.shape[0], T = X.shape[1], K = means.shape[0]
    cdef Py_ssize_t i,t,k
    #log_obs_mat[...] = np.zeros((I,T,K))
    log_obs_mat[:] = 0.
    #cdef float_type[:,:,:] obs_mat_view = log_obs_mat
    print 'making log_obs matrix'
    #for i in prange(I, nogil=True):
    for i in xrange(I):
        for t in xrange(T):
            for k in xrange(K):
                #obs_mat_view[i,t,k] = log_obs(i,t,k,emit,X)
                val = log_obs_gaussian(i,t,k,means, variances,X)
                if np.isfinite(val) and val < 0. or val > 0.:
                    log_obs_mat[i,t,k] = val
                elif not np.isfinite(val):
                    print '******Infinite value!'
                    log_obs_mat[i,t,k] = -100.
                elif val == 0:
                    print '******log_obs value is == 0 ?', val
                    log_obs_mat[i,t,k] = -100.  # small... exp(x) = 5e-5
                    #print 'bad number... Q is ', args.Q[i,t,k]
                else:
                    print '*****something else?', val
                    log_obs_mat[i,t,k] = -100.
    print 'done'


cpdef normalize_trans(theta,
                        np.ndarray[float_type, ndim=2] alpha,
                        np.ndarray[float_type, ndim=2] beta,
                        np.ndarray[float_type, ndim=1] gamma):
    """renormalize transition matrices appropriately"""
    cdef Py_ssize_t K, k, v, h
    K = gamma.shape[0]
    if len(theta.shape)==3:
        t_sum = theta.sum(axis=2)
        for k in range(K):
            theta[:,:,k] /= t_sum
    else:  # case for separate theta
        t_sum = theta.sum(axis=3)
        for k in range(K):
            theta[:,:,:,k] /= t_sum
    a_sum = alpha.sum(axis=1)
    b_sum = beta.sum(axis=1)
    g_sum = gamma.sum()
    # all probability goes to one of K states
    for k in range(K):
        alpha[:, k] /= a_sum
        beta[:, k] /= b_sum
    gamma /= g_sum


cpdef normalize_emit(np.ndarray[float_type, ndim=3] Q,
                        np.ndarray[float_type, ndim=2] emit_probs,
                        float_type pseudocount, args, renormalize=True):
    """renormalize emission probabilities using Q"""
    cdef Py_ssize_t I, T, K, L, i, t, k, l
    I = Q.shape[0]
    T = Q.shape[1]
    K = emit_probs.shape[0]
    L = emit_probs.shape[1]
    cdef np.ndarray[float_type, ndim=2] e_sum = np.ones((K,L), dtype=np.double) * pseudocount * T
    # all probability goes to one of K states
    for k in range(K):
        for i in xrange(I):
            real_i = args.real_species_i if args.real_species_i is not None else i
            for l in xrange(L):
                if args.mark_avail[real_i,l]:
                    for t in xrange(T):
                    #e_sum[k] += Q[i,t,k]
                        e_sum[k,l] += Q[i,t,k]
    if renormalize:
        #emit_probs[:] = np.dot(np.diag(1./e_sum), emit_probs)
        emit_probs[:] = emit_probs/e_sum
    args.emit_sum = e_sum


cpdef mf_random_q(I, T, K):
    """Create a random Q distribution for mean-field inference"""
    # each i,t has a distribution over K
    Q = np.random.rand(I, T, K).astype(np.double)
    q_sum = Q.sum(axis=2)
    for i in xrange(I):
        for t in xrange(T):
            Q[i, t, :] /= q_sum[i, t]
    return Q


cpdef float_type mf_free_energy(args):
    """Calculate the free energy for Q"""
    cdef np.int8_t[:,:,:] X
    cdef float_type[:,:,:] Q, theta
    cdef float_type[:,:] alpha, beta, emit
    cdef float_type[:] gamma
    cdef np.int8_t[:] vert_parent
    cdef float_type[:,:,:] log_obs_mat
    X, Q, theta, alpha, beta, gamma, vert_parent, vert_children, log_obs_mat = (args.X, args.Q, args.theta,
                                                   args.alpha, args.beta,
                                                   args.gamma, args.vert_parent, args.vert_children, args.log_obs_mat)
    cdef Py_ssize_t I = Q.shape[0], T = Q.shape[1], K = Q.shape[2]
    cdef Py_ssize_t i,t,v,h,k, ch_i, vp, len_v_chs
    cdef float_type[:] log_gamma
    cdef float_type[:,:] log_alpha, log_beta
    cdef float_type[:,:,:] log_theta
    #print 'mf_free_energy'
    log_theta = np.log(theta)
    log_alpha = np.log(alpha)
    log_beta = np.log(beta)
    log_gamma = np.log(gamma)

    cdef float_type total_free = (Q * np.log(Q)).sum()
    for i in xrange(I):
    #for i in prange(I, nogil=True):
        vp = vert_parent[i]
        for t in xrange(T):
            for k in xrange(K):
                for v in xrange(K):
                    if t == 0 and i ==0:
                        # GAMMA
                        total_free -= Q[i,t,k] * (log_gamma[k] + log_obs_mat[i,t,k])
                    else:
                        if i > 0 and t > 0:
                            # THETA
                            for h in xrange(K):
                                total_free -= Q[vp,t,v] * Q[i,t-1,h] * Q[i,t,k] * (log_theta[v,h,k] + log_obs_mat[i,t,k])
                        elif i == 0:
                            # ALPHA
                            total_free -= Q[i,t-1,v] * Q[i,t,k] * (log_alpha[v,k] + log_obs_mat[i,t,k])
                        else:
                            # BETA
                            total_free -= Q[vp,t,v] * Q[i,t,k] * (log_beta[v,k] + log_obs_mat[i,t,k])
    return total_free


cpdef mf_update_q(args):
    """Calculate q_{it} for the fixed parameters"""
    cdef np.int8_t[:,:,:] X
    cdef np.ndarray[float_type, ndim=3] Q
    cdef float_type[:,:,:] theta
    cdef float_type[:,:] alpha, beta, emit
    cdef float_type[:] gamma
    cdef np.int8_t[:] vert_parent
    cdef float_type[:,:,:] log_obs_mat
    #Q = args.Q
    X = args.X
    Q, theta, alpha, beta, gamma, vert_parent, vert_children, log_obs_mat = (args.Q, args.theta,
                                                   args.alpha, args.beta,
                                                   args.gamma, args.vert_parent, args.vert_children, args.log_obs_mat)
    args.Q_prev = copy.deepcopy(args.Q)
    cdef Py_ssize_t I = Q.shape[0], T = Q.shape[1], K = Q.shape[2]
    cdef Py_ssize_t i,t,v,h,k,ch_i,vp
    cdef Py_ssize_t len_v_chs
    cdef np.int32_t[:] v_chs
    cdef float_type[:] log_gamma
    cdef float_type[:,:] log_alpha, log_beta
    cdef float_type[:,:,:] log_theta
    #print 'mf_update_q'
    log_theta = np.log(theta)
    log_alpha = np.log(alpha)
    log_beta = np.log(beta)
    log_gamma = np.log(gamma)

    #cdef np.ndarray[float_type, ndim=2] phi = np.zeros((T,K))
    cdef float_type[:,:] phi = np.zeros((T,K), dtype=np.double)
    cdef np.ndarray[float_type, ndim=1] totals = np.zeros(T, dtype=np.double)
    for i in xrange(I):
        #print 'i', i
        phi = np.zeros((T,K), dtype=np.double)
        #numpy_array = np.asarray(<float_type[:T,:K]> phi)
        #numpy_array[:] = 0.
        v_chs = vert_children[i]
        len_v_chs = v_chs.size
        vp = vert_parent[i]
        totals[:] = 0.
        for t in xrange(T):
            for k in xrange(K):
                for v in xrange(K):
                    if t == 0 and i ==0:
                        # GAMMA
                        #phi[t,k] += log_obs_mat[i,t,k]
                        phi[t,k] += log_gamma[k] + log_obs_mat[i,t,k]
                        if t + 1 < T:
                            phi[t,k] += Q[i,t+1,v] * (log_alpha[k,v])
                            #phi[t,k] += Q[i,t+1,v] * (log_alpha[k,v] + log_obs_mat[i,t+1,v])
                        for j in xrange(len_v_chs):
                            ch_i = v_chs[j]
                            phi[t,k] += Q[ch_i,t,v] * (log_beta[k,v])
                            #phi[t,k] += Q[ch_i,t,v] * (log_beta[k,v] + log_obs_mat[ch_i,t,v])
                    else:
                        if i > 0 and t > 0:
                            # THETA
                            for h in xrange(K):
                                phi[t,k] += Q[vp,t,v] * Q[i,t-1,h] * (log_theta[v,h,k] + log_obs_mat[i,t,k])
                                if t + 1 < T:
                                    phi[t,k] += Q[vp,t+1,v] * Q[i,t+1,h] * (log_theta[v,k,h])
                                    #phi[t,k] += Q[vp,t+1,v] * Q[i,t+1,h] * (log_theta[v,k,h] + log_obs_mat[i,t,h])
                                for j in xrange(len_v_chs):
                                    ch_i = v_chs[j]
                                    #phi[t,k] += Q[ch_i,t,v] * Q[ch_i,t-1,h] * (log_theta[k,h,v] + log_obs_mat[ch_i,t,v])
                                    phi[t,k] += Q[ch_i,t,v] * Q[ch_i,t-1,h] * (log_theta[k,h,v])

                        elif i == 0:
                            # ALPHA
                            phi[t,k] += Q[i,t-1,v] * (log_alpha[v,k] + log_obs_mat[i,t,k])
                            if t + 1 < T:
                                #phi[t,k] += Q[i,t+1,v] * (log_alpha[k,v] + log_obs_mat[i,t+1,v])
                                phi[t,k] += Q[i,t+1,v] * (log_alpha[k,v])
                            for j in xrange(len_v_chs):
                                ch_i = v_chs[j]
                                for h in xrange(K):
                                    phi[t,k] += Q[ch_i,t,v] * Q[ch_i,t-1,h] * (log_theta[k,h,v])
                                    #phi[t,k] += Q[ch_i,t,v] * Q[ch_i,t-1,h] * (log_theta[k,h,v] + log_obs_mat[ch_i, t,v])
                        else:
                            # BETA
                            phi[t,k] += Q[vp,t,v] * (log_beta[v,k] + log_obs_mat[i,t,k])
                            if t + 1 < T:
                                for h in xrange(K):
                                    phi[t,k] += Q[i,t+1,h] * (log_theta[v,k,h])
                                    #phi[t,k] += Q[i,t+1,h] * (log_theta[v,k,h] + log_obs_mat[i,t+1,h])
                            for j in xrange(len_v_chs):
                                ch_i = v_chs[j]
                                #phi[t,k] += Q[ch_i,t,v] * (log_beta[k,v] + log_obs_mat[ch_i,t,v])
                                phi[t,k] += Q[ch_i,t,v] * (log_beta[k,v])

            for k in xrange(K):
                phi[t,k] = exp(phi[t,k])
                totals[t] += phi[t,k]
            for k in xrange(K):
                Q[i, t, k] = phi[t,k] / totals[t]
        ## end t
        ####phi[:] = np.exp(phi)
        #for t in xrange(T):
        #    for k in xrange(K):
        #        Q[i,t,k] = phi[t,k] / totals[t]



cpdef mf_update_params(args, renormalize=True):
    cdef np.ndarray[np.int8_t, ndim=3] X
    cdef np.ndarray[float_type, ndim=3] Q, theta
    cdef np.ndarray[float_type, ndim=2] alpha, beta, emit_probs
    cdef np.ndarray[float_type, ndim=1] gamma
    cdef np.ndarray[np.int8_t, ndim=1] vert_parent
    cdef float_type[:,:,:] log_obs_mat
    cdef float_type pseudocount
    X = args.X
    Q, theta, alpha, beta, gamma, emit_probs, vert_parent, vert_children, log_obs_mat, pseudocount, mark_avail = (args.Q, args.theta, args.alpha, args.beta,
                        args.gamma, args.emit_probs, args.vert_parent, args.vert_children, args.log_obs_mat, args.pseudocount, args.mark_avail)
    cdef int I = Q.shape[0], T = Q.shape[1], K = Q.shape[2]
    cdef int L = X.shape[2]
    cdef Py_ssize_t i,t,v,h,k,vp,l
    #print 'mf_update_params'
    if args.continuous_observations:
        new_means = np.zeros_like(args.means)
        new_variances = np.zeros_like(args.variances)
        new_means[:] = pseudocount  # need a pseudocount for mean and variance?
        new_variances[:] = pseudocount
        total_q = np.ones((K,L)) * pseudocount
    else:
        emit_probs = args.emit_probs
        emit_probs[:] = pseudocount
    theta[:] = pseudocount*T
    alpha[:] = pseudocount*T
    beta[:] = pseudocount
    gamma[:] = pseudocount
    emit_probs[:] = pseudocount*T
    for i in xrange(I):
        real_i = args.real_species_i if args.real_species_i is not None else i
    #for i in prange(I, nogil=True):
        vp = vert_parent[i]
        for t in xrange(T):
            for k in xrange(K):
                if i==0 and t==0:
                    gamma[k] += Q[i, t, k]
                else:
                    for v in xrange(K):
                        if t == 0:
                            beta[v,k] += Q[i,t,k] * Q[vp,t,v]
                        elif i == 0:
                            alpha[v,k] += Q[i,t,k] * Q[i,t-1,v]
                        else:
                            for h in xrange(K):
                                theta[v,h,k] += Q[i,t,k] * Q[i,t-1,h] * Q[vp,t,v]
                if not args.continuous_observations:
                    for l in xrange(L):
                        if mark_avail[real_i,l] and X[i,t,l]:
                            emit_probs[k, l] += Q[i, t, k]
    if args.continuous_observations:
        for i in xrange(I):
            for t in xrange(T):
                for k in xrange(K):
                    for l in xrange(L):
                        new_means[k,l] += Q[i, t, k] * X[i,t,l]  # expectation of X wrt Q
                        total_q[k,l] += Q[i,t,k]
        for i in xrange(I):
            for t in xrange(T):
                for k in xrange(K):
                    for l in xrange(L):
                        new_variances[k,l] += Q[i, t, k] * (X[i,t,l] - new_means[k,l]) ** 2  # XXX should be new_means?
        args.means[:] = new_means / total_q
        args.variances[:] = new_variances / (I * T)
    else:
        normalize_emit(Q, emit_probs, pseudocount, args, renormalize)

    if renormalize:
        normalize_trans(theta, alpha, beta, gamma)
    normalize_emit(Q, emit_probs, pseudocount, args, renormalize)

    if args.continuous_observations:
        make_log_obs_matrix_gaussian(args)
    else:
        make_log_obs_matrix(args)

def mf_check_convergence(args):
    return (np.abs(args.Q_prev - args.Q).max(axis=0) < 1e-3).all()
