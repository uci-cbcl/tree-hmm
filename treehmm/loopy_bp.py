# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 10:55:55 2012

@author: yuanfeng wang
"""

#!/usr/bin/env python
# loopy BP for learning graphical model of  chromatin modification 

import scipy as sp
import copy
from numpy import array, random, diag
from math import log

from vb_mf import normalize_trans, normalize_emit, make_log_obs_matrix 
try:
    from ipdb import set_trace as breakpoint
except ImportError:
    from pdb import set_trace as breakpoint
#sp.random.seed([10])


def bp_initialize_msg(args):
    I, T, K, vert_children = args.I , args.T, args.K, args.vert_children
    lmds = sp.zeros(I, dtype = object)
    #lmds = [0 for i in range(I)]
    pis = sp.zeros(I, dtype = object)
    # all messages needed to store for the specific tree we have now
#    for i in range(I):
#        lmds[i] = sp.rand(2, T, K)  # lmds[0][0, :,: ] is not used because there is no vertical parent, only horizontal
#        pis[i] = sp.rand(len(vert_children[i])+1, T, K) # i=0 has size (9, T, K), others has size (1, T, K)
#    #lmds = sp.rand(2*I-1, T, K) 
#    #pis = sp.rand(2*I-1, T, K)
#    normalize_msg(lmds, pis)
    
    # breakpoint()
    for i in range(I):
        lmds[i] = sp.ones((2,T+1,K))
        pis[i] =  sp.ones((len(vert_children[i])+1, T,K))    
#    print lmds[0].shape
    return lmds, pis


def normalize_msg(lmds, pis):
    #normalize both lmds and pis
    I = lmds.shape[0]
    
    for i in xrange(I):
        Im, Tp1, K = lmds[i].shape
        for im in xrange(Im):
            for t in xrange(Tp1):
                lmds[i][im,t,:] /= lmds[i][im,t,:].sum()
                
        Im, T, K = pis[i].shape        
        for im in xrange(Im):
            for t in xrange(T):
                pis[i][im,t,:] /= pis[i][im,t,:].sum()


def bp_update_msg_new(args):
    '''now assume the tree are denoted as vert_PARENTS {1:Null, 2:1, 3:2, ...}, vert_Children {1:2, 2:3 , ...}'''
    print 'loopy2'    
    lmds, pis = args.lmds, args.pis
    vert_children = args.vert_children
    vert_parent = args.vert_parent
    I, T, L = args.X.shape
    #print I, T, L
    #print  pis[0].shape
    K = args.gamma.shape[0]
    emit_probs_mat = sp.exp(args.log_obs_mat)
    emit_probs_mat /= emit_probs_mat.max(axis=2).reshape(I, T, 1)
    theta, alpha, beta, gamma, emit_probs = args.theta, args.alpha, args.beta, args.gamma, args.emit_probs 
    
    lmds_prev = args.lmds_prev = copy.deepcopy(lmds) #sp.copy(lmds)
    pis_prev = args.pis_prev = copy.deepcopy(pis) #sp.copy(pis)
    if ~hasattr(args, 'evidence'):
        evidence  = args.evidence = evid_allchild(lmds_prev, vert_children)
    else:
        evidence = args.evidence 
        
    for i in range(I):
        lmds[i][:] = args.pseudocount
        pis[i][:] = args.pseudocount
    #breakpoint()
    for i in xrange(I):  # can be parallelized
        vcs = vert_children[i]
        vp = vert_parent[i]
        if i != 0:
            idx_i = vert_children[vp].tolist().index(i)
        for t in xrange(T):
            #print i, T
            #print pis[i][0,t]
            #print pis_prev[i][0,t]
                
            if i == 0 and t == 0:
                # msg to vertical child
                for id_vc, vc in enumerate(vcs):
                    pis[i][id_vc, t, :] += gamma * emit_probs_mat[0, t, :] * evidence[i,t] / lmds_prev[vc][0,t,:]
                # msg to horizontal child
                pis[i][-1, t,:] += gamma * emit_probs_mat[0, t, :] * evidence[i,t] / lmds_prev[i][1,t+1,:]
                # no lambda meessage in this case

            elif i==0:  # different than t=0 case : now there is a parent, need to sum over; also there is lambda message
                tmp1, tmp2 = sp.ix_(pis_prev[i][-1, t-1,:], emit_probs_mat[i,t,:] * evidence[i,t,:])
                BEL = alpha * (tmp1 * tmp2)
                # msg to (iterate over) vertical child
                for id_vc, vc in enumerate(vcs):
                    pis[i][id_vc, t,:] += sp.dot(BEL, diag(1./lmds_prev[vc][0,t,:])).sum(axis=0)  
#                    tmp = BEL.sum(axis=0) /lmds_prev[vc][0,t,:]
#                    print pis[i][id_vc, t,:]
#                    print tmp
#                    breakpoint()
                # msg to horizontal child
                pis[i][-1, t,:] += sp.dot(BEL, diag(1./lmds_prev[i][1,t+1,:])).sum(axis=0)
                    
                # msg to horizontal parent
                lmds[i][1, t,:] += sp.dot(diag(1/pis_prev[i][-1, t-1,:]), BEL).sum(axis=1)
                #breakpoint()

            elif t==0:
#                    tmp1, tmp2 = sp.ix_(pis_prev[vp][idx_i,t,:], 
#                    BEL = 
                if len(vcs) == 0:
                    pis[i][-1,t,:] += sp.dot(pis_prev[vp][idx_i,t,:], beta) * (emit_probs_mat[i,t,:])
                else:
                    pis[i][-1,t,:] += sp.dot(pis_prev[vp][idx_i,t,:], beta) * (emit_probs_mat[i,t,:]* evidence[i,t] /lmds_prev[i][-1,t+1,:]) # only  change: i-1 -> 0 
                    for id_vc, vc in enumerate(vcs):
                        pis[i][id_vc, t, :] += sp.dot(pis_prev[vp][idx_i,t,:], beta) * emit_probs_mat[i,t,:] * evidence[i,t] / lmds_prev[vc][0,t,:]
    #                    pis[i][0,t,:] += sp.dot(pis_prev[vp][idx_i,t,:], beta) * (emit_probs_mat[i,t,:]*lmds_prev[i][-1, t+1,:])
                # in general, should iterate over parents
                lmds[i][0,t,:] += sp.dot(beta, emit_probs_mat[i,t,:]*evidence[i,t])
                lmds[i][1,t,:] += sp.ones(K) #  doesn't matter for there is no horizontal parent
                
                #tmp1, tmp2 = sp.ix_(pis_prev[vp][i-1,t,:], emit_probs_mat[i,t,:]*evidence[i,t,:])
                #BEL = beta * (tmp1 * tmp2)
                ## in general, should iterate over children
                #pis[i][-1,t,:] += sp.dot(BEL, 1/lmds_prev[i][1,t+1,:]).sum(axis=0) 
                ##pis[i][-1,t,:] += (BEL/lmds_prev[i][1,t+1,:].reshape(1,K)).sum(axis=0) 
                ## in general, should iterate over parents
                #lmds[i][0,t,:] += sp.dot(diag(1/pis_prev[vp][i-1, t,:]), BEL).sum(axis=1)  # The index i-1 is a hand-waiving way to do it, in principle should match the index of child species i in pis
                ##lmds[i][0,t,:] += (BEL//pis_prev[vp][i-1, t,:].reshape(K,1)).sum(axis=1)
                #lmds[i][1,t,:] += sp.ones(K) #  doesn't matter for there is no horizontal parent

            else:
                #tmp = sp.zeros(K)
                #for k1 in range(K):
                #    for k2 in range(K):
                #        tmp += theta[k1, k2, :] * pis_prev[vp][i-1,t,k1] * pis_prev[i][-1, t-1, k2] * emit_probs_mat[i,t,:]
                #pis[i][-1,t,:] = tmp
                tmp1, tmp2, tmp3 = sp.ix_(pis_prev[vp][idx_i,t,:], pis_prev[i][-1, t-1, :], emit_probs_mat[i,t,:]*evidence[i,t,:])
                BEL = theta* (tmp1* tmp2 * tmp3)
                if len(vcs) == 0:
                    pis[i][-1,t,:] += (BEL.sum(axis=0)).sum(axis=0) /lmds_prev[i][1,t+1,:]  
#                    breakpoint()
                else:
                    pis[i][-1,t,:] += (BEL.sum(axis=0)).sum(axis=0) /lmds_prev[i][1,t+1,:]  
#                    pis[i][-1,t,:] += (sp.dot(BEL, 1/lmds_prev[i][1,t+1,:]).sum(axis=0)).sum(axis=0)
                    for id_vc, vc in enumerate(vcs):       
#                        pis[i][id_vc,t,:] += (sp.dot(BEL, 1/lmds_prev[vc][0,t,:]).sum(axis=0)).sum(axis=0)
                        pis[i][id_vc,t,:] += (BEL.sum(axis=0)).sum(axis=0) /lmds_prev[vc][0,t,:]
                

                
                #tmp_lmd1= sp.zeros(K)
                #tmp_lmd2= sp.zeros(K)
                #for k1 in range(K):
                #    for k2 in range(K):
                #        tmp_lmd1 += theta[:,k1, k2] * pis_prev[i][0, t-1, k2]* emit_probs_mat[i,t,k2] * evidence[i,t,k2]
                #        tmp_lmd2 += theta[k1,:,k2] * pis_prev[vp][i-1, t, k2]* emit_probs_mat[i,t,k2] * evidence[i,t,k2]
                #print tmp_lmd1
                #print tmp_lmd2
                
                ##lmds[i][0,t,:] += tmp_lmd1
                ##lmds[i][1,t,:] += tmp_lmd2
                
#                tmp_lmd1 = ((BEL / pis_prev[vp][idx_i, t, :].reshape(K,1,1)).sum(axis=1)).sum(axis=1)
#                tmp_lmd2 = ((BEL / pis_prev[i][-1, t-1, :].reshape(1,K,1)).sum(axis=0)).sum(axis=1)     # t=T is not used
                lmds[i][0,t,:] += (BEL.sum(axis=1)).sum(axis=1) / pis_prev[vp][idx_i, t, :]
                lmds[i][1,t,:] += (BEL.sum(axis=0)).sum(axis=1) /pis_prev[i][-1, t-1, :]
                #print lmds[i][0,t,:]
                #checked, same as above version (from line 353)
                #breakpoint()

    normalize_msg(lmds, pis)
#    for i in range(I):
#        print pis[i].shape
#        print pis[i][0,t,:]
#        print lmds[i].shape
#        print lmds[i][0,t,:]
    args.evidence = evid_allchild(lmds, vert_children)
    
    
def evid_allchild(lmds, vert_children): 
    '''calculate the evidence from all children(lambda messages), emit_probs not included'''
    I = lmds.shape[0]
    tmp0, T, K = lmds[0].shape
    T = T-1
    evidence = sp.ones((I, T, K))
    for i in xrange(I):
        if len(vert_children[i]) != 0: 
            for t in xrange(T):                           
    #            if i==0:
                    #print 'wrong species index i'
                    #vc = vert_children[i]
                tmp = sp.zeros(K)
                for vc in vert_children[i]:
                    #evidence[i, t] *= lmds[vc][0, t, :]
                    tmp += sp.log(lmds[vc][0, t, :])
                evidence[i, t] = sp.exp(tmp) * lmds[i][-1, t+1,:]
                
#                if t < T-1:
#                    evidence[i, t] *= lmds[i][-1, t+1,:]
        else:
             for t in xrange(T-1):
                 evidence[i, t] = lmds[i][-1, t+1,:]
    return evidence


def bp_marginal_onenode(lmds, pis, args):
    """calculate marginal dist. of node i,t"""
    I, T, L = args.X.shape
    K = args.gamma.shape[0]
    marginal = sp.ones((I, T, K))
    emit_probs_mat  = sp.exp(args.log_obs_mat)
    emit_probs_mat /= emit_probs_mat.max(axis=2).reshape(I, T, 1)
    theta, alpha, beta, gamma, emit_probs, X = (args.theta, args.alpha, args.beta, args.gamma, args.emit_probs,
                                        args.X)
    evidence = evid_allchild(lmds, args.vert_children)
    for i in xrange(I):
        vp = args.vert_parent[i]

        for t in xrange(T):
            if i==0 and t==0:
                m = gamma *(emit_probs_mat[i, t, :] *evidence[i,t,:])
                #tmp1, tmp2 = sp.ix_(pis[i][-1,0,:], evidence[i,t+1,:]*emit_probs_mat[i, t+1, :])
                #m = (tmp1*tmp2 * alpha).sum(axis=1)
                #m /= m.sum()
                #breakpoint()
            else:
                if i == 0:
                    #tmp1, tmp2 = sp.ix_(pis[i][-1,t-1,:], emit_probs_mat[i, t, :]*evidence[i,t,:])
                    #tmp = alpha *(tmp1*tmp2)
                    #m = tmp.sum(axis=1)
                    #print m
                    tmp = sp.dot(pis[i][-1,t-1,:], alpha)
                    m = tmp * emit_probs_mat[i, t, :]*evidence[i,t,:]
             
                elif t == 0:
                    tmp = sp.dot(pis[vp][i-1,t,:], beta)
                    m = tmp* emit_probs_mat[i, t, :]*evidence[i,t,:]
                    
                    #tmp1, tmp2 = sp.ix_(pis[vp][i-1,t,:], emit_probs_mat[i, t, :]*evidence[i,t,:])
                    #tmp = beta *(tmp1*tmp2)
                    #m = tmp.sum(axis=0)
                else:
                    tmp1, tmp2, tmp3 = sp.ix_(pis[vp][i-1,t,:], pis[i][-1, t-1,:], emit_probs_mat[i, t, :]*evidence[i,t,:])
                    tmp= theta *(tmp1*tmp2*tmp3)
                    m = (tmp.sum(axis=0)).sum(axis=0)
                
            m /= m.sum()
            marginal[i,t,:] = m
    return marginal
    
def bp_update_params_new(args, renormalize=True):
    lmds, pis = args.lmds, args.pis
    vert_parent, vert_children = args.vert_parent, args.vert_children
    #print  pis[0].shape
    I, T, L = args.X.shape
    K = args.gamma.shape[0]
    theta, alpha, beta, gamma, emit_probs, X = (args.theta, args.alpha, args.beta, args.gamma, args.emit_probs,
                                        args.X)
    evidence = args.evidence #evid_allchild(lmds, vert_children)
    emit_probs_mat = sp.exp(args.log_obs_mat)
    #emit_probs_mat /= emit_probs_mat.max(axis=2).reshape(I, T, 1)
    gamma_p = copy.copy(gamma)
    alpha_p = copy.copy(alpha)
    beta_p =  copy.copy(beta)
    theta_p = copy.copy(theta)
#    emit_probs_p = copy.copy(emit_probs)
    
    theta[:] = args.pseudocount
    alpha[:] = args.pseudocount
    beta[:] = args.pseudocount
    gamma[:] = args.pseudocount
    emit_probs[:] = args.pseudocount

    #evidence = evid_allchild(lmds, args.vert_children)
    ##support = casual_support(pis)
    
    emit_sum = sp.zeros((K, L))
    for i in xrange(I):
        vp = vert_parent[i]
        if i != 0:
            idx_i = vert_children[vp].tolist().index(i)
        for t in xrange(T):
            if i==0 and t==0:
                gamma += emit_probs_mat[i, t, :]*evidence[i,t,:]
                Q = emit_probs_mat[i, t, :]*evidence[i,t,:]
            elif i == 0:
                tmp1, tmp2 = sp.ix_(pis[i][-1,t-1,:], emit_probs_mat[i, t, :]*evidence[i,t,:])
                tmp = alpha_p * (tmp1*tmp2) # belief
                #tmp /= tmp.sum()
                Q = tmp.sum(axis=0)
                alpha += tmp/tmp.sum()
            elif t == 0:
                tmp1, tmp2 = sp.ix_(pis[vp][idx_i,t,:], emit_probs_mat[i, t, :]*evidence[i,t,:]) # i-1->0
                tmp = beta_p *(tmp1*tmp2) # belief
                Q = tmp.sum(axis=0)
                beta += tmp/tmp.sum()
            else:
                tmp1, tmp2, tmp3 = sp.ix_(pis[vp][idx_i,t,:], pis[i][-1, t-1,:], emit_probs_mat[i, t, :]*evidence[i,t,:])
                tmp = theta_p *(tmp1*tmp2*tmp3) 
                Q = (tmp.sum(axis=0)).sum(axis=0)
                theta += tmp/tmp.sum()
                
            Q /= Q.sum()
            for l in xrange(L):
                if args.mark_avail[i,l] and X[i,t,l]:
                    emit_probs[:, l] += Q
                emit_sum[:,l] += Q
    if renormalize:
        normalize_trans(theta, alpha, beta, gamma)
        emit_probs[:] = sp.dot(sp.diag(1./emit_sum), emit_probs)
    args.emit_sum = emit_sum
    make_log_obs_matrix(args)
    
    
    
#def bp_marginal_onenode(lmds, pis, args):
#    """calculate marginal dist. of node i,t"""
#    I, T, L = args.X.shape
#    K = args.gamma.shape[0]
#    marginal = sp.ones((I, T, K))
#    emit_probs_mat  = sp.exp(args.log_obs_mat)
#    emit_probs_mat /= emit_probs_mat.max(axis=2).reshape(I, T, 1)
#    theta, alpha, beta, gamma, emit_probs, X = (args.theta, args.alpha, args.beta, args.gamma, args.emit_probs,
#                                        args.X)
#    evidence = evid_allchild(lmds, args.vert_children)
#    for i in xrange(I):
#        vp = args.vert_parent[i]
#
#        for t in xrange(T):
#            if i==0 and t==0:
#                m = gamma *(emit_probs_mat[i, t, :] *evidence[i,t,:])
#                #tmp1, tmp2 = sp.ix_(pis[i][-1,0,:], evidence[i,t+1,:]*emit_probs_mat[i, t+1, :])
#                #m = (tmp1*tmp2 * alpha).sum(axis=1)
#                #m /= m.sum()
#                #breakpoint()
#            else:
#                if i == 0:
#                    #tmp1, tmp2 = sp.ix_(pis[i][-1,t-1,:], emit_probs_mat[i, t, :]*evidence[i,t,:])
#                    #tmp = alpha *(tmp1*tmp2)
#                    #m = tmp.sum(axis=1)
#                    #print m
#                    tmp = sp.dot(pis[i][-1,t-1,:], alpha)
#                    m = tmp * emit_probs_mat[i, t, :]*evidence[i,t,:]
#             
#                elif t == 0:
#                    tmp = sp.dot(pis[vp][i-1,t,:], beta)
#                    m = tmp* emit_probs_mat[i, t, :]*evidence[i,t,:]
#                    
#                    #tmp1, tmp2 = sp.ix_(pis[vp][i-1,t,:], emit_probs_mat[i, t, :]*evidence[i,t,:])
#                    #tmp = beta *(tmp1*tmp2)
#                    #m = tmp.sum(axis=0)
#                else:
#                    tmp1, tmp2, tmp3 = sp.ix_(pis[vp][i-1,t,:], pis[i][-1, t-1,:], emit_probs_mat[i, t, :]*evidence[i,t,:])
#                    tmp= theta *(tmp1*tmp2*tmp3)
#                    m = (tmp.sum(axis=0)).sum(axis=0)
#                
#            m /= m.sum()
#            marginal[i,t,:] = m
#    return marginal


def bp_bethe_free_energy(args):
    lmds, pis = args.lmds, args.pis
    vert_parent, vert_children = args.vert_parent, args.vert_children
    theta, alpha, beta, gamma, emit_probs, X = (args.theta, args.alpha, args.beta, args.gamma, args.emit_probs,
                        args.X)
    I, T, L = X.shape
#    K = gamma.shape[0]
    free_e = 0.
    #entp = 0.
    log_theta, log_alpha, log_beta, log_gamma = sp.log(theta), sp.log(alpha), sp.log(beta), sp.log(gamma)
    emit_probs_mat  = sp.exp(args.log_obs_mat)
    evidence = evid_allchild(lmds, vert_children) #args.evidence
    
    
    ### replace start here
    #Q = bp_marginal_onenode(lmds, pis, args) # args.Q
    #Q_clq = sp.zeros((K,K))
    #Q_clq3 = sp.zeros((K,K,K))
    
    #log_emit_probs_mat = sp.zeros((K,T))
    for i in xrange(I):
        vp = vert_parent[i]
        if i != 0:
            idx_i = vert_children[vp].tolist().index(i)
            print vp, idx_i
        log_probs_mat_i = args.log_obs_mat[i,:,:].T
        if i==0:
            Qt =  gamma * emit_probs_mat[i,0,:]*evidence[i,0,:]
            Qt /= Qt.sum()
            free_e -= (Qt*log_gamma).sum() + (Qt*log_probs_mat_i[:, 0]).sum()
            free_e -= (Qt*sp.log(Qt)).sum()
            #for k in range(K):
            #    free_e -= Q[i,0,k]*(log_gamma[k] + log_probs_mat_i[k, 0] + log(Q[i,0,k]))
            
            for t in xrange(1 ,T):
                tmp1, tmp2 = sp.ix_(pis[i][-1,t-1,:], emit_probs_mat[i, t, :]*evidence[i,t,:])
                Q_clq = alpha * (tmp1*tmp2)
                Q_clq /= Q_clq.sum()
                Qt = Q_clq.sum(axis=0)
                #breakpoint()
                #free_e -= (Q_clq * log_alpha).sum() +(Qt*log_probs_mat_i[:, t]).sum()
                #free_e += (Q_clq * sp.log(Q_clq)).sum() -2.*(Qt*sp.log(Qt)).sum()
                
                free_e -= (Qt * (log_probs_mat_i[:, t]+ 2.*sp.log(Qt))).sum()
                free_e += (Q_clq * (sp.log(Q_clq) -log_alpha)).sum()
                #entp += (Q_clq * sp.log(Q_clq)).sum() -2.*(Q*sp.log(Q)).sum()
                #Q_clq_sum = 0.
                #for k1 in range(K):
                #    for k2 in range(K):
                #        Q_clq[k1,k2] = alpha[k1,k2]* pis[i][-1,t-1,k1] * emit_probs_mat[i,t,k2]*evidence[i,t,k2]
                #        Q_clq_sum += Q_clq[k1,k2]
                #
                #Q_clq /= Q_clq_sum
                #
                #for k1 in range(K):
                #    free_e -= Q[i,t,k1]*log_probs_mat_i[k1, t] +2.*Q[i,t,k1]*log(Q[i,t,k1])
                #    for k2 in range(K):
                #        free_e -= Q_clq[k1,k2] * log_alpha[k1,k2]
                #        free_e += Q_clq[k1,k2] * log(Q_clq[k1,k2])
        else:
            
            tmp1, tmp2 = sp.ix_(pis[vp][idx_i,0,:], emit_probs_mat[i, 0, :]*evidence[i,0,:])
            Q_clq = beta *(tmp1*tmp2)
            Q_clq /= Q_clq.sum()
            Qt = Q_clq.sum(axis=0)
            free_e -= (Q_clq * log_beta).sum() +(Qt * log_probs_mat_i[:, 0]).sum()
            free_e += (Q_clq * sp.log(Q_clq)).sum()
            free_e -= (Qt * sp.log(Qt)).sum()
            
            #Q_clq_sum = 0.
            #for k1 in xrange(K):   
            #    for k2 in xrange(K):
            #        Q_clq[k1,k2] = beta[k1,k2]* pis[vp][i-1,0,k1] * emit_probs_mat[i,0,k2]*evidence[i,0,k2]
            #        Q_clq_sum += Q_clq[k1,k2]
            #Q_clq /= Q_clq_sum
            #for k1 in xrange(K):
            #     free_e -= Q[i,0,k1] * (log_probs_mat_i[k1, 0] + log(Q[i,0,k1]) )
            #     for k2 in xrange(K):
            #         free_e += Q_clq[k1,k2] * (log(Q_clq[k1,k2]) - log_beta[k1,k2])
            
            
            #entp += (Q_clq * sp.log(Q_clq)).sum()-(Q * sp.log(Q)).sum()
            for t in xrange(1 ,T):
                tmp1, tmp2, tmp3 = sp.ix_(pis[vp][idx_i,t,:], pis[i][-1, t-1,:], emit_probs_mat[i, t, :]*evidence[i,t,:])
                Q_clq3 = theta *(tmp1*tmp2*tmp3)
                Q_clq3 /= Q_clq3.sum()
                Qt = (Q_clq3.sum(axis=0)).sum(axis=0)
                #breakpoint()
                free_e -= (Q_clq3 * log_theta).sum()
                free_e += (Q_clq3 * sp.log(Q_clq3)).sum() ###!!
                free_e -= (Qt * (sp.log(Qt)+log_probs_mat_i[:, t])).sum()
                #entp += (Q_clq3 * sp.log(Q_clq3)).sum() -(Qt * sp.log(Qt)).sum()
                #Q_clq3_sum = 0
                #for k1 in range(K):
                #    for k2 in xrange(K):
                #        for k3 in xrange(K):
                #            Q_clq3[k1,k2, k3] = theta[k1,k2, k3] * pis[vp][i-1,t,k1] * pis[i][-1, t-1,k2] * emit_probs_mat[i,t,k3]*evidence[i,t,k3]
                #            Q_clq3_sum += Q_clq3[k1,k2, k3]
                #Q_clq3 /=  Q_clq3_sum
                #free_e -= (Q_clq3 * log_theta).sum() +(Q[i,t,:] * log_probs_mat_i[:, t] ).sum()
                #free_e += (Q_clq3 * sp.log(Q_clq3)).sum() ###!!
                #free_e -= (Q[i,t,:]*sp.log(Q[i,t,:])).sum()
                #for k1 in xrange(K):    
                #    free_e -= Q[i,t,k1] *(log_probs_mat_i[k1,t] + log(Q[i,t,k1]))
                #    for k2 in xrange(K):
                #        for k3 in xrange(K):
                #            free_e -= Q_clq3[k1,k2,k3] * (log_theta[k1,k2,k3] - log(Q_clq3[k1,k2,k3]))
            
    #print 'free energy:', free_e
    return free_e

def bp_mf_free_energy(args):
    lmds, pis = args.lmds, args.pis
    vert_parent = args.vert_parent
    theta, alpha, beta, gamma, emit_probs, X = (args.theta, args.alpha, args.beta, args.gamma, args.emit_probs,
                        args.X)
    I, T, L = X.shape
    K = gamma.shape[0]
    log_theta, log_alpha, log_beta, log_gamma = sp.log(theta), sp.log(alpha), sp.log(beta), sp.log(gamma)
    log_obs_mat = args.log_obs_mat
    Q = bp_marginal_onenode(lmds, pis, args)
    entropy = (Q * sp.log(Q)).sum()
    #print 'mf entropy', -entropy
    total_free = entropy
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
                            total_free -= Q[i,t-1,v] * Q[i,t,k] * log_alpha[v,k]
                        elif t == 0:
                            total_free -= Q[vp,t,v] * Q[i,t,k] * log_beta[v,k]
                        else:
                            for h in xrange(K):
                                total_free -= Q[vp,t,v] * Q[i,t-1,h] * Q[i,t,k] * log_theta[v,h,k]
    #print 'mf free energy:', total_free
    return total_free

def bp_check_convergence(args):
    return check_conv(args.lmds_prev, args.pis_prev, args.lmds, args.pis)

def check_conv(lmds_prev, pis_prev, lmds, pis):
    I = lmds_prev.shape[0]
    #print_options.set_float_precision(2)
    tmp = abs(lmds - lmds_prev)
    #max_value = [lmds_prev[i].max() for i in xrange(I)]
    diff = [tmp[i].max() for i in xrange(I)]
    #print ["%0.3f" % i for i in max_value]
    tmp = abs(pis - pis_prev)
    #max_value = [lmds_prev[i].max() for i in xrange(I)]
    diff2 = [tmp[i].max() for i in xrange(I)]
    #print ["%5.4f" % i for i in diff]
    if all([diff[i] < 1e-2 for i in xrange(I)]) and all([diff2[i] < 1e-2 for i in xrange(I)]):
        a = True
    else:
        a = False
    return a

