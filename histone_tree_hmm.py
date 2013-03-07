#!/usr/bin/env python
"""
Variational Bayes method to solve phylgoenetic HMM for histone modifications

Need to:
* Preprocess
** load each dataset
** call significant sites for each dataset (vs. one control dataset)
** save out resulting histogrammed data

* Learn
** Init parameters randomly
** E step:  optimize each q_{ij} for fixed \Theta
** M step:  optimize \Theta for fixed Q


for a complete hg19, we have:
  T = 15,478,482
  I = 9
  K = 15
  L = 9
  \Theta is:
    e = K * 2**L
    \theta = K**2 * K
    \alpha = K * K
    \beta = K * K
    \gamma = K
  X = I * T * L  * 1 byte for bool => 2050 MB RAM

  for mf:
  Q = I * T * K  * 4 bytes for float64  =>
                15181614 * 9 * 15 * (4 bytes) / 1e6 = 8198 MB RAM
  \therefore should be okay for 12GB RAM

  for poc:
  \Theta = T * K * K * 4 bytes  => 30 GB RAM
  Q = I * T * K  * 4 bytes => 24 GB RAM
  Q_pairs = I * T * K * K * 4 bytes => :(

  Chromosome 1:
    T = 1246254
      =>  Q = .9 GB
      =>  Q_pairs = 8.9 GB
      =>  X = .1 GB

"""
import argparse
import sys
import operator
import glob
import urllib
import os
import hashlib
import multiprocessing
import time
from math import ceil
import cPickle as pickle
import copy
import re
import tarfile
from cStringIO import StringIO
import time
import random

try:
    import pysam
except ImportError:
    print 'pysam not installed.  Cannot convert data'
import scipy as sp
from scipy.stats import poisson
import scipy.io
import scipy.signal
try:
    import matplotlib
    matplotlib.use('Agg', warn=False)
    #matplotlib.rc('text', usetex=True)
    #matplotlib.rc('ps', usedistiller='xpdf')
    from matplotlib import pyplot
    allow_plots = True
except ImportError:
    allow_plots = False
    print 'matplotlib not installed.  Cannot plot!'

sp.seterr(all='raise')
sp.seterr(under='print')
#sp.random.seed([5])

import vb_mf
import vb_prodc
#import loopy_bp
import loopy_bp2 as loopy_bp
import clique_hmm
#import concatenate_hmm
import vb_gmtkExact_continuous as vb_gmtkExact
import vb_independent

from do_parallel import do_parallel_inference

float_type = sp.longdouble

#try:
#    from ipdb import set_trace as breakpoint
#except ImportError:
#    from pdb import set_trace as breakpoint


def timespan(start=[time.time()]):
    span = time.time() - start[0]
    start[0] = time.time()
    return span

def dependencies_for_freezing():
    from scipy.io.matlab import streams


###################### ENCODE human ######################
#valid_species = ['H1hesc', 'K562', 'Gm12878', 'Hepg2', 'Huvec', 'Hsmm', 'Nhlf', 'Nhek',
#                 'Hmec' ]  # 'Helas3' is extra
#valid_marks = ['Ctcf', 'H2az', 'H3k27ac', 'H3k27me3', 'H3k36me3', 'H3k4me1',
#            'H3k4me2', 'H3k4me3', 'H3k79me2', 'H3k9ac', 'H3k9me3', 'H4k20me1']
#valid_marks = ['Ctcf', 'H3k27me3', 'H3k36me3', 'H4k20me1', 'H3k4me1', 'H3k4me2', 'H3k4me3', 'H3k27ac',
#               'H3k9ac',  'Control',]  # no H2az or H3K9me3
#valid_marks = ['Ctcf', 'H3k27ac', 'H3k27me3', 'H3k36me3', 'H3k4me2', 'H3k4me3',
#               'H3k9ac', 'H4k20me1', 'Control',]  # H3K4me1 has an issue in Hepg2-- skipping for now
#
#phylogeny = {'H1hesc':'H1hesc', 'Huvec':'H1hesc', 'Hsmm':'H1hesc',
#             'Nhlf':'H1hesc', 'Gm12878':'H1hesc', 'K562':'H1hesc',
#             'Nhek':'H1hesc', 'Hmec':'H1hesc', 'Hepg2':'H1hesc'}

###################### modENCODE fly ######################
#valid_species = ['E0', 'E4', 'E8', 'E12', 'E16', 'E20', 'L1', 'L2', 'L3', 'Pupae', 'Adult female', 'Adult male']
#valid_marks = ['H3K27ac', 'H3K27me3', 'H3K4me1','H3K4me3','H3K9ac','H3K9me3']
#phylogeny = {'E4':'E0', 'E8':'E4', 'E12':'E8','E16':'E12', 'E20':'E16',
#             'L1':'E20','L2':'L1', 'L3':'L2', 'Pupae':'L3', 'Adult female':'Pupae', 'Adult male':'Adult female'}
#mark_avail = sp.ones((12,6), dtype = sp.int8)

###################### ENCODE mouse ######################
#valid_species = ['Progenitor', 'Ch12', 'Erythrobl', 'G1eer4e2', 'G1e', 'Megakaryo']
#valid_marks = ['H3k04me1', 'H3k04me3', 'H3k09me3', 'H3k27me3',  'H3k36me3', 'Input']
#phylogeny = {'Ch12':'Progenitor', 'Erythrobl':'Progenitor', 'G1eer4e2':'Progenitor','G1e':'Progenitor', 'Megakaryo':'Progenitor',
#            'G1eer4e2':'G1e'}
#
#mark_avail = sp.array([[0, 0, 0, 0, 0, 0],
#                       [1, 1, 1, 1, 1, 1],
#                       [1, 1, 1, 1, 1, 1],
#                       [1, 1, 1, 1, 1, 1],
#                       [1, 1, 1, 1, 1, 1],
#                       [1, 1, 1, 1, 1, 1]], dtype = sp.int8)

##################### ENCODE mouse2 ######################
#valid_species = ['Bmarrow', 'Ch12', 'Erythrobl', 'G1eer4e2', 'G1e', 'Megakaryo']
#phylogeny = {'Ch12':'Bmarrow', 'Erythrobl':'Bmarrow', 'G1eer4e2':'Bmarrow','G1e':'Bmarrow', 'Megakaryo':'Bmarrow',
#            'G1eer4e2':'G1e'}
#valid_marks = ['H3k27ac', 'H3k04me1', 'H3k04me3', 'H3k09me3', 'H3k27me3',  'H3k36me3', 'Input']
#mark_avail = sp.array([[1, 1, 1, 0, 0, 0, 1],
#                       [1, 1, 1, 1, 1, 1, 1],
#                       [0, 1, 1, 1, 1, 1, 1],
#                       [0, 1, 1, 1, 1, 1, 1],
#                       [0, 1, 1, 1, 1, 1, 1],
#                       [0, 1, 1, 1, 1, 1, 1]], dtype = sp.int8)


#valid_species = ['Bmarrow', 'G1e', 'G1eer4e2']
#valid_marks = ['H3k27ac','H3k04me1', 'H3k04me3', 'H3k09me3', 'H3k27me3', 'H3k36me3', 'Input']
#phylogeny = {'G1e':'Bmarrow', 'G1eer4e2':'G1e'}
###
#mark_avail = sp.array([[1, 1, 1, 0, 0, 0, 1],
#                       [0, 1, 1, 1, 1, 1, 1],
#                       [0, 1, 1, 1, 1, 1, 1]], dtype = sp.int8)



############### ENCODE Human with RNA-seq ################
valid_species = ['H1hesc', 'K562', 'Gm12878', 'Hepg2', 'Huvec', 'Hsmm', 'Nhlf', 'Nhek', 'Hmec' ]
valid_marks = ['Ctcf', 'H2az', 'H3k27ac', 'H3k27me3', 'H3k36me3', 'H3k4me1',
            'H3k4me2', 'H3k4me3', 'H3k79me2', 'H3k9ac', 'H3k9me3', 'H4k20me1']
#valid_marks = ['Ctcf', 'H3k27ac', 'H3k27me3', 'H3k36me3', 'H3k4me2', 'H3k4me3',
#               'H3k9ac', 'H4k20me1', 'Control',]  # H3K4me1 has an issue in Hepg2-- skipping for now
valid_marks += ['CellLongnonpolya', 'CellPap', 'CellTotal'
                'CytosolLongnonpolya', 'CytosolPap',
                'NucleusLongnonpolya', 'NucleusPap',
                'NucleoplasmTotal',
                'NucleolusTotal',
                'ChromatinTotal',]
mark_avail = sp.zeros((len(valid_species), len(valid_marks) * 3), dtype=sp.int8)
phylogeny = {'H1hesc':'H1hesc', 'Huvec':'H1hesc', 'Hsmm':'H1hesc',
             'Nhlf':'H1hesc', 'Gm12878':'H1hesc', 'K562':'H1hesc',
             'Nhek':'H1hesc', 'Hmec':'H1hesc', 'Hepg2':'H1hesc'}




#valid_species = range(20)
#valid_marks = range(20)
#phylogeny = {i: 0 for i in range(20)}

#valid_species = ['G1e', 'G1eer4e2']
#valid_marks = ['H3k04me1', 'H3k04me3', 'H3k09me3', 'H3k27me3', 'H3k36me3', 'Input']
#phylogeny = {'G1eer4e2':'G1e'}
#
#mark_avail = sp.ones((2,6), dtype = sp.int8)


#valid_marks = ['H3k27ac','H3k04me1', 'H3k04me3', 'H3k09me3', 'H3k27me3', 'H3k36me3', 'Input']
#mark_avail = sp.array([[0, 1, 1, 1, 1, 1, 1],
#                       [0, 1, 1, 1, 1, 1, 1],
#                       [0, 1, 1, 1, 1, 1, 1]], dtype = sp.int8)

#mark_avail = sp.ones((3,6), dtype = sp.int8)
inference_types = ['mf', 'poc', 'pot', 'clique', 'concat', 'loopy', 'gmtk', 'indep']


def main(argv=sys.argv[1:]):
    """run a variational EM algorithm"""
    # parse arguments, then call convert_data or do_inference
    parser = make_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, 'mark_avail'):
        args.mark_avail = mark_avail
    elif isinstance(args.mark_avail, basestring):
        args.mark_avail = sp.load(args.mark_avail)
    if args.func == do_inference:
        # allow patterns on the command line
        all_obs = []
        for obs in args.observe_matrix:
            all_obs.extend(glob.glob(obs))
        args.observe_matrix = all_obs

        if args.approx == 'gmtk':
            args.subtask = False
            obs_mat = args.observe_matrix
            args.observe_matrix = args.observe_matrix[0]
            init_args_for_inference(args)
            args.observe_matrix = obs_mat
            del args.func
            vb_gmtkExact.mark_avail = args.mark_avail
            vb_gmtkExact.run_gmtk_lineagehmm(args)
            return

        if len(args.observe_matrix) > 1:
            print 'parallel inference on %s jobs' % len(args.observe_matrix)
            args.func = do_parallel_inference
        else:
            args.subtask = False
            args.observe_matrix = args.observe_matrix[0]
            args.observe = os.path.split(args.observe_matrix)[1]
            args.func = do_inference
        if args.range_k is not None:
            out_template = args.out_dir + "_rangeK"
            for args.K in eval(args.range_k):
                print 'trying K=', args.K
                args.out_dir = out_template
                args.func(args)
            return

    args.func(args)  # do inference, downloading, or data conversion

def do_inference(args):
    """Perform the type of inference specified in args"""
    # set up
    if args.quiet_mode:
        sys.stdout = open('log_%s.log' , 'a')
    init_args_for_inference(args)
    print 'done making args'
    args.out_dir = args.out_dir.format(timestamp=time.strftime('%x_%X').replace('/','-'), **args.__dict__)

    try:
        print 'making', args.out_dir
        os.makedirs(args.out_dir)
    except OSError:
        pass

    if not args.subtask:
        args.iteration = '0_initial'
        plot_params(args)
        if args.plot_iter >= 2:
            plot_data(args)

    for i in xrange(1, args.max_iterations+1):
        if not args.subtask:
            args.iteration = i
        print 'iteration', i

        # run a few times rather than checking free energy
        for j in xrange(1, args.max_E_iter+1 if args.approx != 'clique' else 2):
            args.update_q_func(args)
            if args.approx !='loopy':
                f = args.free_energy_func(args)
                print 'free energy after %s E steps' % j, f
                try:
                    print abs(args.last_free_energy - f) / args.last_free_energy
                    if abs(abs(args.last_free_energy - f) / args.last_free_energy) < args.epsilon_e:
                        args.last_free_energy = f
                        break
                    args.last_free_energy = f
                except:  # no previous free energy
                    args.last_free_energy = f
            else:
                print 'loopy %s E steps' %j
                if loopy_bp.bp_check_convergence(args):
                    args.last_free_energy = f = abs(args.free_energy_func(args))
                    break
        print '# saving Q distribution'
        if args.continuous_observations:
            for k in range(args.K):
                print 'means[%s,:] = ' % k, args.means[k,:]
                print 'stdev[%s,:] = ' % k, sp.sqrt(args.variances[k,:])
        if args.save_Q >= 2:
            for p in args.Q_to_save:
                sp.save(os.path.join(args.out_dir,
                            args.out_params.format(param=p, **args.__dict__)),
                        args.__dict__[p])


        if args.subtask:
            # save the weights without renormalizing
            print 'saving weights for parameters'
            args.update_param_func(args, renormalize=False)
            #plot_Q(args)
            args.free_energy.append(args.last_free_energy)
            for p in args.params_to_save:
                sp.save(os.path.join(args.out_dir, args.out_params.format(param=p, **args.__dict__)),
                    args.__dict__[p])
            break
        else:
            # optimize parameters with new Q
            args.update_param_func(args)
            f = args.free_energy_func(args)
            try:
                if args.approx != 'clique':
                    print abs(args.last_free_energy - f) / args.last_free_energy
                    if abs(abs(args.last_free_energy - f) / args.last_free_energy) < args.epsilon:
                        args.last_free_energy = f
                        break
                    args.last_free_energy = f
            except:  # no previous free energy
                args.last_free_energy = f
            #args.last_free_energy = args.free_energy_func(args)
            args.free_energy.append(args.last_free_energy)
            print 'free energy after M-step', args.free_energy[-1]
            # save current parameter state
            for p in args.params_to_save:
                sp.save(os.path.join(args.out_dir,
                            args.out_params.format(param=p, **args.__dict__)),
                        args.__dict__[p])

            if args.plot_iter != 0 and i % args.plot_iter == 0:
                plot_params(args)
                plot_energy(args)
                if args.plot_iter >= 2:
                    plot_Q(args)
            #import ipdb; ipdb.set_trace()
            if args.compare_inf is not None:
                args.log_obs_mat = sp.zeros((args.I,args.T,args.K), dtype=float_type)
                vb_mf.make_log_obs_matrix(args)
                if 'mf' in args.compare_inf:
                    tmpargs = copy.deepcopy(args)
                    tmpargs.Q = vb_mf.mf_random_q(args.I,args.T,args.K)
                    print 'comparing '
                    for j in xrange(1, args.max_E_iter+1):
                        vb_mf.mf_update_q(tmpargs)
                        if vb_mf.mf_check_convergence(tmpargs):
                            break
                    print 'mf convergence after %s iterations' % j
                    e = vb_mf.mf_free_energy(tmpargs)
                    args.cmp_energy['mf'].append(e)
                if 'poc' in args.compare_inf:
                    tmpargs = copy.deepcopy(args)
                    if args.approx != 'poc':
                        tmpargs.Q, tmpargs.Q_pairs = vb_prodc.prodc_initialize_qs(args.theta, args.alpha, args.beta,
                                            args.gamma, args.emit_probs, args.X, args.log_obs_mat)
                    for j in xrange(1, args.max_E_iter+1):
                        vb_prodc.prodc_update_q(tmpargs)
                        if vb_mf.mf_check_convergence(tmpargs):
                            break
                    print 'poc convergence after %s iterations' % j
                    e = vb_prodc.prodc_free_energy(tmpargs)
                    args.cmp_energy['poc'].append(e)
                    #sp.io.savemat(os.path.join(args.out_dir, 'Artfdata_poc_inferred_params_K{K}_{T}.mat'.format(K=args.K, T=args.max_bins)), dict(alpha = args.alpha, theta=args.theta, beta=args.beta, gamma=args.gamma, emit_probs=args.emit_probs))
                if 'pot' in args.compare_inf:
                    #del args.cmp_energy['pot']
                    pass
                if 'concat' in args.compare_inf:
                    #del args.cmp_energy['concat']
                    pass
                if 'clique' in args.compare_inf:
                    tmpargs = copy.deepcopy(args)
                    if args.approx != 'clique':
                        clique_hmm.clique_init_args(tmpargs)
                    for j in xrange(1):
                        clique_hmm.clique_update_q(tmpargs)
                    e = clique_hmm.clique_likelihood(tmpargs)
                    args.cmp_energy['clique'].append(-e)
                if 'loopy' in args.compare_inf:
                    tmpargs = copy.deepcopy(args)
                    if args.approx != 'loopy':
                        tmpargs.lmds, tmpargs.pis = loopy_bp.bp_initialize_msg(args.I, args.T, args.K, args.vert_children)
                    for j in xrange(1, args.max_E_iter+1):
                        loopy_bp.bp_update_msg_new(tmpargs)
                        if loopy_bp.bp_check_convergence(tmpargs):
                            break
                    print 'loopy convergence after %s iterations' % j
                    e = loopy_bp.bp_bethe_free_energy(tmpargs)
                    #e = loopy_bp.bp_mf_free_energy(tmpargs)
                    args.cmp_energy['loopy'].append(e)
                if args.plot_iter != 0:
                    plot_energy_comparison(args)

    # save the final parameters and free energy to disk
    print 'done iteratin'
    if args.save_Q >= 1:
        for p in args.Q_to_save:
            sp.save(os.path.join(args.out_dir,
                        args.out_params.format(param=p, **args.__dict__)),
                    args.__dict__[p])
    for p in args.params_to_save:
        sp.save(os.path.join(args.out_dir, args.out_params.format(param=p, **args.__dict__)),
                args.__dict__[p])
    
    #pickle.dump(args, os.path.join(args.out_dir, args.out_params.format(param='args', **args.__dict__)))
    print 'done savin'
    if not args.subtask and args.plot_iter != 0:
        plot_energy(args)
        plot_params(args)
        plot_Q(args)
        #scipy.io.savemat('poc_inferred_params_K{K}_{T}.mat'.format(K=args.K, T=args.max_bins), dict(alpha = args.alpha, theta=args.theta, beta=args.beta, gamma=args.gamma, emit_probs=args.emit_probs))


def init_args_for_inference(args):
    """Initialize args with inference variables according to learning method"""
    # read in the datafiles to X array
    print '# loading observations'
    X = sp.load(args.observe_matrix)
    if args.max_bins is not None:
        X = X[:, :args.max_bins, :]
    if args.max_species is not None:
        X = X[:args.max_species, :, :]
    args.X = X
    args.I, args.T, args.L = X.shape
    if args.X.dtype != scipy.int8:
        args.continuous_observations = True
        print 'Inference for continuous observations'
        args.X = X.astype(float_type)
        #args.means = sp.rand(args.K, args.L)
        #args.variances = sp.rand(args.K, args.L)
        args.means, args.variances = initialize_mean_variance(args)
    else:
        args.continuous_observations = False
        print 'Inference for discrete observations'
    match = re.search(r'\.i(\d+)\.', args.observe_matrix)
    args.real_species_i = int(match.groups()[0]) if match and args.I == 1 else None
    args.free_energy = []

    make_tree(args)
    args.Q_to_save = ['Q']
    #if args.approx == 'poc':
    #    args.Q_to_save += ['Q_pairs']
    #elif args.approx == 'clique':
    #    args.Q_to_save += ['clq_Q', 'clq_Q_pairs']

    args.params_to_save = ['free_energy', 'alpha', 'gamma', 'last_free_energy']
    if True: #args.approx not in ['clique', 'concat']:
        args.params_to_save += ['theta', 'beta']
    if args.continuous_observations:
        args.params_to_save += ['means', 'variances']
    else:
        args.params_to_save += ['emit_probs', 'emit_sum']

    if args.compare_inf is not None:
        if 'all' in args.compare_inf:
            args.compare_inf = inference_types
        args.cmp_energy = dict((inf, []) for inf in args.compare_inf if inf not in ['pot', 'concat'])
        args.params_to_save += ['cmp_energy']

    if args.warm_start:  # need to load params
        print '# loading previous params for warm start from %s' % args.warm_start
        tmpargs = copy.deepcopy(args)
        tmpargs.out_dir = args.warm_start
        tmpargs.observe = 'all.npy'
        args.free_energy, args.theta, args.alpha, args.beta, args.gamma, args.emit_probs, args.emit_sum = load_params(tmpargs)
        try:
            args.free_energy = list(args.free_energy)
        except TypeError: # no previous free energy
            args.free_energy = []
        print 'done'
        #import pdb; pdb.set_trace()
	#args.iteration = -1
	#plot_params(args)
    elif args.subtask:  # params in args already
        print '# using previous params from parallel driver'
    else:
        print '# generating random parameters'
        (args.theta, args.alpha, args.beta, args.gamma, args.emit_probs) = \
                                                    random_params(args.K,args.L)
        if args.continuous_observations:
            del args.emit_probs


    if args.approx == 'mf':  # mean-field approximation
        if not args.subtask or args.iteration == 0:
            args.Q = vb_mf.mf_random_q(args.I,args.T,args.K)
        #else:
        #    q_path = os.path.join(args.out_dir, args.out_params.format(param='Q', **args.__dict__))
        #    print 'loading previous Q from %s' % q_path
        #    args.Q = sp.load(q_path)
        args.log_obs_mat = sp.zeros((args.I,args.T,args.K), dtype=float_type)
        vb_mf.make_log_obs_matrix(args)

        args.update_q_func = vb_mf.mf_update_q
        args.update_param_func = vb_mf.mf_update_params
        args.free_energy_func = vb_mf.mf_free_energy
        args.converged_func = vb_mf.mf_check_convergence
    elif args.approx == 'poc':  # product-of-chains approximation
        args.log_obs_mat = sp.zeros((args.I,args.T,args.K), dtype=float_type)
        if args.continuous_observations:
            vb_mf.make_log_obs_matrix_gaussian(args)
        else:
            vb_mf.make_log_obs_matrix(args)

        if not args.subtask or args.iteration == 0:
            print '# generating Qs'
            args.Q, args.Q_pairs = vb_prodc.prodc_initialize_qs(args.theta, args.alpha, args.beta,
                                            args.gamma, args.X, args.log_obs_mat)
        #else:
        #    q_path = os.path.join(args.out_dir, args.out_params.format(param='Q', **args.__dict__))
        #    print 'loading previous Q from %s' % q_path
        #    args.Q = sp.load(q_path)
        #    args.Q_pairs = sp.load(os.path.join(args.out_dir, args.out_params.format(param='Q_pairs', **args.__dict__)))

        args.update_q_func = vb_prodc.prodc_update_q
        args.update_param_func = vb_prodc.prodc_update_params
        args.free_energy_func = vb_prodc.prodc_free_energy
        args.converged_func = vb_mf.mf_check_convergence
    elif args.approx == 'indep':  # completely independent chains
        args.log_obs_mat = sp.zeros((args.I, args.T, args.K), dtype=float_type)
        if args.continuous_observations:
            vb_mf.make_log_obs_matrix_gaussian(args)
        else:
            vb_mf.make_log_obs_matrix(args)

        if not args.subtask or args.iteration == 0:
            print '# generating Qs'
            args.Q = sp.zeros((args.I, args.T, args.K), dtype=float_type)
            args.Q_pairs = sp.zeros((args.I, args.T, args.K, args.K), dtype=float_type)
            vb_independent.independent_update_qs(args)
        #else:
        #    q_path = os.path.join(args.out_dir, args.out_params.format(param='Q', **args.__dict__))
        #    print 'loading previous Q from %s' % q_path
        #    args.Q = sp.load(q_path)
        #    args.Q_pairs = sp.load(os.path.join(args.out_dir, args.out_params.format(param='Q_pairs', **args.__dict__)))

        args.update_q_func = vb_independent.independent_update_qs
        args.update_param_func = vb_independent.independent_update_params
        args.free_energy_func = vb_independent.independent_free_energy
        args.converged_func = vb_mf.mf_check_convergence
    elif args.approx == 'pot':  # product-of-trees approximation
        raise NotImplementedError("Product of Trees is not implemented yet!")
    elif args.approx == 'clique':
        print 'making cliqued Q'
        args.Q = sp.zeros((args.I, args.T, args.K), dtype=float_type)
        clique_hmm.clique_init_args(args)
        args.update_q_func = clique_hmm.clique_update_q
        args.update_param_func = clique_hmm.clique_update_params
        args.free_energy_func = clique_hmm.clique_likelihood
        args.converged_func = vb_mf.mf_check_convergence
    elif args.approx == 'concat':
        raise NotImplementedError("Concatenated HMM is not implemented yet!")
    elif args.approx == 'loopy':
        if not args.subtask or args.iteration == 0:
            args.Q = vb_mf.mf_random_q(args.I, args.T, args.K)
        #else:
        #    q_path = os.path.join(args.out_dir, args.out_params.format(param='Q', **args.__dict__))
        #    print 'loading previous Q from %s' % q_path
        #    args.Q = sp.load(q_path)
        args.lmds, args.pis = loopy_bp.bp_initialize_msg(args)
        args.log_obs_mat = sp.zeros((args.I,args.T,args.K), dtype=float_type)
        vb_mf.make_log_obs_matrix(args)

        args.update_q_func = loopy_bp.bp_update_msg_new
        args.update_param_func = loopy_bp.bp_update_params_new
        args.free_energy_func = loopy_bp.bp_bethe_free_energy
        #args.free_energy_func = loopy_bp.bp_mf_free_energy
        args.converged_func = loopy_bp.bp_check_convergence
    elif args.approx == 'gmtk':
        pass
    else:
        raise RuntimeError('%s not recognized as valid inference method!' % args.approx)

def distance(x1, x2):
    return scipy.sqrt((x1 - x2) * (x1 - x2)).sum()

def initialize_mean_variance(args):
    """Initialize the current mean and variance values semi-intelligently.

    Inspired by the kmeans++ algorithm: iteratively choose new centers from the data
    by weighted sampling, favoring points that are distant from those already chosen
    """
    X = args.X.reshape(args.X.shape[0] * args.X.shape[1], args.X.shape[2])

    # kmeans++ inspired choice
    centers = [random.choice(X)]
    min_dists = scipy.array([distance(centers[-1], x) for x in X])
    for l in range(1, args.K):
        weights = min_dists * min_dists
        new_center = weighted_sample(zip(weights, X), 1).next()
        centers.append(new_center)

        min_dists = scipy.fmin(min_dists, scipy.array([distance(centers[-1], x) for x in X]))

    means = scipy.array(centers)

    # for the variance, get the variance of the data in this cluster
    variances = []
    for c in centers:
        idxs = tuple(i for i, (x, m) in enumerate(zip(X, min_dists)) if distance(c, x) == m)
        v = scipy.var(X[idxs, :], axis=0)
        variances.append(v)
    variances = scipy.array(variances) + args.pseudocount

    #import pdb; pdb.set_trace()
    #for k in range(args.K):
    #    print sp.sqrt(variances[k,:])
    variances[variances < .1] = .1

    return means, variances


def weighted_sample(items, n):
    total = float(sum(w for w, v in items))
    i = 0
    w, v = items[0]
    while n:
        x = total * (1 - random.random() ** (1.0 / n))
        total -= x
        while x > w:
            x -= w
            i += 1
            w, v = items[i]
        w -= x
        yield v
        n -= 1


def make_parser():
    """Make a parser for variational inference"""
    parser = argparse.ArgumentParser()
    tasks_parser = parser.add_subparsers()

    # parameters for converting datasets from BAM to observation matrix
    convert_parser = tasks_parser.add_parser('convert', help='Convert BAM reads'
                                             ' into a matrix of observations')
    convert_parser.add_argument('--download_first', action='store_true',
                                help='Download the raw sequence data from UCSC,'
                                ' then convert it.')
    convert_parser.add_argument('--species', nargs='+', default=valid_species,
                                help='The set of species with observations. By '
                                'default, use all species: %(default)s')
    convert_parser.add_argument('--marks', nargs='+', default=valid_marks,
                                help='The set of marks with observations. By '
                                'default, use all histone marks: %(default)s')
    convert_parser.add_argument('--windowsize', type=int, default=200,
                        help='histogram bin size used in conversion')
    convert_parser.add_argument('--chromosomes', nargs='+', default='all',
                                help='which chromosomes to convert. By default,'
                                ' convert all autosomes')
    convert_parser.add_argument('--min_reads', type=float, default=.5,
                              help='The minimum number of reads for a region to be included. default: %(default)s')
    convert_parser.add_argument('--min_size', type=int, default=25,
                              help='The minimum length (in bins) to include a chunk. default: %(default)s')
    convert_parser.add_argument('--max_pvalue', type=float, default=1e-4,
                        help='p-value threshold to consider the read count'
                        ' significant, using a local poisson rate defined by'
                        ' the control data')
    convert_parser.add_argument('--outfile', default='observations.npy',
                                help='Where to save the binarized reads')
    #convert_parser.add_argument('--bam_template', help='bam file template.',
    #                default='wgEncodeBroadHistone{species}{mark}StdAlnRep*.bam')
    convert_parser.add_argument('--bam_template', help='bam file template.',
                default='wgEncode*{species}{mark}*Rep*.bam')
    convert_parser.set_defaults(func=convert_data_continuous_features_and_split)

    # to trim off telomeric regions
    trim_parser = tasks_parser.add_parser('trim', help='trim off regions without'
                                          'any observations in them')
    trim_parser.add_argument('observe_matrix', nargs='+',
                        help='Files to be trimmed (converted from bam'
                        ' using "%(prog)s convert" command).')
    trim_parser.set_defaults(func=trim_data)

    # to split a converted dataset into pieces
    split_parser = tasks_parser.add_parser('split', help='split observations '
                            'into smaller pieces, retaining only regions with '
                            'a smoothed minimum read count.')
    split_parser.add_argument('observe_matrix', nargs='+',
                        help='Files containing observed data (converted from bam'
                        ' using "%(prog)s convert" command). If multiple files '
                        'are specified, each is treated as its own chain but '
                        'the parameters are shared across all chains')
    #split_parser.add_argument('--chunksize', type=int, default=100000,
    #                          help='the number of bins per chunk. default: %(default)s')
    split_parser.add_argument('--min_reads', type=float, default=.5,
                              help='The minimum number of reads for a region to be included. default: %(default)s')
    split_parser.add_argument('--window_size', type=int, default=200,
                              help='The size of the gaussian smoothing window. default: %(default)s')
    split_parser.add_argument('--min_size', type=int, default=25,
                              help='The minimum length (in bins) to include a chunk. default: %(default)s')
    split_parser.set_defaults(func=split_data)

    # parameters for learning and inference with converted observations
    infer_parser = tasks_parser.add_parser('infer')
    infer_parser.add_argument('K', type=int, help='The number of hidden states'
                              ' to infer')
    infer_parser.add_argument('observe_matrix', nargs='+',
                        help='Files containing observed data (converted from bam'
                        ' using "%(prog)s convert" command). If multiple files '
                        'are specified, each is treated as its own chain but '
                        'the parameters are shared across all chains')
    infer_parser.add_argument('--approx', choices=inference_types,
                              default='mf',
                              help='Which approximation to make in inference')
    infer_parser.add_argument('--out_params', default='{approx}_{param}_{observe}',
                           help='Where to save final parameters')
    infer_parser.add_argument('--epsilon', type=float, default=1e-4,
                              help='Convergence criteria: change in Free energy'
                              ' during M step must be < epsilon')
    infer_parser.add_argument('--epsilon_e', type=float, default=1e-3,
                              help='Convergence criteria: change in Free energy'
                              ' during E step must be < epsilon')
    infer_parser.add_argument('--max_iterations', type=int, default=50,
                              help='Maximum number of EM steps before stopping')
    infer_parser.add_argument('--max_E_iter', type=int, default=10,
                              help='Maximum number of E steps per M step')
    infer_parser.add_argument('--max_bins', default=None, type=int,
                                help='Restrict the total number of bins (T)')
    infer_parser.add_argument('--max_species', default=None, type=int,
                                help='Restrict the total number of species (I)')
    infer_parser.add_argument('--pseudocount', type=float_type, default=1e-6,
                              help='pseudocount to add to each parameter matrix')
    infer_parser.add_argument('--plot_iter', type=int, default=1,
                              help='draw a plot per *plot_iter* iterations.'
                              '0 => plot only at the end. Default is %(default)s')
    infer_parser.add_argument('--out_dir', type=str, default='{run_name}_out/{approx}/I{I}_K{K}_T{T}_{timestamp}',
                              help='Output parameters and plots in this directory'
                              ' (default: %(default)s')
    infer_parser.add_argument('--run_name', type=str, default='infer',
                              help='name of current run type (default: %(default)s')
    infer_parser.add_argument('--num_processes', type=int, default=None,
                              help='Maximum number of processes to use '
                              'simultaneously (default: all)')
    infer_parser.add_argument('--warm_start', type=str, default=None,
                              help="Resume iterations using parameters and Q's "
                              "from a previous run. Q's that are not found are "
                              "regenerated")
    infer_parser.add_argument('--compare_inf', nargs='+', type=str, default=None, choices=inference_types + ['all'],
                              help="While learning using --approx method, "
                              "compare the inferred hidden states and energies "
                              "from these inference methods.")
    infer_parser.add_argument('--range_k', type=str, default=None,
                              help="perform inference over a range of K values. Argument is passed as range(*arg*)")
    infer_parser.add_argument('--save_Q', type=int, choices=[0,1,2,3], default=1,
                              help="Whether to save the inferred marginals for hidden variables. 0 => no saving, 1 => save at end, 2 => save at each iteration. 3 => for parallel jobs, reconstruct the chromsomal Q distribution at each iteration. Default: %(default)s")
    infer_parser.add_argument('--quiet_mode', action='store_true', help="Turn off printing for this run")
    infer_parser.add_argument('--run_local', action='store_true', help="Force parallel jobs to run on the local computer, even when SGE is available")
    infer_parser.add_argument('--mark_avail', help='npy matrix of available marks',
                                default=mark_avail)
    infer_parser.set_defaults(func=do_inference)
    return parser


def load_params(args):
    #print args.out_params
    #print args.__dict__.keys()
    #print 'loading from', os.path.join(args.out_dir, args.out_params.format(param='last_free_energy', **args.__dict__))
    free_energy = sp.load(os.path.join(args.out_dir, args.out_params.format(param='free_energy', **args.__dict__)))

    #print 'loading from', os.path.join(args.out_dir, args.out_params.format(param='theta', **args.__dict__))
    theta = sp.load(os.path.join(args.out_dir, args.out_params.format(param='theta', **args.__dict__)))

    #print 'loading from', os.path.join(args.out_dir, args.out_params.format(param='alpha', **args.__dict__))
    alpha = sp.load(os.path.join(args.out_dir, args.out_params.format(param='alpha', **args.__dict__)))

    #print 'loading from', os.path.join(args.out_dir, args.out_params.format(param='beta', **args.__dict__))
    beta = sp.load(os.path.join(args.out_dir, args.out_params.format(param='beta', **args.__dict__)))

    #print 'loading from', os.path.join(args.out_dir, args.out_params.format(param='gamma', **args.__dict__))
    gamma = sp.load(os.path.join(args.out_dir, args.out_params.format(param='gamma', **args.__dict__)))

    #print 'loading from', os.path.join(args.out_dir, args.out_params.format(param='emit_probs', **args.__dict__))
    emit_probs = sp.load(os.path.join(args.out_dir, args.out_params.format(param='emit_probs', **args.__dict__)))

    #print 'loading from', os.path.join(args.out_dir, args.out_params.format(param='emit_sum', **args.__dict__))
    emit_sum = sp.load(os.path.join(args.out_dir, args.out_params.format(param='emit_sum', **args.__dict__)))
    
    return free_energy, theta, alpha, beta, gamma, emit_probs, emit_sum

def plot_energy_comparison(args):
    """Plot energy trajectories for comparison"""
    outfile = (args.out_params + '.png').format(param='cmp_free_energy', **args.__dict__)
    pyplot.figure()
    names = args.cmp_energy.keys()
    vals = -sp.array([args.cmp_energy[n] for n in names]).T
    print names
    print vals
    lines = pyplot.plot(vals)
    line_types = ['--','-.',':','-','steps'] * 3
    [pyplot.setp(l, linestyle=t) for l,t in zip(lines, line_types)]
    pyplot.legend(names,loc='lower right')
    #pyplot.title('Free energy learning with %s' % args.approx)
    formatter = pyplot.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-3,3))
    pyplot.gca().yaxis.set_major_formatter(formatter)
    pyplot.xlabel("Iteration")
    pyplot.ylabel('-Free Energy')
    pyplot.savefig(os.path.join(args.out_dir, outfile))
    pyplot.close('all')

def plot_energy(args):
    #outfile = 'free_energy_{approx}.png'.format(**args.__dict__)
    outfile = (args.out_params + '.png').format(param='free_energy', **args.__dict__)
    pyplot.savefig(os.path.join(args.out_dir, outfile))
    pyplot.figure()
    pyplot.title('Free energy for %s' % args.approx)
    pyplot.plot([-f for f in args.free_energy], label='Current run')
    pyplot.xlabel("iteration")
    pyplot.ylabel('-Free Energy')
    pyplot.savefig(os.path.join(args.out_dir, outfile))
    pyplot.close('all')

    if hasattr(args, 'prev_free_energy'):
        pyplot.figure()
        pyplot.title('Free energy for %s' % args.approx)
        pyplot.plot(range(len(args.prev_free_energy)), [-f for f in args.prev_free_energy], linestyle='-', label='Previous run')
        pyplot.plot(range(len(args.prev_free_energy), len(args.free_energy) + len(args.prev_free_energy)), [-f for f in args.free_energy], label='Current run')
        pyplot.legend(loc='lower left')
        pyplot.xlabel("iteration")
        pyplot.ylabel('-Free Energy')
        pyplot.savefig(os.path.join(args.out_dir, (args.out_params + 'vs_previous.png').format(param='free_energy', **args.__dict__)))
        pyplot.close('all')
def plot_Q(args):
    """Plot Q distribution"""
    outfile = (args.out_params + '_it{iteration}.png').format(param='Q', **args.__dict__)
    print outfile
    I,T,K = args.Q.shape
    I,T,L = args.X.shape
    fig, axs = pyplot.subplots(I+1, 1, sharex=True, sharey=True, squeeze=False)
    for i in xrange(I):
        axs[i,0].plot(args.Q[i,:,:])
    mark_total = args.X.sum(axis=2).T
    axs[I,0].plot(mark_total / float(L))

    fig.suptitle("Q distribution for {approx} at iteration {iteration}".
                    format(approx=args.approx, iteration=args.iteration))
    fig.suptitle("chromosome bin", x=.5, y=.02)
    fig.suptitle("species", x=.02, y=.5, verticalalignment='center', rotation=90)
    #fig.savefig(os.path.join(args.out_dir, 'Q_dist_%s.png' % args.iteration))
    fig.savefig(os.path.join(args.out_dir, outfile))
    pyplot.close('all')

def plot_data(args):
    """Plot X as an interpolated image"""
    outfile = (args.out_params + '.png').format(param='X_colors', **args.__dict__)
    #fig, axs = pyplot.subplots(args.I+1, 1, sharex=True, sharey=True, squeeze=False)
    fig, axs = pyplot.subplots(args.I, 1, sharex=True, sharey=True, squeeze=False)
    fig.set_size_inches(24,20)
    I,T,L = args.X.shape
    extent = [0, T, 0, L]
    #extent = [0, 100, 0, 5]
    for i in xrange(args.I):
        im = axs[i,0].imshow(sp.flipud(args.X[i,:,:].T), interpolation='sinc', vmin=0, vmax=1, extent=extent, aspect='auto')
        im.set_cmap('spectral')
        axs[i,0].set_yticks(sp.linspace(0, L, L, endpoint=False) + .5)
        axs[i,0].set_yticklabels(valid_marks[:L])
        axs[i,0].text(T/2, L+1, valid_species[i], horizontalalignment='center', verticalalignment='top')

    fig.savefig(os.path.join(args.out_dir, outfile), dpi=120)
    pyplot.close('all')

def plot_params(args):
    """Plot alpha, theta, and the emission probabilities"""
    old_err = sp.seterr(under='ignore')
    oldsize = matplotlib.rcParams['font.size']
    K, L = args.emit_probs.shape if not args.continuous_observations else args.means.shape

    # alpha
    #matplotlib.rcParams['font.size'] = 12
    pyplot.figure()
    _, xedges, yedges = sp.histogram2d([0,K], [0,K], bins=[K,K])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    pyplot.imshow(args.alpha.astype(sp.float64), extent=extent, interpolation='nearest',
                  vmin=0, vmax=1,  cmap='OrRd', origin='lower')
    pyplot.xticks(sp.arange(K) + .5, sp.arange(K)+1)
    pyplot.gca().set_xticks(sp.arange(K)+1, minor=True)
    pyplot.yticks(sp.arange(K) + .5, sp.arange(K)+1)
    pyplot.gca().set_yticks(sp.arange(K)+1, minor=True)
    pyplot.grid(which='minor', alpha=.2)
    for line in pyplot.gca().yaxis.get_ticklines() + pyplot.gca().xaxis.get_ticklines() + pyplot.gca().xaxis.get_ticklines(minor=True) + pyplot.gca().yaxis.get_ticklines(minor=True):
    # label is a Text instance
        line.set_markersize(0)
    pyplot.ylabel('Horizontal parent state')
    pyplot.xlabel('Node state')
    pyplot.title(r"Top root transition ($\alpha$) for {approx} iteration {iteration}".
                        format(approx=args.approx, iteration=args.iteration))
    b = pyplot.colorbar(shrink=.9)
    b.set_label("Probability")
    outfile = (args.out_params + '_it{iteration}.png').format(param='alpha', **args.__dict__)
    pyplot.savefig(os.path.join(args.out_dir, outfile), dpi=240)


    # beta
    pyplot.figure()
    _, xedges, yedges = sp.histogram2d([0,K], [0,K], bins=[K,K])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    pyplot.clf()
    pyplot.imshow(args.beta.astype(sp.float64), extent=extent, interpolation='nearest',
                  vmin=0, vmax=1, cmap='OrRd', origin='lower')
    pyplot.xticks(sp.arange(K) + .5, sp.arange(K)+1)
    pyplot.gca().set_xticks(sp.arange(K)+1, minor=True)
    pyplot.yticks(sp.arange(K) + .5, sp.arange(K)+1)
    pyplot.gca().set_yticks(sp.arange(K)+1, minor=True)
    pyplot.grid(which='minor', alpha=.2)
    for line in pyplot.gca().yaxis.get_ticklines() + pyplot.gca().xaxis.get_ticklines() + pyplot.gca().xaxis.get_ticklines(minor=True) + pyplot.gca().yaxis.get_ticklines(minor=True):
    # label is a Text instance
        line.set_markersize(0)
    pyplot.ylabel('Vertical parent state')
    pyplot.xlabel('Node state')
    pyplot.title(r"Left root transition ($\beta$) for {approx} iteration {iteration}".
                        format(approx=args.approx, iteration=args.iteration))
    b = pyplot.colorbar(shrink=.9)
    b.set_label("Probability")
    outfile = (args.out_params + '_it{iteration}.png').format(param='beta', **args.__dict__)
    pyplot.savefig(os.path.join(args.out_dir, outfile), dpi=240)


    # theta
    for theta_name in ['theta'] + ['theta_%s' % i for i in range(20)]:
        #print 'trying', theta_name
        if not hasattr(args, theta_name):
            #print 'missing', theta_name
            continue
        _, xedges, yedges = sp.histogram2d([0,K], [0,K], bins=[K,K])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        if K == 18:
            numx_plots = 6
            numy_plots = 3
        elif K == 15:
            numx_plots = 5
            numy_plots = 3
        else:
            numx_plots = int(ceil(sp.sqrt(K)))
            numy_plots = int(ceil(sp.sqrt(K)))
        matplotlib.rcParams['font.size'] = 8
        fig, axs = pyplot.subplots(numy_plots, numx_plots, sharex=True, sharey=True, figsize=(numx_plots*2.5,numy_plots*2.5))
        for k in xrange(K):
            pltx, plty = k // numx_plots, k % numx_plots
            #axs[pltx,plty].imshow(args.theta[k,:,:], extent=extent, interpolation='nearest',
            axs[pltx,plty].imshow(getattr(args, theta_name)[:,k,:].astype(sp.float64), extent=extent, interpolation='nearest',
                          vmin=0, vmax=1, cmap='OrRd', aspect='auto', origin='lower')
            #if k < numx_plots:
            #axs[pltx,plty].text(0 + .5, K - .5, 'vp=%s' % (k+1), horizontalalignment='left', verticalalignment='top', fontsize=10)
            axs[pltx,plty].text(0 + .5, K - .5, 'hp=%s' % (k+1), horizontalalignment='left', verticalalignment='top', fontsize=10)
            #axs[pltx,plty].xticks(sp.arange(K) + .5, sp.arange(K))
            #axs[pltx,plty].yticks(sp.arange(K) + .5, sp.arange(K))
            axs[pltx,plty].set_xticks(sp.arange(K) + .5)
            axs[pltx,plty].set_xticks(sp.arange(K)+1, minor=True)
            axs[pltx,plty].set_xticklabels(sp.arange(K) + 1)
            axs[pltx,plty].set_yticks(sp.arange(K) + .5)
            axs[pltx,plty].set_yticks(sp.arange(K)+1, minor=True)
            axs[pltx,plty].set_yticklabels(sp.arange(K) + 1)
            for line in axs[pltx,plty].yaxis.get_ticklines() + axs[pltx,plty].xaxis.get_ticklines() + axs[pltx,plty].yaxis.get_ticklines(minor=True) + axs[pltx,plty].xaxis.get_ticklines(minor=True):
                line.set_markersize(0)
            axs[pltx,plty].grid(True, which='minor', alpha=.2)

        #fig.suptitle(r"$\Theta$ with fixed parents for {approx} iteration {iteration}".
        #                    format(approx=args.approx, iteration=args.iteration),
        #                    fontsize=14, verticalalignment='top')
        fig.suptitle('Node state', y=.03, fontsize=14, verticalalignment='center')
        #fig.suptitle('Horizontal parent state', y=.5, x=.02, rotation=90,
        fig.suptitle('Vertical parent state', y=.5, x=.02, rotation=90,
                     verticalalignment='center', fontsize=14)
        matplotlib.rcParams['font.size'] = 6.5
        fig.subplots_adjust(wspace=.05, hspace=.05, left=.05, right=.95)
        #b = fig.colorbar(shrink=.9)
        #b.set_label("Probability")
        outfile = (args.out_params + '_vertparent_it{iteration}.png').format(param=theta_name, **args.__dict__)
        pyplot.savefig(os.path.join(args.out_dir, outfile), dpi=240)


        fig, axs = pyplot.subplots(numy_plots, numx_plots, sharex=True, sharey=True, figsize=(numx_plots*2.5,numy_plots*2.5))
        for k in xrange(K):
            pltx, plty = k // numx_plots, k % numx_plots
            axs[pltx,plty].imshow(getattr(args, theta_name)[k,:,:].astype(sp.float64), extent=extent, interpolation='nearest',
            #axs[pltx,plty].imshow(args.theta[:,k,:], extent=extent, interpolation='nearest',
                          vmin=0, vmax=1, cmap='OrRd', aspect='auto', origin='lower')
            #if k < numx_plots:
            axs[pltx,plty].text(0 + .5, K - .5, 'vp=%s' % (k+1), horizontalalignment='left', verticalalignment='top', fontsize=10)
            #axs[pltx,plty].xticks(sp.arange(K) + .5, sp.arange(K))
            #axs[pltx,plty].yticks(sp.arange(K) + .5, sp.arange(K))
            axs[pltx,plty].set_xticks(sp.arange(K) + .5)
            axs[pltx,plty].set_xticks(sp.arange(K)+1, minor=True)
            axs[pltx,plty].set_xticklabels(sp.arange(K) + 1)
            axs[pltx,plty].set_yticks(sp.arange(K) + .5)
            axs[pltx,plty].set_yticks(sp.arange(K)+1, minor=True)
            axs[pltx,plty].set_yticklabels(sp.arange(K) + 1)
            for line in axs[pltx,plty].yaxis.get_ticklines() + axs[pltx,plty].xaxis.get_ticklines() + axs[pltx,plty].yaxis.get_ticklines(minor=True) + axs[pltx,plty].xaxis.get_ticklines(minor=True):
                line.set_markersize(0)
            axs[pltx,plty].grid(True, which='minor', alpha=.2)

        #fig.suptitle(r"$\Theta$ with fixed parents for {approx} iteration {iteration}".
        #                    format(approx=args.approx, iteration=args.iteration),
        #                    fontsize=14, verticalalignment='top')
        fig.suptitle('Node state', y=.03, fontsize=14, verticalalignment='center')
        fig.suptitle('Horizontal parent state', y=.5, x=.02, rotation=90,
        #fig.suptitle('Vertical parent state', y=.5, x=.02, rotation=90,
                     verticalalignment='center', fontsize=14)
        matplotlib.rcParams['font.size'] = 6.5
        fig.subplots_adjust(wspace=.05, hspace=.05, left=.05, right=.95)
        #b = fig.colorbar(shrink=.9)
        #b.set_label("Probability")
        outfile = (args.out_params + '_it{iteration}.png').format(param=theta_name, **args.__dict__)
        pyplot.savefig(os.path.join(args.out_dir, outfile), dpi=240)


    # emission probabilities
    if args.continuous_observations:
        # plot mean values
        matplotlib.rcParams['font.size'] = 8
        pyplot.figure(figsize=(max(1,round(L/3.)),max(1, round(K/3.))))
        print (max(1,round(L/3.)),max(1, round(K/3.)))
        pyplot.imshow(args.means.astype(sp.float64), interpolation='nearest', aspect='auto',
                      vmin=0, vmax=args.means.max(), cmap='OrRd', origin='lower')
        for k in range(K):
            for l in range(L):
                pyplot.text(l, k, '%.1f' % (args.means[k,l]), horizontalalignment='center', verticalalignment='center', fontsize=5)
        pyplot.yticks(sp.arange(K), sp.arange(K)+1)
        pyplot.gca().set_yticks(sp.arange(K)+.5, minor=True)
        pyplot.xticks(sp.arange(L), valid_marks, rotation=30, horizontalalignment='right')
        pyplot.gca().set_xticks(sp.arange(L)+.5, minor=True)
        pyplot.grid(which='minor', alpha=.2)
        for line in pyplot.gca().yaxis.get_ticklines() + pyplot.gca().xaxis.get_ticklines() + pyplot.gca().xaxis.get_ticklines(minor=True) + pyplot.gca().yaxis.get_ticklines(minor=True):
        # label is a Text instance
            line.set_markersize(0)
        pyplot.ylabel('Hidden State')
        pyplot.title("Emission Mean")
        #b = pyplot.colorbar(shrink=.7)
        #b.set_label("Probability")
        outfile = (args.out_params + '_it{iteration}.png').format(param='emission_means', **args.__dict__)
        pyplot.savefig(os.path.join(args.out_dir, outfile), dpi=240)

        # plot variances
        pyplot.figure(figsize=(max(1,round(L/3.)),max(1, round(K/3.))))
        print (L/3,K/3.)
        pyplot.imshow(args.variances.astype(sp.float64), interpolation='nearest', aspect='auto',
                      vmin=0, vmax=args.variances.max(), cmap='OrRd', origin='lower')
        for k in range(K):
            for l in range(L):
                pyplot.text(l, k, '%.1f' % (args.variances[k,l]), horizontalalignment='center', verticalalignment='center', fontsize=5)
        pyplot.yticks(sp.arange(K), sp.arange(K)+1)
        pyplot.gca().set_yticks(sp.arange(K)+.5, minor=True)
        pyplot.xticks(sp.arange(L), valid_marks, rotation=30, horizontalalignment='right')
        pyplot.gca().set_xticks(sp.arange(L)+.5, minor=True)
        pyplot.grid(which='minor', alpha=.2)
        for line in pyplot.gca().yaxis.get_ticklines() + pyplot.gca().xaxis.get_ticklines() + pyplot.gca().xaxis.get_ticklines(minor=True) + pyplot.gca().yaxis.get_ticklines(minor=True):
        # label is a Text instance
            line.set_markersize(0)
        pyplot.ylabel('Hidden State')
        pyplot.title("Emission Variance")
        #b = pyplot.colorbar(shrink=.7)
        #b.set_label("Probability")
        outfile = (args.out_params + '_it{iteration}.png').format(param='emission_variances', **args.__dict__)
        pyplot.savefig(os.path.join(args.out_dir, outfile), dpi=240)
    else:
        matplotlib.rcParams['font.size'] = 8
        pyplot.figure(figsize=(max(1,round(L/3.)),max(1, round(K/3.))))
        print (L/3,K/3.)
        pyplot.imshow(args.emit_probs.astype(sp.float64), interpolation='nearest', aspect='auto',
                      vmin=0, vmax=1, cmap='OrRd', origin='lower')
        for k in range(K):
            for l in range(L):
                pyplot.text(l, k, '%2.0f' % (args.emit_probs[k,l] * 100), horizontalalignment='center', verticalalignment='center')
        pyplot.yticks(sp.arange(K), sp.arange(K)+1)
        pyplot.gca().set_yticks(sp.arange(K)+.5, minor=True)
        pyplot.xticks(sp.arange(L), valid_marks, rotation=30, horizontalalignment='right')
        pyplot.gca().set_xticks(sp.arange(L)+.5, minor=True)
        pyplot.grid(which='minor', alpha=.2)
        for line in pyplot.gca().yaxis.get_ticklines() + pyplot.gca().xaxis.get_ticklines() + pyplot.gca().xaxis.get_ticklines(minor=True) + pyplot.gca().yaxis.get_ticklines(minor=True):
        # label is a Text instance
            line.set_markersize(0)
        pyplot.ylabel('Hidden State')
        pyplot.title("Emission probabilities")
        #b = pyplot.colorbar(shrink=.7)
        #b.set_label("Probability")
        outfile = (args.out_params + '_it{iteration}.png').format(param='emission', **args.__dict__)
        pyplot.savefig(os.path.join(args.out_dir, outfile), dpi=240)


    #broad_paper_enrichment = sp.array([[16,2,2,6,17,93,99,96,98,2],
    #                               [12,2,6,9,53,94,95,14,44,1],
    #                               [13,72,0,9,48,78,49,1,10,1],
    #                               [11,1,15,11,96,99,75,97,86,4],
    #                               [5,0,10,3,88,57,5,84,25,1],
    #                               [7,1,1,3,58,75,8,6,5,1],
    #                               [2,1,2,1,56,3,0,6,2,1],
    #                               [92,2,1,3,6,3,0,0,1,1],
    #                               [5,0,43,43,37,11,2,9,4,1],
    #                               [1,0,47,3,0,0,0,0,0,1],
    #                               [0,0,3,2,0,0,0,0,0,0],
    #                               [1,27,0,2,0,0,0,0,0,0],
    #                               [0,0,0,0,0,0,0,0,0,0],
    #                               [22,28,19,41,6,5,26,5,13,37],
    #                               [85,85,91,88,76,77,91,73,85,78],
    #                               [float('nan'), float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan'),float('nan')]
    #                            ]) / 100.
    #mapping_from_broad = dict(zip(range(K), (5,2,0,14,4,6,9,1,12,-1,3,12,8,7,10,12,11,13)))
    #broad_paper_enrichment = broad_paper_enrichment[tuple(mapping_from_broad[i] for i in range(K)), :]
    #broad_names = ['Active promoter', 'Weak promoter', 'Inactive/poised promoter', 'Strong enhancer',
    #               'Strong enhancer', 'weak/poised enhancer', 'Weak/poised enhancer', 'Insulator',
    #               'Transcriptional transition', 'Transcriptional elongation', 'Weak transcribed',
    #               'Polycomb repressed', 'Heterochrom; low signal', 'Repetitive/CNV', 'Repetitive/CNV',
    #               'NA', 'NA', 'NA']
    #pyplot.figure(figsize=(L/3,K/3.))
    #print (L/3,K/3.)
    #pyplot.imshow(broad_paper_enrichment, interpolation='nearest', aspect='auto',
    #              vmin=0, vmax=1, cmap='OrRd', origin='lower')
    #for k in range(K):
    #    for l in range(L):
    #        pyplot.text(l, k, '%2.0f' % (broad_paper_enrichment[k,l] * 100), horizontalalignment='center', verticalalignment='center')
    #    pyplot.text(L, k, broad_names[mapping_from_broad[k]], horizontalalignment='left', verticalalignment='center', fontsize=6)
    #pyplot.yticks(sp.arange(K), sp.arange(K)+1)
    #pyplot.gca().set_yticks(sp.arange(K)+.5, minor=True)
    #pyplot.xticks(sp.arange(L), valid_marks, rotation=30, horizontalalignment='right')
    #pyplot.gca().set_xticks(sp.arange(L)+.5, minor=True)
    #pyplot.grid(which='minor', alpha=.2)
    #for line in pyplot.gca().yaxis.get_ticklines() + pyplot.gca().xaxis.get_ticklines() + pyplot.gca().xaxis.get_ticklines(minor=True) + pyplot.gca().yaxis.get_ticklines(minor=True):
    ## label is a Text instance
    #    line.set_markersize(0)
    #pyplot.ylabel('Hidden State')
    #pyplot.title("Broad paper Emission probabilities")
    ##b = pyplot.colorbar(shrink=.7)
    ##b.set_label("Probability")
    #pyplot.subplots_adjust(right=.7)
    #outfile = (args.out_params + '_broadpaper.png').format(param='emission', **args.__dict__)
    #pyplot.savefig(os.path.join(args.out_dir, outfile), dpi=240)

    pyplot.close('all')
    sp.seterr(**old_err)
    matplotlib.rcParams['font.size'] = oldsize

def make_tree(args):
    """build a tree from the vertical parents specified in args"""
    I = args.I
    # define the tree structure
    #tree_by_parents = {0:sp.inf, 1:0, 2:0}  # 3 species, 2 with one parent
    #tree_by_parents = {0:sp.inf, 1:0}  # 3 species, 2 with one parent
    #tree_by_parents = dict((args.species.index(k), args.species.index(v)) for k, v in phylogeny.items())
    tree_by_parents = dict((valid_species.index(k), valid_species.index(v)) for k, v in phylogeny.items() if valid_species.index(k) in xrange(I) and valid_species.index(v) in xrange(I))
    tree_by_parents[0] = sp.inf #'Null'
    print tree_by_parents.keys()
    # [inf, parent(1), parent(2), ...]
    global vert_parent
    #I = max(tree_by_parents) + 1

    vert_parent = sp.array([tree_by_parents[c] if c > 0 else I for c in
                                    xrange(I)], dtype=sp.int8)  # error if 0's parent is accessed
    args.vert_parent = vert_parent

    print 'vert_parent', vert_parent
#    args.vert_parent = tree_by_parents
    # {inf:0, 0:[1,2], 1:[children(1)], ...}
    global vert_children
    vert_children = dict((pa, []) for pa in
                            tree_by_parents.keys())# + tree_by_parents.values())
    for pa in tree_by_parents.values():
        for ch in tree_by_parents.keys():
            if tree_by_parents[ch] == pa:
                if pa not in vert_children:
                    vert_children[pa] = []
                if ch not in vert_children[pa]:
                    vert_children[pa].append(ch)
    del vert_children[sp.inf]
    for pa in vert_children:
        vert_children[pa] = sp.array(vert_children[pa], dtype=sp.int32)
    args.vert_children = vert_children

#    vert_children = sp.ones(I,  dtype = 'object')
#    for pa in range(I):
#        vert_children[pa] = []
#        for child, parent in tree_by_parents.items():
#            if pa == parent:
#                vert_children[pa].append(child)
#    print vert_children
#    args.vert_children = vert_children
def random_params(K, L):
    """Create and normalize random parameters for Mean-Field inference"""
    #sp.random.seed([5])
    theta = sp.rand(K, K, K).astype(float_type)
    alpha = sp.rand(K, K).astype(float_type)
    beta = sp.rand(K, K).astype(float_type)
    gamma = sp.rand(K).astype(float_type)
    emit_probs = sp.rand(K, L).astype(float_type)
    vb_mf.normalize_trans(theta, alpha, beta, gamma)
    return theta, alpha, beta, gamma, emit_probs

def trim_data(args):
    """Trim regions without any observations from the start and end of the
    obervation matrices
    """
    for f in args.observe_matrix:
        print '# trimming ', f, 'start is ',
        X = sp.load(f).astype(sp.int8)
        S = X.cumsum(axis=0).cumsum(axis=2)  # any species has any observation
        for start_t in xrange(X.shape[1]):
            if S[-1, start_t, -1] > 0:
                break
        for end_t in xrange(X.shape[1] - 1, -1, -1):
            if S[-1, end_t, -1] > 0:
                break
        tmpX = X[:, start_t:end_t, :]
        print start_t
        sp.save(os.path.splitext(f)[0] + '.trimmed', tmpX)

def split_data(args):
    """Split the given observation matrices into smaller chunks"""
    sizes = []
    total_size = 0
    covered_size = 0
    start_positions = {}
    for f in args.observe_matrix:
        print '# splitting ', f
        X = sp.load(f).astype(sp.int8)
        total_size += X.shape[1]
        #start_ts = xrange(0, X.shape[1], args.chunksize)
        #end_ts = xrange(args.chunksize, X.shape[1] + args.chunksize, args.chunksize)
        
        density = X.sum(axis=0).sum(axis=1)  # sumation over I, then L
        #from ipdb import set_trace; set_trace()
        gk = _gauss_kernel(args.window_size)
        smoothed_density = scipy.signal.convolve(density, gk, mode='same')
        regions_to_keep = smoothed_density >= args.min_reads
        
        # find the regions where a transition is made from no reads to reads, and reads to no reads
        start_ts = sp.where(sp.diff(regions_to_keep.astype(sp.int8)) > 0)[0]
        end_ts = sp.where(sp.diff(regions_to_keep.astype(sp.int8)) < 0)[0]
        
        cur_regions = [r for r in zip(start_ts, end_ts) if r[1] - r[0] >= args.min_size]
        sizes.extend([end_t - start_t for start_t, end_t in cur_regions])
        
        print 'saving %s regions' % len(sizes)
        for chunknum, (start_t, end_t) in enumerate(cur_regions):
            covered_size += end_t - start_t
            tmpX = X[:, start_t:end_t, :]
            name = os.path.splitext(f)[0] + '.chunk%s.npy' % chunknum
            sp.save(name, tmpX)
            start_positions[name] = start_t
    print '# plotting size distribution'
    pyplot.figure()
    pyplot.figtext(.5,.01,'%s regions; %s bins total; %s bins covered; coverage = %.3f' % (len(sizes),total_size, covered_size, covered_size / float(total_size)), ha='center')
    pyplot.hist(sizes, bins=100)
    pyplot.title('chunk sizes for all chroms, min_reads %s, min_size %s, window_size %s' % (args.min_reads, args.min_size, args.window_size))
    pyplot.savefig('chunk_sizes.minreads%s.minsize%s.windowsize%s.png' % (args.min_reads, args.min_size, args.window_size))
    
    pickle.dump(start_positions, open('start_positions.pkl', 'w'))
    # --min_reads .5 --min_size 25 --window_size 200;

    
        
def extract_local_features(args):
    """extract some local features from the given data, saving an X array with extra dimensions"""
    sizes = []
    total_size = 0
    covered_size = 0
    start_positions = {}
    for f in args.observe_matrix:
        print '# features on ', f
        X = sp.load(f).astype(sp.int8)
        total_size += X.shape[1]
        #start_ts = xrange(0, X.shape[1], args.chunksize)
        #end_ts = xrange(args.chunksize, X.shape[1] + args.chunksize, args.chunksize)

        density = X.sum(axis=0).sum(axis=1)  # summation over I, then L
        #from ipdb import set_trace; set_trace()
        gk = _gauss_kernel(args.window_size)
        smoothed_density = scipy.signal.convolve(density, gk, mode='same')
        regions_to_keep = smoothed_density >= args.min_reads

        # find the regions where a transition is made from no reads to reads, and reads to no reads
        start_ts = sp.where(sp.diff(regions_to_keep.astype(sp.int8)) > 0)[0]
        end_ts = sp.where(sp.diff(regions_to_keep.astype(sp.int8)) < 0)[0]

        cur_regions = [r for r in zip(start_ts, end_ts) if r[1] - r[0] >= args.min_size]
        sizes.extend([end_t - start_t for start_t, end_t in cur_regions])

        print 'saving %s regions' % len(sizes)
        for chunknum, (start_t, end_t) in enumerate(cur_regions):
            covered_size += end_t - start_t
            tmpX = X[:, start_t:end_t, :]
            name = os.path.splitext(f)[0] + '.chunk%s.npy' % chunknum
            sp.save(name, tmpX)
            start_positions[name] = start_t
    print '# plotting size distribution'
    pyplot.figure()
    pyplot.figtext(.5,.01,'%s regions; %s bins total; %s bins covered; coverage = %.3f' % (len(sizes),total_size, covered_size, covered_size / float(total_size)), ha='center')
    pyplot.hist(sizes, bins=100)
    pyplot.title('chunk sizes for all chroms, min_reads %s, min_size %s, window_size %s' % (args.min_reads, args.min_size, args.window_size))
    pyplot.savefig('chunk_sizes.minreads%s.minsize%s.windowsize%s.png' % (args.min_reads, args.min_size, args.window_size))

    pickle.dump(start_positions, open('start_positions.pkl', 'w'))
    # --min_reads .5 --min_size 25 --window_size 200;



def convert_data_continuous_features_and_split(args):
    """histogram both treatment and control data as specified by args
    This saves the complete X matrix

    This version doesn't binarize the data, smooths out the read signal (gaussian convolution)
    and adds derivative information
    """
    if args.download_first:
        download_data(args)
    I = len(args.species)
    L = len(args.marks)
    final_data = None
    total_size = 0
    covered_size = 0
    start_positions = {}
    # make sure all the data is present...
    for species in args.species:
        for mark in args.marks:
            d_files = [f for f in glob.glob(args.bam_template.format(
                                                species=species, mark=mark))]
            if len(d_files) == 0:
                print("No histone data for species %s mark %s Expected: %s" %
                              (species, mark, args.bam_template.format(
                                                species=species, mark=mark)))

    for i, species in enumerate(args.species):
        for l, mark in enumerate(args.marks):
            d_obs = []
            d_files = [f for f in glob.glob(args.bam_template.format(
                                                species=species, mark=mark))]
            if len(d_files) == 0:
                args.mark_avail[i, l] = 0
            else:
                args.mark_avail[i, l] = 1
                for mark_file in d_files:
                    try:
                        d_obs.append(histogram_reads(mark_file, args.windowsize,
                                                    args.chromosomes))
                    except ValueError as e:
                        print e.message
                    print d_obs[-1].sum()
                    print d_obs[-1].shape
                d_obs = reduce(operator.add, d_obs)  # add all replicants together
                #print 'before per million:', d_obs.sum()
                #d_obs /= (d_obs.sum() / 1e7)  # convert to reads mapping per ten million
                # convert to a binary array with global poisson
                #genome_rate = d_obs / (d_obs.sum() / 1e6)

                if final_data is None:
                    final_data = sp.zeros((I, len(d_obs), L), dtype=sp.float32)
                final_data[i, :, l] = d_obs
                total_size = final_data.shape[1]


    regions_to_keep = (final_data[:, :, tuple(range(L))].sum(axis=0).sum(axis=1) >= args.min_reads).astype(sp.int8)
    # find the regions where a transition is made from no reads to reads, and reads to no reads
    start_ts = sp.where(sp.diff(regions_to_keep) > 0)[0]
    end_ts = sp.where(sp.diff(regions_to_keep) < 0)[0]

    cur_regions = [r for r in zip(start_ts, end_ts) if r[1] - r[0] >= args.min_size]
    sizes = [end_t - start_t for start_t, end_t in cur_regions]

    print 'saving %s regions' % len(sizes)
    tarout = tarfile.open(args.outfile + '.tar.gz', 'w:gz')
    for chunknum, (start_t, end_t) in enumerate(cur_regions):
        covered_size += end_t - start_t
        tmpX = final_data[:, start_t:end_t, :]

        print 'adding chunk', chunknum, 'of', len(cur_regions)
        s = StringIO()
        sp.save(s, tmpX)
        name = args.outfile + '.chunk%s.npy' % chunknum
        info = tarfile.TarInfo(name)
        info.size = s.tell(); info.mtime = time.time()
        s.seek(0)
        tarout.addfile(info, s)
        start_positions[name] = start_t

    print '# plotting size distribution'
    pyplot.figure()
    pyplot.figtext(.5,.01,'%s regions; %s bins total; %s bins covered; coverage = %.3f' % (len(sizes),total_size, covered_size, covered_size / float(total_size)), ha='center')
    pyplot.hist(sizes, bins=100)
    pyplot.title('chunk sizes for all chroms, min_reads %s, min_size %s, windowsize %s' % (args.min_reads, args.min_size, args.windowsize))
    pyplot.savefig('chunk_sizes.minreads%s.minsize%s.windowsize%s.png' % (args.min_reads, args.min_size, args.windowsize))

    s = StringIO()
    pyplot.savefig(s)
    info = tarfile.TarInfo('chunk_sizes.minreads%s.minsize%s.windowsize%s.png' % (args.min_reads, args.min_size, args.windowsize))
    info.size = s.tell(); info.mtime = time.time()
    s.seek(0)
    tarout.addfile(info, s)

    s = StringIO()
    pickle.dump(start_positions, s)
    info = tarfile.TarInfo('start_positions.pkl')
    info.size = s.tell(); info.mtime = time.time()
    s.seek(0)
    tarout.addfile(info, s)
    pickle.dump(start_positions, open('start_positions.pkl', 'w'))
    # --min_reads .5 --min_size 25 --window_size 200;


    s = StringIO()
    sp.save(s, args.mark_avail)
    info = tarfile.TarInfo('available_marks.npy')
    info.size = s.tell(); info.mtime = time.time()
    s.seek(0)
    tarout.addfile(info, s)
    pickle.dump(start_positions, open('start_positions.pkl', 'w'))
    # --min_reads .5 --min_size 25 --window_size 200;

    tarout.close()


    print "output file:", args.outfile
    print 'available marks:', args.mark_avail
    #with open(args.outfile, 'wb') as outfile:
    #    sp.save(outfile, final_data)
    with open(args.outfile + '.available_marks', 'wb') as outfile:
        sp.save(outfile, args.mark_avail)




def convert_data_continuous_features_and_split_old(args):
    """histogram both treatment and control data as specified by args
    This saves the complete X matrix

    This version doesn't binarize the data, smooths out the read signal (gaussian convolution)
    and adds derivative information
    """
    if args.download_first:
        download_data(args)
    I = len(args.species)
    L = len(args.marks)
    final_data = None
    total_size = 0
    covered_size = 0
    start_positions = {}
    # make sure all the data is present...
    for species in args.species:
        for mark in args.marks:
            d_files = [f for f in glob.glob(args.bam_template.format(
                                                species=species, mark=mark))]
            if len(d_files) == 0:
                print("No histone data for species %s mark %s Expected: %s" %
                              (species, mark, args.bam_template.format(
                                                species=species, mark=mark)))

    for i, species in enumerate(args.species):
        for l, mark in enumerate(args.marks):
            l = l * 3
            d_obs = []
            d_files = [f for f in glob.glob(args.bam_template.format(
                                                species=species, mark=mark))]
            if len(d_files) == 0:
                args.mark_avail[i, l] = 0
                args.mark_avail[i, l+1] = 0
                args.mark_avail[i, l+2] = 0
            else:
                args.mark_avail[i, l] = 1
                args.mark_avail[i, l+1] = 1
                args.mark_avail[i, l+2] = 1
                for mark_file in d_files:
                    try:
                        d_obs.append(histogram_reads(mark_file, args.windowsize,
                                                    args.chromosomes))
                    except ValueError as e:
                        print e.message
                    print d_obs[-1].sum()
                    print d_obs[-1].shape
                d_obs = reduce(operator.add, d_obs)  # add all replicants together
                #print 'before per million:', d_obs.sum()
                #d_obs /= (d_obs.sum() / 1e7)  # convert to reads mapping per ten million
                # convert to a binary array with global poisson
                genome_rate = d_obs / (d_obs.sum() / 1e6)

                if final_data is None:
                    final_data = sp.zeros((I, len(d_obs), L * 3), dtype=sp.float32)
                asinh_obs = sp.log(genome_rate + sp.sqrt(genome_rate * genome_rate + 1))
                gk = _gauss_kernel(3)
                smoothed_obs = scipy.signal.convolve(asinh_obs, gk, mode='same')
                smooth_deriv = sp.gradient(smoothed_obs)
                smooth_deriv2 = sp.gradient(smooth_deriv)
                final_data[i, :, l] = smoothed_obs
                final_data[i, :, l + 1] = smooth_deriv
                final_data[i, :, l + 2] = smooth_deriv2
                total_size = final_data.shape[1]


    regions_to_keep = (final_data[:, :, tuple(range(0, L * 3, 3))].sum(axis=0).sum(axis=1) >= args.min_reads).astype(sp.int8)
    # find the regions where a transition is made from no reads to reads, and reads to no reads
    start_ts = sp.where(sp.diff(regions_to_keep) > 0)[0]
    end_ts = sp.where(sp.diff(regions_to_keep) < 0)[0]

    cur_regions = [r for r in zip(start_ts, end_ts) if r[1] - r[0] >= args.min_size]
    sizes = [end_t - start_t for start_t, end_t in cur_regions]

    print 'saving %s regions' % len(sizes)
    tarout = tarfile.open(args.outfile + '.tar.gz', 'w:gz')
    for chunknum, (start_t, end_t) in enumerate(cur_regions):
        covered_size += end_t - start_t
        tmpX = final_data[:, start_t:end_t, :]

        print 'adding chunk', chunknum, 'of', len(cur_regions)
        s = StringIO()
        sp.save(s, tmpX)
        name = args.outfile + '.chunk%s.npy' % chunknum
        info = tarfile.TarInfo(name)
        info.size = s.tell(); info.mtime = time.time()
        s.seek(0)
        tarout.addfile(info, s)
        start_positions[name] = start_t

    print '# plotting size distribution'
    pyplot.figure()
    pyplot.figtext(.5,.01,'%s regions; %s bins total; %s bins covered; coverage = %.3f' % (len(sizes),total_size, covered_size, covered_size / float(total_size)), ha='center')
    pyplot.hist(sizes, bins=100)
    pyplot.title('chunk sizes for all chroms, min_reads %s, min_size %s, windowsize %s' % (args.min_reads, args.min_size, args.windowsize))
    pyplot.savefig('chunk_sizes.minreads%s.minsize%s.windowsize%s.png' % (args.min_reads, args.min_size, args.windowsize))

    s = StringIO()
    pyplot.savefig(s)
    info = tarfile.TarInfo('chunk_sizes.minreads%s.minsize%s.windowsize%s.png' % (args.min_reads, args.min_size, args.windowsize))
    info.size = s.tell(); info.mtime = time.time()
    s.seek(0)
    tarout.addfile(info, s)

    s = StringIO()
    pickle.dump(start_positions, s)
    info = tarfile.TarInfo('start_positions.pkl')
    info.size = s.tell(); info.mtime = time.time()
    s.seek(0)
    tarout.addfile(info, s)
    pickle.dump(start_positions, open('start_positions.pkl', 'w'))
    # --min_reads .5 --min_size 25 --window_size 200;


    s = StringIO()
    sp.save(s, args.mark_avail)
    info = tarfile.TarInfo('available_marks.npy')
    info.size = s.tell(); info.mtime = time.time()
    s.seek(0)
    tarout.addfile(info, s)
    pickle.dump(start_positions, open('start_positions.pkl', 'w'))
    # --min_reads .5 --min_size 25 --window_size 200;

    tarout.close()


    print "output file:", args.outfile
    print 'available marks:', args.mark_avail
    #with open(args.outfile, 'wb') as outfile:
    #    sp.save(outfile, final_data)
    with open(args.outfile + '.available_marks', 'wb') as outfile:
        sp.save(outfile, args.mark_avail)



def split_data(args):
    """Split the given observation matrices into smaller chunks"""
    sizes = []
    total_size = 0
    covered_size = 0
    start_positions = {}
    for f in args.observe_matrix:
        print '# splitting ', f
        X = sp.load(f).astype(sp.int8)
        total_size += X.shape[1]
        #start_ts = xrange(0, X.shape[1], args.chunksize)
        #end_ts = xrange(args.chunksize, X.shape[1] + args.chunksize, args.chunksize)

        density = X.sum(axis=0).sum(axis=1)  # sumation over I, then L
        #from ipdb import set_trace; set_trace()
        gk = _gauss_kernel(args.window_size)
        smoothed_density = scipy.signal.convolve(density, gk, mode='same')
        regions_to_keep = smoothed_density >= args.min_reads

        # find the regions where a transition is made from no reads to reads, and reads to no reads
        start_ts = sp.where(sp.diff(regions_to_keep.astype(sp.int8)) > 0)[0]
        end_ts = sp.where(sp.diff(regions_to_keep.astype(sp.int8)) < 0)[0]

        cur_regions = [r for r in zip(start_ts, end_ts) if r[1] - r[0] >= args.min_size]
        sizes.extend([end_t - start_t for start_t, end_t in cur_regions])

        print 'saving %s regions' % len(sizes)
        for chunknum, (start_t, end_t) in enumerate(cur_regions):
            covered_size += end_t - start_t
            tmpX = X[:, start_t:end_t, :]
            name = os.path.splitext(f)[0] + '.chunk%s.npy' % chunknum
            sp.save(name, tmpX)
            start_positions[name] = start_t
    print '# plotting size distribution'
    pyplot.figure()
    pyplot.figtext(.5,.01,'%s regions; %s bins total; %s bins covered; coverage = %.3f' % (len(sizes),total_size, covered_size, covered_size / float(total_size)), ha='center')
    pyplot.hist(sizes, bins=100)
    pyplot.title('chunk sizes for all chroms, min_reads %s, min_size %s, window_size %s' % (args.min_reads, args.min_size, args.window_size))
    pyplot.savefig('chunk_sizes.minreads%s.minsize%s.windowsize%s.png' % (args.min_reads, args.min_size, args.window_size))

    pickle.dump(start_positions, open('start_positions.pkl', 'w'))
    # --min_reads .5 --min_size 25 --window_size 200;






def _gauss_kernel(winsize):
    x = sp.mgrid[-int(winsize):int(winsize)+1]
    g = sp.exp(-(x**2/float(winsize)))
    return g / g.sum()

def convert_data(args):
    """histogram both treatment and control data as specified by args
    This saves the complete X matrix
    """
    if args.download_first:
        download_data(args)
    I = len(args.species)
    L = len(args.marks)
    final_data = None
    # make sure all the data is present...
    for species in args.species:
        for mark in args.marks:
            d_files = [f for f in glob.glob(args.bam_template.format(
                                                species=species, mark=mark))]
            if len(d_files) == 0:
                print("No histone data for species %s mark %s Expected: %s" %
                              (species, mark, args.bam_template.format(
                                                species=species, mark=mark)))

    for i, species in enumerate(args.species):
        for l, mark in enumerate(args.marks):

            d_obs = []
            d_files = [f for f in glob.glob(args.bam_template.format(
                                                species=species, mark=mark))]
            if len(d_files) == 0:
                pass
            else:
                for mark_file in d_files:
                    d_obs.append(histogram_reads(mark_file, args.windowsize,
                                                 args.chromosomes))
                    print d_obs[-1].sum()
                    print d_obs[-1].shape
                d_obs = reduce(operator.add, d_obs)  # add all replicants together
                #print 'before per million:', d_obs.sum()
                #d_obs /= (d_obs.sum() / 1e7)  # convert to reads mapping per ten million
                # convert to a binary array with global poisson
                genome_rate = d_obs.sum() / float(len(d_obs))
                print 'after per million', len(d_obs), d_obs.sum(), genome_rate
                d_obs = call_significant_sites(d_obs, genome_rate, args.max_pvalue)
                if final_data is None:
                    final_data = sp.zeros((I, len(d_obs), L), dtype=sp.int8)
                print 'bg_rate', genome_rate, 'total_above', d_obs.sum()
                final_data[i, :, l] = d_obs
    print "output file:", args.outfile
    with open(args.outfile, 'wb') as outfile:
        sp.save(outfile, final_data)

def download_data(args):
    """Download any missing histone modification data from UCSC and check md5s.
    """
    base_url = ('http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/'
                'wgEncodeBroadHistone/%s')
    md5s = urllib.urlopen(base_url % 'md5sum.txt').read().strip().split('\n')
    md5s = dict(reversed(l.strip().split()) for l in md5s)
    for species in args.species:
        for mark in args.marks:
            for rep in range(5):
                fname = args.bam_template.format(species=species,
                                            mark=mark).replace('*', str(rep))
                if fname not in md5s:
                    continue
                if os.path.exists(fname):
                    #m = hashlib.md5(open(fname, 'rb').read()).hexdigest()
                    #if m != md5s[fname]:  # destroy if md5 doesn't match
                    #    print 'removing incomplete file: %s' % fname
                    #    print m, md5s[fname]
                    #    os.unlink(fname)
                    #else:
                    #    print 'skipping already downloaded %s' % fname
                        continue
                with open(fname, 'wb') as outfile:
                    try:
                        print 'downloading %s' % fname
                        page = urllib.urlopen(base_url % fname)
                        while True:
                            data = page.read(81920)
                            if not data:
                                break
                            outfile.write(data)
                    except RuntimeError as e:
                        print 'Skipping...', e.message



def histogram_reads(bam_file, windowsize, chromosomes='all'):
    """Histogram the counts along bam_file, resulting in a vector.

    This will concatenate all chromosomes, together, so to get the
    counts for a particular chromosome, pass it as a list, a la
    >>> histogram_reads(my_bam_file, chromosomes=['chr1'])
    """
    print 'histogramming', bam_file
    reads_bam = pysam.Samfile(bam_file, 'rb')
    # get the chromosome name and lengths for our subset
    if chromosomes == 'all':
        chromosomes = filter(lambda c: c not in ['chrM', 'chrY', 'chrX'],
                             reads_bam.references)
        actual_chromosomes = list(set(chromosomes)) # uniquefy the chroms
        chrom_lengths = [reads_bam.lengths[reads_bam.references.index(c)]
                                       for c in chromosomes]
        if len(actual_chromosomes) != len(chromosomes):
            print 'SANITY CHECK... removing the first half of the chromosome bin for %s' % bam_file
            chrom_lengths = chrom_lengths[len(chrom_lengths)/2:]
    else:
        chromosomes = filter(lambda c: c not in ['chrM', 'chrY', 'chrX'],
                             chromosomes)
        chrom_lengths = [reads_bam.lengths[reads_bam.references.index(c)]
                                       for c in chromosomes if c not in
                                                    ['chrM', 'chrY', 'chrX']]
    # offset of each chromosome into concatenated chrom bins
    chrom_ends = list(((sp.array(chrom_lengths) // windowsize) + 1).cumsum())
    chrom_starts = dict(zip(chromosomes, [0] + chrom_ends[:-1]))

    # create the histogram: 1 x sum(lengths) array
    read_counts = sp.zeros(chrom_ends[-1], dtype=float_type)

    # count the reads in the input
    for read in reads_bam:
        #if (read.is_qcfail or read.is_unmapped or read.is_secondary or
        #        read.is_duplicate or read.mapq == 0):
        #    continue  # filter out non-mapping reads
        chrom = reads_bam.references[read.tid]
        if chrom in chrom_starts:  # chrom requested?
            offset = chrom_starts[chrom]
            if read.is_paired:
                if read.is_proper_pair:
                    # add at the middle of the mates
                    bin = offset + ((read.pos + read.mpos +
                                     read.rlen) / 2) // windowsize
                    read_counts[min(chrom_ends[-1] - 1, bin)] += 1.
            else:
                # add at the middle of the fragment
                bin = offset + (read.pos + 100) // windowsize
                read_counts[min(chrom_ends[-1] - 1, bin)] += 1.
    return read_counts


def call_significant_sites(fg_counts, bg_counts, max_pvalue):
    """binarize fg_counts (significant=1) using bg_counts as a local poisson
    rate. the poisson survival must be < sig_level.
    """
    print fg_counts.max(), fg_counts.min(), bg_counts
    return poisson.sf(fg_counts, bg_counts) < max_pvalue

if __name__ == '__main__':
    main()
