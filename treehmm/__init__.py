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


from treehmm.static import valid_species, valid_marks, mark_avail, phylogeny, inference_types, float_type

from treehmm import vb_mf
from treehmm import vb_prodc
#import loopy_bp
from treehmm import loopy_bp
from treehmm import clique_hmm
#import concatenate_hmm
from treehmm import vb_gmtkExact_continuous as vb_gmtkExact
from treehmm import vb_independent
from treehmm.plot import plot_params, plot_data, plot_Q, plot_energy, plot_energy_comparison

from treehmm.do_parallel import do_parallel_inference




def main(argv=sys.argv[1:]):
    """run a variational EM algorithm"""
    # parse arguments, then call convert_data or do_inference
    parser = make_parser()
    args = parser.parse_args(argv)
    global phylogeny
    args.phylogeny = eval(args.phylogeny)
    phylogeny = args.phylogeny
    if not hasattr(args, 'mark_avail'):
        args.mark_avail = mark_avail
    elif isinstance(args.mark_avail, basestring):
        args.mark_avail = sp.load(args.mark_avail)
    if args.func == do_inference:
        # allow patterns on the command line
        all_obs = []
        for obs_pattern in args.observe_matrix:
            obs_files = glob.glob(obs_pattern)
            if len(obs_files) == 0:
                parser.error('No files matched the pattern %s' % obs_pattern)
            all_obs.extend(obs_files)
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
                    #e = loopy_bp.bp_bethe_free_energy(tmpargs)
                    e = loopy_bp.bp_mf_free_energy(tmpargs)
                    args.cmp_energy['loopy'].append(e)
                if args.plot_iter != 0:
                    plot_energy_comparison(args)

    # save the final parameters and free energy to disk
    print 'done iteration'
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
        #tmpargs.observe = 'all.npy'
        args.free_energy, args.theta, args.alpha, args.beta, args.gamma, args.emit_probs, args.emit_sum = load_params(tmpargs)
        try:
            args.free_energy = list(args.free_energy)
        except TypeError: # no previous free energy
            args.free_energy = []
        print 'done'
    elif args.subtask:  # params in args already
        print '# using previous params from parallel driver'
    else:
        print '# generating random parameters'
        (args.theta, args.alpha, args.beta, args.gamma, args.emit_probs) = \
                                                    random_params(args.I, args.K, args.L, args.separate_theta)
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
        if not args.separate_theta:
            import vb_prodc
        else:
            import vb_prodc_sepTheta as vb_prodc
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
        if args.separate_theta:
           raise RuntimeError('separate_theta not implemented yet for clique')
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
        if args.separate_theta:
           raise RuntimeError('separate_theta not implemented yet for clique')
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
        #args.free_energy_func = loopy_bp.bp_bethe_free_energy
        args.free_energy_func = loopy_bp.bp_mf_free_energy
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
    convert_parser.add_argument('--base_url', default='http://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeBroadHistone/%s',
                                help='When downloading, string-format the template into this url.')
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
    convert_parser.add_argument('--outfile', default='observations.{chrom}.npy',
                                help='Where to save the binarized reads')
    #convert_parser.add_argument('--bam_template', help='bam file template.',
    #                default='wgEncodeBroadHistone{species}{mark}StdAlnRep*.bam')
    convert_parser.add_argument('--bam_template', help='bam file template. default: %(default)s',
                default='wgEncode*{species}{mark}StdAlnRep{repnum}.bam')
    convert_parser.set_defaults(func=convert_data)

    # # to trim off telomeric regions
    # trim_parser = tasks_parser.add_parser('trim', help='trim off regions without'
    #                                       'any observations in them')
    # trim_parser.add_argument('observe_matrix', nargs='+',
    #                     help='Files to be trimmed (converted from bam'
    #                     ' using "%(prog)s convert" command).')
    # trim_parser.set_defaults(func=trim_data)

    # to split a converted dataset into pieces
    split_parser = tasks_parser.add_parser('split', help='split observations '
                            'into smaller pieces, retaining only regions with '
                            'a smoothed minimum read count.')
    split_parser.add_argument('observe_matrix', nargs='+',
                        help='Files containing observed data (converted from bam'
                        ' using "%(prog)s convert" command). If multiple files '
                        'are specified, each is treated as its own chain but '
                        'the parameters are shared across all chains')
    split_parser.add_argument('start_positions', help='start_positions.pkl file generated during `convert` step.')
    #split_parser.add_argument('--chunksize', type=int, default=100000,
    #                          help='the number of bins per chunk. default: %(default)s')
    split_parser.add_argument('--min_reads', type=float, default=.5,
                              help='The minimum number of reads for a region to be included. default: %(default)s')
    split_parser.add_argument('--gauss_window_size', type=int, default=200,
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
    infer_parser.add_argument('--separate_theta', action='store_true', help='use a separate theta matrix for each node of the tree (only works for GMTK)')
    infer_parser.add_argument('--mark_avail', help='npy matrix of available marks',
                                default=mark_avail)
    infer_parser.add_argument('--phylogeny', help='the phylogeny connecting each species, as a python dictionary with children for keys and parents for values. Note: this does not have to be a singly-rooted or even a bifurcating phylogeny!  You may specify multiple trees, chains, stars, etc, but should not have loops in the phylogeny.',
                                default=str(phylogeny))
    infer_parser.add_argument('--chunksize', help='The number of chunks (for convert+split data) or chromosomes (for convert only) to submit to each runner.  When running on SGE, you should set this number relatively high (in 100s?) since each job has a very slow startup time. When running locally, this is the number of chunks each subprocess will handle at a time.',
                                default=1)
    infer_parser.set_defaults(func=do_inference)


    bed_parser = tasks_parser.add_parser('q_to_bed')
    bed_parser.add_argument('q_root_dir', help='Root directory for the Q outputs to convert. '
                            'Should look something like: infer_out/mf/<timestamp>/')
    bed_parser.add_argument('start_positions', help='the pickled offsets generated by `tree-hmm convert` or `tree-hmm split`.',)
    bed_parser.add_argument('--bed_template', help='template for bed output files. Default: %(default)s',
                            default='treehmm_states.{species}.state{state}.bed')
    bed_parser.add_argument('--save_probs', action='store_true', help='Instead of saving the most likely state for each bin, record the probability of being in that state at each position. NOTE: this will greatly increase the BED file size!')
    bed_parser.set_defaults(func=q_to_bed)
    return parser


def q_to_bed(args):
    attrs = pickle.load(open(args.start_positions))
    windowsize = attrs['windowsize']
    start_positions = attrs['start_positions']
    valid_species = attrs['valid_species']
    valid_marks = attrs['valid_marks']
    
    outfiles = {}
    for f in glob.glob(os.path.join(args.q_root_dir, '*_Q_*.npy')):
        Q = scipy.load(f)
        if not args.save_probs:
            best_states = Q.argmax(axis=2)
        obs = f.split('_Q_')[1]
        I, T, K = Q.shape
        chrom, bin_offset = start_positions[obs]
        for i in range(I):
            for t in range(T):
                if args.save_probs:
                    for k in range(K):
                        bedline = '\t'.join([chrom, str((bin_offset + t) * windowsize),
                                             str((bin_offset + t + 1) * windowsize),
                                             '{species}.state{k}'.format(species=valid_species[i], k=k), 
                                             str(Q[i,t,k]), '+']) + '\n'
                        if (i,k) not in outfiles:
                            outfiles[(i,k)] = open(args.bed_template.format(species=valid_species[i], state=k), 'w')
                        outfiles[(i,k)].write(bedline)
                else:
                    k = best_states[i,t]
                    bedline = '\t'.join([chrom, str((bin_offset + t) * windowsize),
                                         str((bin_offset + t + 1) * windowsize),
                                         '{species}.state{k}'.format(species=valid_species[i], k=k), 
                                         str(Q[i,t,k]), '+']) + '\n'
                    if (i,k) not in outfiles:
                        outfiles[(i,k)] = open(args.bed_template.format(species=valid_species[i], state=k), 'w')
                    outfiles[(i,k)].write(bedline)


def load_params(args):
    #print args.out_params
    #print args.__dict__.keys()
    #print 'loading from', os.path.join(args.out_dir, args.out_params.format(param='last_free_energy', **args.__dict__))
    free_energy = sp.load(os.path.join(args.out_dir, args.out_params.format(param='free_energy', **args.__dict__)))

    #print 'loading from', os.path.join(args.out_dir, args.out_params.format(param='theta', **args.__dict__))
    theta = sp.load(os.path.join(args.out_dir, args.out_params.format(param='theta', **args.__dict__)))
    if len(theta.shape)==3 and args.separate_theta:
        tmp = sp.zeros((args.I-1, args.K, args.K, args.K), dtype=float_type)
        for i in range(args.I-1):
            tmp[i,:,:,:] = theta
        theta = tmp
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



def make_tree(args):
    """build a tree from the vertical parents specified in args"""
    I = args.I
    # define the tree structure
    #tree_by_parents = {0:sp.inf, 1:0, 2:0}  # 3 species, 2 with one parent
    #tree_by_parents = {0:sp.inf, 1:0}  # 3 species, 2 with one parent
    #tree_by_parents = dict((args.species.index(k), args.species.index(v)) for k, v in phylogeny.items())
    tree_by_parents = dict((valid_species.index(k), valid_species.index(v)) 
                            for k, v in args.phylogeny.items() if 
                                valid_species.index(k) in xrange(I) and 
                                valid_species.index(v) in xrange(I))
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

def random_params(I, K, L, separate_theta):
    """Create and normalize random parameters for inference"""
    #sp.random.seed([5])
    if separate_theta:
        theta = sp.rand(I-1, K, K, K).astype(float_type)
    else:
        theta = sp.rand(K, K, K).astype(float_type)
    alpha = sp.rand(K, K).astype(float_type)
    beta = sp.rand(K, K).astype(float_type)
    gamma = sp.rand(K).astype(float_type)
    emit_probs = sp.rand(K, L).astype(float_type)
    vb_mf.normalize_trans(theta, alpha, beta, gamma)
    return theta, alpha, beta, gamma, emit_probs

# def trim_data(args):
#     """Trim regions without any observations from the start and end of the
#     obervation matrices
#     """
#     for f in args.observe_matrix:
#         print '# trimming ', f, 'start is ',
#         X = sp.load(f).astype(sp.int8)
#         S = X.cumsum(axis=0).cumsum(axis=2)  # any species has any observation
#         for start_t in xrange(X.shape[1]):
#             if S[-1, start_t, -1] > 0:
#                 break
#         for end_t in xrange(X.shape[1] - 1, -1, -1):
#             if S[-1, end_t, -1] > 0:
#                 break
#         tmpX = X[:, start_t:end_t, :]
#         print start_t
#         sp.save(os.path.splitext(f)[0] + '.trimmed', tmpX)

def split_data(args):
    """Split the given observation matrices into smaller chunks"""
    sizes = []
    total_size = 0
    covered_size = 0
    attrs = pickle.load(open(args.start_positions))
    valid_species = attrs['valid_species']
    valid_marks = attrs['valid_marks']
    windowsize = attrs['windowsize']
    old_starts = attrs['start_positions']
    start_positions = {}
    for f in args.observe_matrix:
        print '# splitting ', f
        chrom = old_starts[os.path.split(f)[1]][0]

        X = sp.load(f).astype(sp.int8)
        total_size += X.shape[1]
        #start_ts = xrange(0, X.shape[1], args.chunksize)
        #end_ts = xrange(args.chunksize, X.shape[1] + args.chunksize, args.chunksize)

        density = X.sum(axis=0).sum(axis=1)  # sumation over I, then L
        #from ipdb import set_trace; set_trace()
        gk = _gauss_kernel(args.gauss_window_size)
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
            fname = os.path.split(name)[1]
            start_positions[fname] = (chrom, start_t)
    print '# plotting size distribution'
    pyplot.figure()
    pyplot.figtext(.5,.01,'%s regions; %s bins total; %s bins covered; coverage = %.3f' % (len(sizes),total_size, covered_size, covered_size / float(total_size)), ha='center')
    pyplot.hist(sizes, bins=100)
    pyplot.title('chunk sizes for all chroms, min_reads %s, min_size %s, gauss_window_size %s' % (args.min_reads, args.min_size, args.gauss_window_size))
    pyplot.savefig('chunk_sizes.minreads%s.minsize%s.windowsize%s.png' % (args.min_reads, args.min_size, args.gauss_window_size))

    with open('start_positions_split.pkl', 'w') as outfile:
        attrs = dict(windowsize=windowsize, start_positions=start_positions, 
                     valid_species=valid_species, valid_marks=valid_marks)
        pickle.dump(attrs, outfile, -1)
    # --min_reads .5 --min_size 25 --window_size 200;



# def extract_local_features(args):
#     """extract some local features from the given data, saving an X array with extra dimensions"""
#     sizes = []
#     total_size = 0
#     covered_size = 0
#     start_positions = {}
#     for f in args.observe_matrix:
#         print '# features on ', f
#         X = sp.load(f).astype(sp.int8)
#         total_size += X.shape[1]
#         #start_ts = xrange(0, X.shape[1], args.chunksize)
#         #end_ts = xrange(args.chunksize, X.shape[1] + args.chunksize, args.chunksize)

#         density = X.sum(axis=0).sum(axis=1)  # summation over I, then L
#         #from ipdb import set_trace; set_trace()
#         gk = _gauss_kernel(args.window_size)
#         smoothed_density = scipy.signal.convolve(density, gk, mode='same')
#         regions_to_keep = smoothed_density >= args.min_reads

#         # find the regions where a transition is made from no reads to reads, and reads to no reads
#         start_ts = sp.where(sp.diff(regions_to_keep.astype(sp.int8)) > 0)[0]
#         end_ts = sp.where(sp.diff(regions_to_keep.astype(sp.int8)) < 0)[0]

#         cur_regions = [r for r in zip(start_ts, end_ts) if r[1] - r[0] >= args.min_size]
#         sizes.extend([end_t - start_t for start_t, end_t in cur_regions])

#         print 'saving %s regions' % len(sizes)
#         for chunknum, (start_t, end_t) in enumerate(cur_regions):
#             covered_size += end_t - start_t
#             tmpX = X[:, start_t:end_t, :]
#             name = os.path.splitext(f)[0] + '.chunk%s.npy' % chunknum
#             sp.save(name, tmpX)
#             start_positions[name] = start_t
#     print '# plotting size distribution'
#     pyplot.figure()
#     pyplot.figtext(.5,.01,'%s regions; %s bins total; %s bins covered; coverage = %.3f' % (len(sizes),total_size, covered_size, covered_size / float(total_size)), ha='center')
#     pyplot.hist(sizes, bins=100)
#     pyplot.title('chunk sizes for all chroms, min_reads %s, min_size %s, window_size %s' % (args.min_reads, args.min_size, args.window_size))
#     pyplot.savefig('chunk_sizes.minreads%s.minsize%s.windowsize%s.png' % (args.min_reads, args.min_size, args.window_size))

#     pickle.dump(start_positions, open('start_positions.pkl', 'w'))
#     # --min_reads .5 --min_size 25 --window_size 200;



# def convert_data_continuous_features_and_split(args):
#     """histogram both treatment and control data as specified by args
#     This saves the complete X matrix

#     This version doesn't binarize the data, smooths out the read signal (gaussian convolution)
#     and adds derivative information
#     """
#     if args.download_first:
#         download_data(args)
#     I = len(args.species)
#     L = len(args.marks)
#     final_data = None
#     total_size = 0
#     covered_size = 0
#     start_positions = {}
#     # make sure all the data is present...
#     for species in args.species:
#         for mark in args.marks:
#             d_files = [f for f in glob.glob(args.bam_template.format(
#                                                 species=species, mark=mark))]
#             if len(d_files) == 0:
#                 print("No histone data for species %s mark %s Expected: %s" %
#                               (species, mark, args.bam_template.format(
#                                                 species=species, mark=mark)))

#     for i, species in enumerate(args.species):
#         for l, mark in enumerate(args.marks):
#             d_obs = {}
#             d_files = [f for f in glob.glob(args.bam_template.format(
#                                                 species=species, mark=mark))]
#             if len(d_files) == 0:
#                 args.mark_avail[i, l] = 0
#             else:
#                 args.mark_avail[i, l] = 1
#                 for mark_file in d_files:
#                     read_counts = histogram_reads(mark_file, args.windowsize, args.chromosomes)

#                         # d_obs.append(histogram_reads(mark_file, args.windowsize,
#                         #                             args.chromosomes))
#                 for 
#                 d_obs = reduce(operator.add, d_obs)  # add all replicants together
#                 #print 'before per million:', d_obs.sum()
#                 #d_obs /= (d_obs.sum() / 1e7)  # convert to reads mapping per ten million
#                 # convert to a binary array with global poisson
#                 #genome_rate = d_obs / (d_obs.sum() / 1e6)

#                 if final_data is None:
#                     final_data = sp.zeros((I, len(d_obs), L), dtype=sp.float32)
#                 final_data[i, :, l] = d_obs
#                 total_size = final_data.shape[1]


#     regions_to_keep = (final_data[:, :, tuple(range(L))].sum(axis=0).sum(axis=1) >= args.min_reads).astype(sp.int8)
#     # find the regions where a transition is made from no reads to reads, and reads to no reads
#     start_ts = sp.where(sp.diff(regions_to_keep) > 0)[0]
#     end_ts = sp.where(sp.diff(regions_to_keep) < 0)[0]

#     cur_regions = [r for r in zip(start_ts, end_ts) if r[1] - r[0] >= args.min_size]
#     sizes = [end_t - start_t for start_t, end_t in cur_regions]

#     print 'saving %s regions' % len(sizes)
#     tarout = tarfile.open(args.outfile + '.tar.gz', 'w:gz')
#     for chunknum, (start_t, end_t) in enumerate(cur_regions):
#         covered_size += end_t - start_t
#         tmpX = final_data[:, start_t:end_t, :]

#         print 'adding chunk', chunknum, 'of', len(cur_regions)
#         s = StringIO()
#         sp.save(s, tmpX)
#         name = args.outfile + '.chunk%s.npy' % chunknum
#         info = tarfile.TarInfo(name)
#         info.size = s.tell(); info.mtime = time.time()
#         s.seek(0)
#         tarout.addfile(info, s)
#         start_positions[name] = start_t

#     print '# plotting size distribution'
#     pyplot.figure()
#     pyplot.figtext(.5,.01,'%s regions; %s bins total; %s bins covered; coverage = %.3f' % (len(sizes),total_size, covered_size, covered_size / float(total_size)), ha='center')
#     pyplot.hist(sizes, bins=100)
#     pyplot.title('chunk sizes for all chroms, min_reads %s, min_size %s, windowsize %s' % (args.min_reads, args.min_size, args.windowsize))
#     pyplot.savefig('chunk_sizes.minreads%s.minsize%s.windowsize%s.png' % (args.min_reads, args.min_size, args.windowsize))

#     s = StringIO()
#     pyplot.savefig(s)
#     info = tarfile.TarInfo('chunk_sizes.minreads%s.minsize%s.windowsize%s.png' % (args.min_reads, args.min_size, args.windowsize))
#     info.size = s.tell(); info.mtime = time.time()
#     s.seek(0)
#     tarout.addfile(info, s)

#     s = StringIO()
#     pickle.dump(start_positions, s)
#     info = tarfile.TarInfo('start_positions.pkl')
#     info.size = s.tell(); info.mtime = time.time()
#     s.seek(0)
#     tarout.addfile(info, s)
#     pickle.dump(start_positions, open('start_positions.pkl', 'w'))
#     # --min_reads .5 --min_size 25 --window_size 200;


#     s = StringIO()
#     sp.save(s, args.mark_avail)
#     info = tarfile.TarInfo('available_marks.npy')
#     info.size = s.tell(); info.mtime = time.time()
#     s.seek(0)
#     tarout.addfile(info, s)
#     pickle.dump(start_positions, open('start_positions.pkl', 'w'))
#     # --min_reads .5 --min_size 25 --window_size 200;

#     tarout.close()


#     print "output file:", args.outfile
#     print 'available marks:', args.mark_avail
#     #with open(args.outfile, 'wb') as outfile:
#     #    sp.save(outfile, final_data)
#     with open(args.outfile + '.available_marks', 'wb') as outfile:
#         sp.save(outfile, args.mark_avail)




# def convert_data_continuous_features_and_split_old(args):
#     """histogram both treatment and control data as specified by args
#     This saves the complete X matrix

#     This version doesn't binarize the data, smooths out the read signal (gaussian convolution)
#     and adds derivative information
#     """
#     if args.download_first:
#         download_data(args)
#     I = len(args.species)
#     L = len(args.marks)
#     final_data = None
#     total_size = 0
#     covered_size = 0
#     start_positions = {}
#     # make sure all the data is present...
#     for species in args.species:
#         for mark in args.marks:
#             d_files = [f for f in glob.glob(args.bam_template.format(
#                                                 species=species, mark=mark))]
#             if len(d_files) == 0:
#                 print("No histone data for species %s mark %s Expected: %s" %
#                               (species, mark, args.bam_template.format(
#                                                 species=species, mark=mark)))

#     for i, species in enumerate(args.species):
#         for l, mark in enumerate(args.marks):
#             l = l * 3
#             d_obs = []
#             d_files = [f for f in glob.glob(args.bam_template.format(
#                                                 species=species, mark=mark))]
#             if len(d_files) == 0:
#                 args.mark_avail[i, l] = 0
#                 args.mark_avail[i, l+1] = 0
#                 args.mark_avail[i, l+2] = 0
#             else:
#                 args.mark_avail[i, l] = 1
#                 args.mark_avail[i, l+1] = 1
#                 args.mark_avail[i, l+2] = 1
#                 for mark_file in d_files:
#                     try:
#                         d_obs.append(histogram_reads(mark_file, args.windowsize,
#                                                     args.chromosomes))
#                     except ValueError as e:
#                         print e.message
#                     print d_obs[-1].sum()
#                     print d_obs[-1].shape
#                 d_obs = reduce(operator.add, d_obs)  # add all replicants together
#                 #print 'before per million:', d_obs.sum()
#                 #d_obs /= (d_obs.sum() / 1e7)  # convert to reads mapping per ten million
#                 # convert to a binary array with global poisson
#                 genome_rate = d_obs / (d_obs.sum() / 1e6)

#                 if final_data is None:
#                     final_data = sp.zeros((I, len(d_obs), L * 3), dtype=sp.float32)
#                 asinh_obs = sp.log(genome_rate + sp.sqrt(genome_rate * genome_rate + 1))
#                 gk = _gauss_kernel(3)
#                 smoothed_obs = scipy.signal.convolve(asinh_obs, gk, mode='same')
#                 smooth_deriv = sp.gradient(smoothed_obs)
#                 smooth_deriv2 = sp.gradient(smooth_deriv)
#                 final_data[i, :, l] = smoothed_obs
#                 final_data[i, :, l + 1] = smooth_deriv
#                 final_data[i, :, l + 2] = smooth_deriv2
#                 total_size = final_data.shape[1]


#     regions_to_keep = (final_data[:, :, tuple(range(0, L * 3, 3))].sum(axis=0).sum(axis=1) >= args.min_reads).astype(sp.int8)
#     # find the regions where a transition is made from no reads to reads, and reads to no reads
#     start_ts = sp.where(sp.diff(regions_to_keep) > 0)[0]
#     end_ts = sp.where(sp.diff(regions_to_keep) < 0)[0]

#     cur_regions = [r for r in zip(start_ts, end_ts) if r[1] - r[0] >= args.min_size]
#     sizes = [end_t - start_t for start_t, end_t in cur_regions]

#     print 'saving %s regions' % len(sizes)
#     tarout = tarfile.open(args.outfile + '.tar.gz', 'w:gz')
#     for chunknum, (start_t, end_t) in enumerate(cur_regions):
#         covered_size += end_t - start_t
#         tmpX = final_data[:, start_t:end_t, :]

#         print 'adding chunk', chunknum, 'of', len(cur_regions)
#         s = StringIO()
#         sp.save(s, tmpX)
#         name = args.outfile + '.chunk%s.npy' % chunknum
#         info = tarfile.TarInfo(name)
#         info.size = s.tell(); info.mtime = time.time()
#         s.seek(0)
#         tarout.addfile(info, s)
#         start_positions[name] = start_t

#     print '# plotting size distribution'
#     pyplot.figure()
#     pyplot.figtext(.5,.01,'%s regions; %s bins total; %s bins covered; coverage = %.3f' % (len(sizes),total_size, covered_size, covered_size / float(total_size)), ha='center')
#     pyplot.hist(sizes, bins=100)
#     pyplot.title('chunk sizes for all chroms, min_reads %s, min_size %s, windowsize %s' % (args.min_reads, args.min_size, args.windowsize))
#     pyplot.savefig('chunk_sizes.minreads%s.minsize%s.windowsize%s.png' % (args.min_reads, args.min_size, args.windowsize))

#     s = StringIO()
#     pyplot.savefig(s)
#     info = tarfile.TarInfo('chunk_sizes.minreads%s.minsize%s.windowsize%s.png' % (args.min_reads, args.min_size, args.windowsize))
#     info.size = s.tell(); info.mtime = time.time()
#     s.seek(0)
#     tarout.addfile(info, s)

#     s = StringIO()
#     pickle.dump(start_positions, s)
#     info = tarfile.TarInfo('start_positions.pkl')
#     info.size = s.tell(); info.mtime = time.time()
#     s.seek(0)
#     tarout.addfile(info, s)
#     pickle.dump(start_positions, open('start_positions.pkl', 'w'))
#     # --min_reads .5 --min_size 25 --window_size 200;


#     s = StringIO()
#     sp.save(s, args.mark_avail)
#     info = tarfile.TarInfo('available_marks.npy')
#     info.size = s.tell(); info.mtime = time.time()
#     s.seek(0)
#     tarout.addfile(info, s)
#     pickle.dump(start_positions, open('start_positions.pkl', 'w'))
#     # --min_reads .5 --min_size 25 --window_size 200;

#     tarout.close()


#     print "output file:", args.outfile
#     print 'available marks:', args.mark_avail
#     #with open(args.outfile, 'wb') as outfile:
#     #    sp.save(outfile, final_data)
#     with open(args.outfile + '.available_marks', 'wb') as outfile:
#         sp.save(outfile, args.mark_avail)


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
    start_positions = {}
    # make sure all the data is present...
    for species in args.species:
        for mark in args.marks:
            d_files = [f for f in glob.glob(args.bam_template.format(
                                                species=species, mark=mark, repnum='*'))]
            if len(d_files) == 0:
                raise RuntimeError("No histone data for species %s mark %s Expected: %s" %
                              (species, mark, args.bam_template.format(
                                                species=species, mark=mark, repnum='*')))

    for i, species in enumerate(args.species):
        for l, mark in enumerate(args.marks):

            d_obs = {}
            d_files = [f for f in glob.glob(args.bam_template.format(
                                                species=species, mark=mark, repnum='*'))]
            if len(d_files) == 0:
                pass
            else:
                for mark_file in d_files:
                    read_counts = histogram_reads(mark_file, args.windowsize, args.chromosomes) 
                    # d_obs.append(histogram_reads(mark_file, args.windowsize,
                    #                              args.chromosomes))
                    for chrom in read_counts:
                        d_obs.setdefault(chrom, []).append(read_counts[chrom])
                for chrom in d_obs:
                    d_obs[chrom] = reduce(operator.add, d_obs[chrom])  # add all replicants together
                #print 'before per million:', d_obs.sum()
                #d_obs /= (d_obs.sum() / 1e7)  # convert to reads mapping per ten million
                # convert to a binary array with global poisson
                num_reads = sum(x.sum() for x in d_obs.values())
                num_bins = float(sum(len(x) for x in d_obs.values()))
                genome_rate = num_reads / num_bins
                print 'after per million', num_reads, num_bins, genome_rate
                if final_data is None:
                    final_data = {}
                for chrom in d_obs:
                    d_obs[chrom] = call_significant_sites(d_obs[chrom], genome_rate, args.max_pvalue)
                    if chrom not in final_data:
                        final_data[chrom] = sp.zeros((I, len(d_obs[chrom]), L), dtype=sp.int8)
                    final_data[chrom][i, :, l] = d_obs[chrom]
    for chrom in final_data:
        start_positions[os.path.split(args.outfile.format(chrom=chrom))[1]] = (chrom, 0)
        print "output file:", args.outfile.format(chrom=chrom)
        with open(args.outfile.format(chrom=chrom), 'wb') as outfile:
            sp.save(outfile, final_data[chrom])
    with open('start_positions.pkl', 'w') as outfile:
        pickle.dump(dict(windowsize=args.windowsize, valid_species=valid_species,
                        valid_marks=valid_marks, start_positions=start_positions),
                    outfile, -1)

def download_data(args):
    """Download any missing histone modification data from UCSC and check md5s.
    """
    md5s = urllib.urlopen(args.base_url % 'md5sum.txt').read().strip().split('\n')
    md5s = dict(reversed(l.strip().split()) for l in md5s)
    for species in args.species:
        for mark in args.marks:
            for rep in range(10):
                fname = args.bam_template.format(species=species,
                                            mark=mark, repnum=rep)
                if fname not in md5s:
                    continue
                if os.path.exists(fname):
                    #m = hashlib.md5(open(fname, 'rb').read()).hexdigest()
                    #if m != md5s[fname]:  # destroy if md5 doesn't match
                    #    print 'removing incomplete file: %s' % fname
                    #    print m, md5s[fname]
                    #    os.unlink(fname)
                    #else:
                        print 'skipping already downloaded %s' % fname
                        continue
                with open(fname, 'wb') as outfile:
                    try:
                        print 'downloading %s' % fname
                        page = urllib.urlopen(args.base_url % fname)
                        while True:
                            data = page.read(81920)
                            if not data:
                                break
                            outfile.write(data)
                    except RuntimeError as e:
                        print 'Skipping...', e.message



def histogram_reads(bam_file, windowsize, chromosomes='all', exclude_chroms=['chrM', 'chrY', 'chrX'],
                    skip_qc_fail=True):
    """Histogram the counts along bam_file, resulting in a vector.

    This will concatenate all chromosomes, together, so to get the
    counts for a particular chromosome, pass it as a list, a la
    >>> histogram_reads(my_bam_file, chromosomes=['chr1'])
    """
    print 'histogramming', bam_file
    reads_bam = pysam.Samfile(bam_file, 'rb')
    # get the chromosome name and lengths for our subset
    if chromosomes == 'all':
        chromosomes = filter(lambda c: c not in exclude_chroms,
                             reads_bam.references)
        chrom_set = set(chromosomes)
        chrom_lengths = {c : reads_bam.lengths[reads_bam.references.index(c)]
                                       for c in chromosomes}
    else:
        chromosomes = filter(lambda c: c not in exclude_chroms,
                             chromosomes)
        chrom_lengths = {c : reads_bam.lengths[reads_bam.references.index(c)]
                                       for c in chromosomes if c not in
                                                    exclude_chroms}
        chrom_set = set(chromosomes)
    # # offset of each chromosome into concatenated chrom bins
    # chrom_ends = list(((sp.array(chrom_lengths) // windowsize) + 1).cumsum())
    # chrom_starts = dict(zip(chromosomes, [0] + chrom_ends[:-1]))

    read_counts = {}
    # create the histogram: 1 x sum(lengths) array
    # read_counts = sp.zeros(chrom_ends[-1], dtype=float_type)

    # count the reads in the input
    for read in reads_bam:
        if skip_qc_fail and (read.is_qcfail or read.is_unmapped or read.is_secondary or
               read.is_duplicate or read.mapq == 0):
           continue  # filter out non-mapping reads
        chrom = reads_bam.references[read.tid]
        if chrom in chrom_set:  # chrom requested?
            # offset = chrom_starts[chrom]
            offset = 0
            if read.is_paired:
                if read.is_proper_pair:
                    # add at the middle of the mates
                    bin = offset + ((read.pos + read.mpos +
                                     read.rlen) / 2) // windowsize
                    # read_counts[min(chrom_ends[-1] - 1, bin)] += 1.
                    if chrom not in read_counts:
                        read_counts[chrom] = sp.zeros(chrom_lengths[chrom] // windowsize, dtype=float_type)
                    read_counts[chrom][min(chrom_lengths[chrom] // windowsize - 1, bin)] += 1.
            else:
                # add at the middle of the fragment
                bin = offset + (read.pos + 100) // windowsize
                if chrom not in read_counts:
                    read_counts[chrom] = sp.zeros(chrom_lengths[chrom] // windowsize, dtype=float_type)
                read_counts[chrom][min(chrom_lengths[chrom] // windowsize - 1, bin)] += 1.
    return read_counts


def call_significant_sites(fg_counts, bg_counts, max_pvalue):
    """binarize fg_counts (significant=1) using bg_counts as a local poisson
    rate. the poisson survival must be < sig_level.
    """
    print 'most reads in a bin:', fg_counts.max(), 'poisson expected rate:', bg_counts
    print 'read count vs binary present:' , {i: poisson.sf(i, bg_counts) < max_pvalue for i in range(20)}
    return poisson.sf(fg_counts, bg_counts) < max_pvalue


if __name__ == '__main__':
    main()
