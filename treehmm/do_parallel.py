
import re
import glob
import multiprocessing
import scipy as sp
import copy
import os
import tempfile
import time

# try:
#    from ipdb import set_trace as breakpoint
# except ImportError:
#    from pdb import set_trace as breakpoint

import sge

def do_parallel_inference(args):
    """Perform inference in parallel on several observations matrices with
    joint parameters
    """
    from treehmm import random_params, do_inference, plot_params, plot_energy, load_params
    from treehmm.vb_mf import normalize_trans
    from treehmm.static import float_type

    _x = sp.load(args.observe_matrix[0])
    args.continuous_observations = _x.dtype != sp.int8
    args.I, _, args.L = _x.shape
    I = args.I
    K = args.K
    L = args.L
    args.T = 'all'
    args.free_energy = []
    args.observe = 'all.npy'
    args.last_free_energy = 0
    args.emit_sum = sp.zeros((K, L), dtype=float_type)

    args.out_dir = args.out_dir.format(timestamp=time.strftime('%x_%X').replace('/', '-'), **args.__dict__)
    try:
        print 'making', args.out_dir
        os.makedirs(args.out_dir)
    except OSError:
        pass

    if args.warm_start:
        # args.last_free_energy, args.theta, args.alpha, args.beta, args.gamma, args.emit_probs, args.emit_sum = load_params(args)
        # args.warm_start = False
        print '# loading previous params for warm start from %s' % args.warm_start
        tmpargs = copy.deepcopy(args)
        tmpargs.out_dir = args.warm_start
        tmpargs.observe = 'all.npy'
        args.free_energy, args.theta, args.alpha, args.beta, args.gamma, args.emit_probs, args.emit_sum = load_params(tmpargs)

        try:
            args.free_energy = list(args.free_energy)
        except TypeError:  # no previous free energy
            args.free_energy = []
        print 'done'
        args.warm_start = False
    else:
        (args.theta, args.alpha, args.beta, args.gamma, args.emit_probs) = \
                                                    random_params(args.I, args.K, args.L, args.separate_theta)
        for p in ['free_energy', 'theta', 'alpha', 'beta', 'gamma', 'emit_probs', 'last_free_energy', 'emit_sum']:
            sp.save(os.path.join(args.out_dir, args.out_params.format(param=p, **args.__dict__)),
                args.__dict__[p])

    
    args.iteration = 0
    plot_params(args)
    print '# setting up job arguments'
    # set up new versions of args for other jobs
    job_args = [copy.copy(args) for i in range(len(args.observe_matrix))]
    for j, a in enumerate(job_args):
        a.observe_matrix = args.observe_matrix[j]
        a.observe = os.path.split(args.observe_matrix[j])[1]
        a.subtask = True
        a.func = None
        a.iteration = 0
        a.max_iterations = 1
        a.quiet_mode = True
        if j % 1000 == 0:
            print j

    if args.run_local:
        pool = multiprocessing.Pool()
    else:
        pool = sge.SGEPool()
        #job_handle = pool.imap_unordered(do_inference, job_args)

    converged = False
    for args.iteration in range(args.max_iterations):
        # import ipdb; ipdb.set_trace()
        # fresh parameters-- to be aggregated after jobs are run
        print 'iteration', args.iteration
        total_free = 0
        if args.separate_theta:
            args.theta = sp.zeros((I - 1, K, K, K), dtype=float_type)
        else:
            args.theta = sp.zeros((K, K, K), dtype=float_type)

        args.alpha = sp.zeros((K, K), dtype=float_type)
        args.beta = sp.zeros((K, K), dtype=float_type)
        args.gamma = sp.zeros((K), dtype=float_type)
        args.emit_probs = sp.zeros((K, L), dtype=float_type)
        if True:  # args.approx == 'clique':
            args.emit_sum = sp.zeros_like(args.emit_probs, dtype=float_type)
        else:
            args.emit_sum = sp.zeros((K, L), dtype=float_type)

        if args.run_local:
            iterator = pool.imap_unordered(do_inference, job_args, chunksize=args.chunksize)
            # wait for jobs to finish
            for result in iterator:
                pass
        else:
            jobs_handle = pool.map_async(do_inference, job_args, chunksize=args.chunksize)
            # wait for all jobs to finish
            for j in jobs_handle:
                j.wait()

        # sum free energies and parameters from jobs
        for a in job_args:
            # print '# loading from %s' % a.observe
            free_energy, theta, alpha, beta, gamma, emit_probs, emit_sum = load_params(a)
            # print 'free energy for this part:', free_energy
            if len(free_energy) > 0:
                last_free_energy = free_energy[-1]
            else:
                last_free_energy = 0
            total_free += last_free_energy
            args.theta += theta
            args.alpha += alpha
            args.beta += beta
            args.gamma += gamma
            args.emit_probs += emit_probs
            args.emit_sum += emit_sum

        # renormalize and plot
        print 'normalize aggregation... total free energy is:', total_free
        args.free_energy.append(total_free)
        if len(args.free_energy) > 1 and args.free_energy[-1] != 0 and args.free_energy[-2] != 0 \
            and abs((args.free_energy[-2] - args.free_energy[-1]) / args.free_energy[-2]) < args.epsilon:
            print 'converged. free energy diff:', args.free_energy, abs(args.free_energy[-2] - args.free_energy[-1]) / args.free_energy[-2]
            converged = True
        normalize_trans(args.theta, args.alpha, args.beta, args.gamma)
        # if True: #args.approx == 'clique':
        #    #print 'clique emit renorm'
        #    args.emit_probs[:] = args.emit_probs / args.emit_sum
        # else:
        #    args.emit_probs[:] = sp.dot(sp.diag(1./args.emit_sum), args.emit_probs)
        args.emit_probs[:] = sp.dot(sp.diag(1. / args.emit_sum), args.emit_probs)
        for a in job_args:
            a.theta, a.alpha, a.beta, a.gamma, a.emit_probs, a.emit_sum = args.theta, args.alpha, args.beta, args.gamma, args.emit_probs, args.emit_sum

        for p in ['free_energy', 'theta', 'alpha', 'beta', 'gamma', 'emit_probs', 'lmd', 'tau']:
            try:
                sp.save(os.path.join(args.out_dir, args.out_params.format(param=p, **args.__dict__)),
                        args.__dict__[p])
            except KeyError:
                pass
        plot_params(args)
        plot_energy(args)

        if args.save_Q >= 3:
            print '# reconstructing chromosomes from *chunk*',
            in_order = {}
            # Q_chr16_all.trimmed.chunk*.npy => Q_chr16_all.trimmed.npy
            all_chunks = glob.glob(os.path.join(args.out_dir, '*_Q_*chunk*.npy'))
            for chunk in all_chunks:
                print chunk
                chunk_num = int(re.search(r'chunk(\d+)', chunk).groups()[0])
                chrom_out = re.sub('chunk(\d+)\.', '', chunk)
                if chrom_out not in in_order:
                    in_order[chrom_out] = {}
                in_order[chrom_out][chunk_num] = sp.load(chunk)
            for chrom_out in in_order:
                print 'reconstructing chromosomes from', in_order[chrom_out]
                if len(in_order[chrom_out]) > 1:
                    final_array = sp.concatenate((in_order[chrom_out][0], in_order[chrom_out][1]), axis=1)
                    for i in range(2, max(in_order[chrom_out])):
                        final_array = sp.concatenate((final_array, in_order[chrom_out][i]), axis=1)
                else:
                    final_array = in_order[chrom_out][0]
                sp.save(chrom_out, final_array)

        if converged:
            break
