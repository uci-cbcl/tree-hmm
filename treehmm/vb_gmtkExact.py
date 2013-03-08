
import operator
import glob
import sys
import re
import os
from StringIO import StringIO
import textwrap
import subprocess
import multiprocessing

import numpy

import gmtkParam
import sge
import vb_mf


class BlankNamespace:
    pass


def npy_observations_to_txt(filename):
    obs = numpy.load(filename)
    I, T, L = obs.shape
    obs_reshape = numpy.zeros((T, I * L))
    for t in range(T):
        for i in range(I):
            for l in range(L):
                obs_reshape[t, i * L + l] = obs[i, t, l]
    out_txt = os.path.splitext(filename)[0] + '.txt'
    numpy.savetxt(out_txt, obs_reshape, fmt='%d')
    #numpy.savetxt(out_txt, obs.reshape(obs.shape[0]*obs.shape[2], obs.shape[1]).T, fmt='%d')
    return obs, out_txt



def npy_params_to_workspace(args):
    w = gmtkParam.Workspace()
    for param in ['alpha', 'beta', 'gamma']:
        array = getattr(args, param)
        cpt = gmtkParam.DCPT(w, param, array.shape[:-1], array.shape[-1])
        cpt.setTableFromNumpyArray(array)
    K = args.alpha.shape[0]
    for theta_name in ['theta'] if not args.separate_theta else [('theta_%s' % i) for i in range(2,args.I+1)]:
        print theta_name, getattr(args, theta_name).shape
        theta_reshape = numpy.zeros((K * K, K))
        for vp in range(K):
            for hp in range(K):
                for k in range(K):
                    theta_reshape[vp * K + hp, k] = getattr(args, theta_name)[vp, hp, k]
        cpt = gmtkParam.DCPT(w, theta_name, getattr(args, theta_name).shape[:-1], getattr(args, theta_name).shape[-1])
        #cpt.setTableFromNumpyArray(theta_reshape)
        for vp, hp in cpt.getPossibleParents():
            cpt._table[(vp, hp)] = list(theta_reshape[vp * K + hp, :])
    for l in range(args.emit_probs.shape[1]):
        e = args.emit_probs[:, l]
        cpt = gmtkParam.DCPT(w, 'emit_probs_mark%s' % (l + 1), args.emit_probs.shape[:-1], 2)
        cpt.setTableFromNumpyArray(numpy.c_[1. - e, e])
    return w


def workspace_to_npy_args(filename):
    args = BlankNamespace()
    args.observe = os.path.split(filename)[1]
    args.out_dir = 'gmtk_images'
    args.approx = 'gtmk'
    try:
        iteration = re.search('(\d+|)\.gmp$', os.path.split(filename)[1]).groups()[0]
    except:
        iteration = ''
    args.iteration = iteration if iteration else 'final'
    args.out_params = '{approx}_{param}_{observe}'

    w = gmtkParam.Workspace()
    w.readTrainableParamsFile(filename)

    for param, matrix in w.objects[gmtkParam.DCPT].items():
        setattr(args, param, matrix.array)
    args.emit_probs = numpy.c_.__getitem__([getattr(args, 'emit_probs_mark%s' % (l + 1))[:, 1] for l in range(20) if ('emit_probs_mark%s' % (l + 1)) in args.__dict__]).T
    K = args.alpha.shape[0]
    for theta_name in ['theta'] + ['theta_%s' % i for i in range(20)]:
        theta = numpy.zeros((K, K, K))
        try:
            workspace_theta = w.objects[gmtkParam.DCPT][theta_name]
        except KeyError:
            continue
        for vp in range(K):
            for hp in range(K):
                for k in range(K):
                    #theta[vp, hp, k] = workspace_theta._table[vp * K + hp, k]
                    theta[vp, hp, k] = workspace_theta._table[(vp, hp)][k]
        setattr(args, theta_name, theta)
    try:
        #args.separate_theta:
        args.theta = numpy.array([getattr(args, 'theta_' % i) for i in range(2, args.I)])
        #for i in range(2, args.I):
        #    delattr(args, 'theta_' % i)
    except:
        pass
    return args


def plot_params(wkspacefile):
    args = workspace_to_npy_args(wkspacefile)
    try:
        os.mkdir(args.out_dir)
    except:
        pass
    plot_params(args)


def get_npy_params_as_args(observe, path='.', approx='clique', template='{path}/{approx}_{param}_{observe}'):
    args = BlankNamespace()
    args.observe = observe
    args.approx = approx
    for param in['alpha', 'beta', 'gamma', 'theta', 'emit_probs']:
        setattr(args, param, numpy.load(template.format(path=path, param=param, **args.__dict__)))
    return args


def write_workspace_simple_master(workspace, filename='input.master'):
    cpts = workspace.objects[gmtkParam.DCPT]
    io = gmtkParam.WorkspaceIO.withFile(filename, 'w')
    try:
        io._file.write('DENSE_CPT_IN_FILE inline\n')
        io.writeInt(len(cpts))
        io.writeNewLine()
        io.writeNewLine()
        for i, (name, obj) in enumerate(cpts.items()):
            io.writelnInt(i)
            obj.writeToFile(io)
            io.writeNewLine()
    finally:
        io.close()



def make_lineage_model(filename, I, K, L, vert_parent, mark_avail, separate_theta):
    print locals()

    #if vert_parents is None:
    #    # use i=0 as root
    #    vert_parents = {i: i - 1 for i in range(I)}
    #
    #if mark_avail is None:
    #    # assume all are available
    #    mark_avail = numpy.ones((I, L))

    # frame 0
    state_tmpl = """variable : %(name)s {
        type : discrete hidden cardinality %(K)s;
        switchingparents : nil;
        conditionalparents : %(parents)s using DenseCPT("%(param)s");
    }"""
    states0 = [dict(name='state_sp1', parents='nil', param='gamma', K=K)]
    states0.extend([dict(name='state_sp%s' % i,
                        parents='state_sp%s(0)' % (vert_parent[i-1] + 1),
                        param='beta',
                        K=K)
                   for i in range(2, I+1)])

    obs_tmpl = """variable : %(name)s {
        type : discrete observed %(index)s:%(index)s cardinality 2;
        switchingparents : nil;
        conditionalparents : %(parent)s using DenseCPT("%(param)s");
    }"""
    #obs_hidden_tmpl = """variable : %(name)s {
    #    type : discrete hidden cardinality 2;
    #    switchingparents : nil;
    #    conditionalparents : %(parent)s using DenseCPT("%(param)s");
    #}"""
    obs_hidden_tmpl = None

    obs0 = [dict(name='obs_sp%s_mark%s' % (i, l),
                index=(i - 1) * L + (l - 1),
                parent='state_sp%s(0)' % i,
                param='emit_probs_mark%s' % l,
                template=obs_tmpl if mark_avail[i - 1, l - 1] else obs_hidden_tmpl)
           for i in range(1, I + 1) for l in range(1, L + 1)]
    statestr = '\n'.join(state_tmpl % s for s in states0)
    obsstr = '\n'.join(s['template'] % s if s['template'] is not None else '' for s in obs0)
    frame0 = "frame : 0 {\n%s\n%s\n}" % (statestr, obsstr)

    # frame 1
    states1 = [dict(name='state_sp1', parents='state_sp1(-1)', param='alpha', K=K)]
    states1.extend([dict(name='state_sp%s' % i,
                        parents='state_sp%s(0), state_sp%s(-1)' % (vert_parent[i-1] + 1, i),
                        param='theta' if not separate_theta else ('theta_%s' % i),
                        K=K)
                   for i in range(2, I+1)])
    obs1 = [dict(name='obs_sp%s_mark%s' % (i, l),
                index=(i - 1) * L + (l - 1),
                parent='state_sp%s(0)' % i,
                param='emit_probs_mark%s' % l,
                template=obs_tmpl if mark_avail[i - 1, l - 1] else obs_hidden_tmpl)
           for i in range(1, I + 1) for l in range(1, L + 1)]
    statestr = '\n'.join(state_tmpl % s for s in states1)
    obsstr = '\n'.join(s['template'] % s if s['template'] is not None else '' for s in obs1)
    frame1 = "frame : 1 {\n%s\n%s\n}" % (statestr, obsstr)
    outstr = 'GRAPHICAL_MODEL lineagehmm\n%s\n%s\nchunk 1:1;' % (frame0, frame1)

    with open(filename, 'w') as outfile:
        outfile.write(outstr)


def write_params_for_chunk(observe, path='.', num_states=5, approx='clique', template='{path}/{approx}_{param}_{observe}'):
    strfile = 'lineagehmm.str'
    obs, obs_txt = npy_observations_to_txt(os.path.join(path, observe))
    args = get_npy_params_as_args(observe, path, approx)
    w = npy_params_to_workspace(args)

    gmtk_master = '%s.master' % observe
    gmtk_obs = '%s.observations' % observe
    gmtk_obs = os.path.split(gmtk_obs)[1]
    write_workspace_simple_master(w, gmtk_master)
    with open(gmtk_obs, 'w') as outfile:
        outfile.write(obs_txt)
        # obs_per_slice is I * L
    make_lineage_model(strfile, obs.shape[0], num_states, obs.shape[2], vert_parent=vert_parent, mark_avail=mark_avail)

    try:
        os.remove(strfile + '.trifile')
    except OSError:
        pass
    cmd = 'gmtkTriangulate -strFile %s -rePart T -findBest T  -triangulation completed ' % strfile
    print cmd
    subprocess.check_call(cmd, shell=True)

    cmd = ('gmtkJT -strFile {strfile} -fmt1 ascii -of1 {gmtk_obs} -nf1 0 '
           '-ni1 {obs_per_slice} -inputMasterFile {gmtk_master}'
           .format(obs_per_slice=args.emit_probs.shape[1] * obs.shape[0],
                   **locals()))
    print cmd
    subprocess.check_call(cmd, shell=True)


def do_em_from_args(args):
    """Do one EM iteration, modifying args with the new parameters"""
    index, iteration, matrix_txt, I, L, gmtk_master, gmtk_obs = args
    gmtk_obs = os.path.split(gmtk_obs)[1]
    strfile = 'lineagehmm.str'
    with open(gmtk_obs, 'w') as outfile:
        outfile.write(matrix_txt)

    cmd = ('gmtkEMtrainNew -strFile {strfile} -fmt1 ascii -of1 {gmtk_obs} '
           '-nf1 0 -ni1 {obs_per_slice} -inputMasterFile {gmtk_master} -maxE 1 '
           '-llStoreFile ll.iter{iter}.part{index} '
           '-outputTrainableParameters params.acc.iter{iter}.part{index} '
           '-accFileIsBinary true -storeAccFile acc.iter{iter}.part{index}.data'
           .format(iter=iteration, obs_per_slice=I * L, **locals()))
    print cmd
    s = subprocess.check_call(cmd, shell=True)


def accumulate_em_runs(args, gmtk_obs, gmtk_master):
    strfile = 'lineagehmm.str'
    # reset params
    total_free = 0
    K = args.K
    gmtk_obs = os.path.split(gmtk_obs)[1]
    # accumulate parts into a single EM-trained file
    cmd = ('gmtkEMtrainNew -strFile {strfile} -fmt1 ascii -of1 {gmtk_obs} '
           '-nf1 0 -ni1 {obs_per_slice} -inputMasterFile {gmtk_master} -maxE 1 '
           '-llStoreFile ll.iter{iter}.all '
           '-outputTrainableParameters params.acc.iter{iter}.all '
           '-accFileIsBinary true -loadAccRange 0:{num_pieces} '
           '-loadAccFile acc.iter{iter}.part@D.data -trrng nil'
           .format(iter=args.iteration, obs_per_slice=args.I * args.L,
                   num_pieces=len(args.observe_txt) - 1, **locals()))
    print cmd
    s = subprocess.check_call(cmd, shell=True)

    # save the trainable params to args
    new_args = workspace_to_npy_args('params.acc.iter{iter}.all'.format(iter=args.iteration))
    for a in 'alpha beta gamma theta'.split() + ['theta_%s' % i for i in range(20)]:
        #print 'accumulating', a
        try:
            setattr(args, a, getattr(new_args, a))
        except AttributeError:
            #print 'skipping', a
            continue
    if args.separate_theta:
        args.theta = numpy.array([getattr(args, 'theta_%s' % i) for i in range(20) if hasattr(args, 'theta_%s' % i)])
        #for i in range(20):
        #    if hasattr(args, 'theta_%s' % i):
        #        delattr(args, 'theta_%s' % i)
    args.emit_probs = numpy.array([getattr(new_args, 'emit_probs_mark%s' % (l+1))[:, 1] for l in range(args.L)]).T
    total_free = float(open('ll.iter{}.all'.format(args.iteration)).read().strip())
    args.free_energy.append(total_free)


#def accumulate_em_runs(args):
#    from vb_mf import normalize_trans
#    strfile = 'lineagehmm.str'
#    # reset params
#    total_free = 0
#    K = args.K
#    args.alpha = numpy.zeros((args.K, args.K), dtype=numpy.float64)
#    args.beta = numpy.zeros((args.K, args.K), dtype=numpy.float64)
#    args.gamma = numpy.zeros((args.K), dtype=numpy.float64)
#    args.theta = numpy.zeros((args.K, args.K, args.K), dtype=numpy.float64)
#    emit_probs_sums = [numpy.zeros((args.K, 2)) for l in range(args.L)]
#    # accumulate parts
#    for index, obs_file in enumerate(args.observe_matrix):
#        weight = numpy.load(obs_file).shape[1]  # slice count for this param
#        train_in = 'params.acc.iter{}.part{}'.format(args.iteration, index)
#        new_args = workspace_to_npy_args(train_in)
#        args.alpha += new_args.alpha * (weight - 1)
#        args.beta += new_args.beta
#        args.gamma += new_args.gamma
#        args.theta += new_args.theta * (weight - 1)
#        for l in range(args.L):
#            emit_probs_sums[l] += getattr(new_args, 'emit_probs_mark%s' % (l + 1)) * weight
#        total_free += float(open('ll.iter{}.part{}'.format(args.iteration, index))
#                            .readlines()[-1])
#    # renormalize
#    normalize_trans(args.theta, args.alpha, args.beta, args.gamma)
#    for l in range(args.L):
#        emit_probs_sums[l] /= emit_probs_sums[l].sum(axis=1).reshape(args.K, 1)
#    args.emit_probs = numpy.array([emit_probs_sums[l][:, 1] for l in range(args.L)]).T
#    args.free_energy.append(total_free)


def do_viterbi_from_args(args):
    """decode using viterbi"""
    index, iteration, matrix_txt, I, L, gmtk_master, gmtk_obs = args
    gmtk_obs = os.path.split(gmtk_obs)[1]
    strfile = 'lineagehmm.str'
    with open(gmtk_obs, 'w') as outfile:
        outfile.write(matrix_txt)

    cmd = ('gmtkViterbi -strFile {strfile} -fmt1 ascii -of1 {gmtk_obs} '
           '-nf1 0 -ni1 {obs_per_slice} -inputMasterFile {gmtk_master} '
           '-pVitValsFile viterbi.{gmtk_obs}.out'
           .format(iter=iteration, obs_per_slice=I * L, **locals()))
    print cmd
    s = subprocess.check_call(cmd, shell=True)


def parse_viterbi_states_to_Q(args, gmtk_obs):
    regex = re.compile('state_sp(\d+)\((\d+)\)=(\d+)')
    state_assigned = {}
    gmtk_obs = os.path.split(gmtk_obs)[1]
    with open('viterbi.{gmtk_obs}.out'.format(gmtk_obs=os.path.split(gmtk_obs)[1])) as infile:
        matches = regex.finditer(infile.read())
        for m in matches:
            i, t, k = map(int, m.groups())
            i -= 1
            state_assigned[i,t] = k
            #if k > args.K:
            #    import ipdb; ipdb.set_trace()
    highest_i_t = max(state_assigned.keys())  # get tuple with highest i,t
    #highest_k = max(state_assigned.values())
    highest_k = args.K - 1
    Q = numpy.zeros(tuple(d + 1 for d in highest_i_t) + (highest_k + 1,))
    for (i,t), k in state_assigned.iteritems():
        Q[i,t,k] = 1.
    return Q


def run_gmtk_lineagehmm(args):
    try:
        os.mkdir('gmtk_images')
    except:
        pass

    args.iteration = 'initial'
    args.observe = 'all'
    args.run_name = 'gmtk'
    args.out_params = 'gmtk_images/gmtk_{param}_{observe}'
    args.out_dir = '.'
    args.free_energy = []
    plot_params(args)
    # convert .npy parameters and data into gmtk format
    args.observe_txt = []
    for f in args.observe_matrix:
        obs, obs_txt = npy_observations_to_txt(f)
        args.observe_txt.append(obs_txt)

    # prepare and triangulate the graphical model
    strfile = 'lineagehmm.str'
    make_lineage_model(strfile, obs.shape[0], args.K, obs.shape[2], vert_parent=args.vert_parent, mark_avail=mark_avail, separate_theta=args.separate_theta)
    cmd = 'gmtkTriangulate -strFile %s -rePart T -findBest T  -triangulation completed ' % strfile
    #cmd = 'gmtkTriangulate -strFile %s' % strfile
    subprocess.check_call(cmd, shell=True)

    # populate theta parts if they don't exist
    if args.separate_theta and any(not hasattr(args, 'theta_%s' % i) for i in range(2,args.I+1)):
        for i in range(2,args.I+1):
            if len(args.theta.shape) == 3:
                setattr(args, 'theta_%s' % i, args.theta)
            else:
                setattr(args, 'theta_%s' % i, args.theta[i-2,:,:,:])
            args.params_to_save.append('theta_%s' % i)
        #del args.theta
        #args.params_to_save.remove('theta')


    # for each iteration...
    for args.iteration in range(1, args.max_iterations+1):
        # save args to disk
        w = npy_params_to_workspace(args)

        em_args = []
        for index in range(len(args.observe_txt)):
            gmtk_master = '%s.master' % args.observe_txt[index]
            gmtk_obs = '%s.observations' % args.observe_txt[index]
            write_workspace_simple_master(w, gmtk_master)
            em_args.append((index, args.iteration, args.observe_txt[index], args.I, args.L, gmtk_master, gmtk_obs))

        # run a gmtk em iteration on each file input, accumulating results
        if args.run_local:
            try:
                pool = multiprocessing.Pool(maxtasksperchild=1)
                pool.map_async(do_em_from_args, em_args).get(99999999)
                #map(do_em_from_args, [(i, w, args) for i in range(len(args.observe_txt))])
            except KeyboardInterrupt:
                print "Caught KeyboardInterrupt, terminating workers"
                pool.terminate()
                pool.join()
            else:
                pool.close()
                pool.join()
        else:
            pool = sge.SGEPool()
            #jobs_handle = pool.map_async(do_em_from_args, [(i, w, args) for i in range(len(args.observe_txt))], chunksize=10)
            #jobs_handle = pool.map_async(do_em_from_args, [(i, i, i) for i in range(len(args.observe_txt))], chunksize=10)
            #jobs_handle = pool.imap_unordered(do_em_from_args, em_args, chunksize=1)
            jobs_handle = pool.map_async(do_em_from_args, em_args, chunksize=1)
            # wait for all jobs to finish
            for j in jobs_handle:
                #pass
                j.wait()

        # run one final accumulator to get params
        accumulate_em_runs(args, gmtk_obs, gmtk_master)

        plot_params(args)
        plot_energy(args)

        #check convergence
        f = args.free_energy[-1]
        try:
            print 'free energy is', f, 'percent change ll:', abs(args.last_free_energy - f) / args.last_free_energy
        except AttributeError:
            print 'first iteration. free energy is', f
        else:
            if abs(abs(args.last_free_energy - f) / args.last_free_energy) < args.epsilon_e:
                print 'converged! free energy is', f
                break
        finally:
            args.last_free_energy = f

        for p in args.params_to_save:
            numpy.save(os.path.join(args.out_dir, args.out_params.format(param=p, **args.__dict__)),
                    args.__dict__[p])

    if args.run_local:
        try:
            pool = multiprocessing.Pool()
            pool.map_async(do_viterbi_from_args, em_args).get(99999999)
            #map(do_em_from_args, [(i, w, args) for i in range(len(args.observe_txt))])
        except KeyboardInterrupt:
            print "Caught KeyboardInterrupt, terminating workers"
            pool.terminate()
            pool.join()
        else:
            pool.close()
            pool.join()
    else:
        pool = sge.SGEPool()
        jobs_handle = pool.map_async(do_viterbi_from_args, em_args, chunksize=1)
        # wait for all jobs to finish
        for j in jobs_handle:
            j.wait()
    for a in em_args:
        numpy.save('viterbi_Q_' + os.path.split(a[2])[1], parse_viterbi_states_to_Q(args, a[-1]))


#if __name__ == '__main__':
#    #write_params_for_chunk(*sys.argv[1:])
#    args = get_npy_params_as_args('all.npy', approx='poc', path='../')
#    #args.observe_matrix = ['sub_mm_I3_L7.npy', 'sub_mm_I3_L7.npy']
#    args.observe_matrix = sum([glob.glob(a) for a in sys.argv[1:]], [])
#    args.free_energy = []
#    args.I, args.T, args.L = numpy.load(args.observe_matrix[0]).shape
#    args.K = args.emit_probs.shape[0]
#    mark_avail = numpy.array([[1, 1, 1, 0, 0, 0, 1],
#                           [1, 1, 1, 1, 1, 1, 1],
#                           [0, 1, 1, 1, 1, 1, 1],
#                           [0, 1, 1, 1, 1, 1, 1],
#                           [0, 1, 1, 1, 1, 1, 1],
#                           [0, 1, 1, 1, 1, 1, 1]], dtype = numpy.int8)
#
#    args.max_iterations = 30
#    args.run_local = True
#    vb_mf.normalize_trans(args.theta, args.alpha, args.beta, args.gamma)
#    run_gmtk_lineagehmm(args)
