
import os
from math import ceil

import matplotlib
from matplotlib import pyplot
import scipy as sp

from treehmm.static import valid_species, valid_marks


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
    if args.separate_theta:
        theta_tmp = args.theta
        for i in range((args.theta.shape)[0]):
            setattr(args, 'theta_%s'%(i+1), args.theta[i,:,:,:])

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
