
This package implements a tree hidden Markov model for learning epigenetic 
states in multiple cell types. It's similar to the bioinformatics tools 
`ChromHMM <http://compbio.mit.edu/ChromHMM/>`_ and 
`Segway <http://noble.gs.washington.edu/proj/segway/>`_, except that it allows 
the user to explicitly model the relationships between cell types or
species. Please see our `paper <http://www.ncbi.nlm.nih.gov/pubmed/23734743>`_ 
for further details!

For the development version, see our page on 
`Github <https://github.com/uci-cbcl/tree-hmm>`_.

Quickstart
----------
::

    # INSTALLATION
    # If you're on Ubuntu
    sudo apt-get install python-pip python-scipy python-matplotlib cython git

    # OR if you're on a mac
    ruby -e "$(curl -fsSL https://raw.github.com/mxcl/homebrew/go)"
    brew install python scipy matplotlib cython git

    # don't forget python prerequisite `pysam`
    sudo pip install -U pysam


    # grab and build the tree-hmm code
    # easiest way first:
    sudo pip install -U treehmm

    # OR using the latest development version
    git clone https://github.com/uci-cbcl/tree-hmm.git
    cd tree-hmm

    # building the package:
    # don't install-- just test:
    python setup.py build_ext --inplace
    export PYTHONPATH=$PYTHONPATH:`pwd`
    export PATH=$PATH:`pwd`/bin

    # OR go ahead and install
    sudo pip install -e .


    # RUNNING
    # download some sample .bam files from our server and histogram them to binary .npy
    mkdir data && cd data
    tree-hmm convert 
        --download_first \
        --base_url "https://cbcl.ics.uci.edu/public_data/tree-hmm-sample-data/%s" \
        --chromosomes chr19 \
        --bam_template "wgEncodeBroadHistone{species}{mark}StdAlnRep{repnum}.REF_chr19.bam" \
        --marks Ctcf H3k27me3 H3k36me3 H4k20me1 H3k4me1 H3k4me2 H3k4me3 H3k27ac H3k9ac \
        --species H1hesc K562 Gm12878 Huvec Hsmm Nhlf Nhek Hmec


    # split chr19 into smaller pieces
    tree-hmm split observations.chr19.npy


    # do inference on the resulting chunks (creates infer_out directory)
    tree-hmm infer 5 --max_iter 3 --approx poc --run_local "observations.chr19.chunk*.npy"


    # convert the inferred states into BED format for viewing on the genome browser
    tree-hmm q_to_bed infer_out/mf/*


    # upload the BED files to UCSC, analyze, etc


Prerequisites
-------------
-  Scipy
-  Matplotlib (optional for plotting)
-  PySAM (optional for converting SAM files to 0/1 Numpy matrices)
-  Cython >= 0.15.1+ (required for converting .pyx => .c => .so)

Platform-specific installation instructions are available in the quickstart. 
Please note we don't support windows!


Installing tree-hmm
-------------------
-  The easiest way::

    sudo pip install treehmm

-  For the latest development version::
   
    sudo pip install git+https://github.com/uci-cbcl/tree-hmm.git#egg=treehmm

-  To be able to hack on the code yourself::

    git clone https://github.com/uci-cbcl/tree-hmm.git
    cd tree-hmm
    sudo pip install -e .

-  Or alternatively, if you don't have admin rights::

    git clone https://github.com/uci-cbcl/tree-hmm.git
    cd tree-hmm
    python setup.py build_ext --inplace
    export PYTHONPATH=`pwd`:$PYTHONPATH
    export PATH=`pwd`/bin:$PATH


Running the commands
--------------------
Use the tree-hmm command from the command line to perform the major
commands, including:

-  **convert** a set of BAM-formatted mapped reads to numpy matrices 

-  **split** a chromosome into several variable-length chunks, 
   determined via a gaussian convolution of the raw read signal 

-  **infer** underlying chromatin states from a converted binary matrix

-  **q_to_bed** convert the numpy probability matrices into BED files
   (maximum a posteriori state)

Each of these tasks has its own command-line help, accessible via::

    tree-hmm convert --help
    tree-hmm split --help
    tree-hmm infer --help
    tree-hmm q_to_bed --help


Data Conversion
---------------
::

    tree-hmm convert \
        --download_first \
        --marks Ctcf H3k4me2 \
        --species H1hesc K562

We require at least one BAM input file for each species/mark combination.
You can specify the naming convention your data follows via the 
``--bam_template`` argument.

During this step, all of the BAM files will be scanned and
histogrammed. Replicants are pooled together and the reads are binarized
using a poisson background rate specific to each mark/species
combination, i.e.,::

    bgrate_{il} = \frac{\sum_{t=1}^T count_{itl}} {T}
    markpresence_{il} = poisson.survival(count_{itl}, bg_rate_{il}) < max_pvalue

where `max_pvalue` can be specified by the user. This simply imposes a
threshold value that's specific to a species/mark combination and should
account for sequencing depth differences and a variable number of
replicates. Finally, the histograms are split by chromosome and, by
default, written to ``observations.{chrom}.npy``.

The ENCODE histone data available on UCSC can be downloaded by
specifying the ``--download_first`` option, or you can specify other
datasets by changing ``--base_url`` and ``--bam_template``, just so long
as the files are named systematically. See the quickstart above for an
example using sample ENCODE data from hg19's chr19.

The species and marks in use can be modified either by directly editing
the treehmm/static.py file or by specifying ``--species`` and/or
``--marks`` in the convert step. Again, see the quickstart for an
example.

This step creates a new file ``start_positions.pkl`` which contains
information about the original coordinates and the species and marks
used during the conversion. This file (or its twin created by the
``split`` step) is required by the final ``q_to_bed`` step.

Note: there is already preliminary support for missing marks/species
already present which I can make more user-friendly if there is
interest.  There is also some work on allowing continuous observations (rather
than the current binary-only observations). Raise an issue/feature request on 
Github if you're interested.


Splitting the data
------------------
Since the majority of the genome will not have any signal in any of the
marks, it is very useful to split up the histone data and only perform
inference on the regions with adequate signal. This step can also speed
up inference as only a portion of the genome is used and each chunk is
considered independent of all others.

We use a gaussian smoothing filter over the binarized data and cut out
regions where there is no signal in any species or mark. A histogram of
the resulting read lengths is drawn to help identify how much of the
genome is retained for a given setting. The defaults retained about 50%
of hg19 on the ENCODE data.
::

    tree-hmm split \
        observations.chr*.npy 
        start_positions.pkl \
        --min_reads .1 \
        --gauss_window_size 5 \
        --min_size 25

Note: the ``--gauss_window_size`` and ``--min_size`` are in terms of
*bins*. So if you want a smoothing window that acts over 10kb up and
downstream (20kb total), and had specified a ``--window_size`` of 200bp
in the convert phase, you'd specify a ``--gauss_window_size`` of 50.

This step creates a new file ``start_positions_split.pkl`` which retains
information about the original read coordinates. This file (or its twin
created by the ``convert`` step) is required during the final
``q_to_bed`` step.


Inference
---------
For inference, you must specify the number of hidden states to infer and
one or more input files. There are also many parameters you can
fine-tune with defaults as used in the paper.

Inference will try to submit jobs to an SGE cluster. If you don't have
such a cluster, make sure you specify the ``--run_local`` option. If you
do, you should clean up the ``SGE_*`` files when inference is complete.
Those files contain parameters and return values job submissions.

There are several inference engines implemented for comparison
including:

:mean field approximation (mf):  the simplest variational
    approximation, with every node optimized independently. Memory
    use is O(I \* T \* K), that is, it scales linearly with the 
    number of species (I), the number of bins (T), and the number of states
    (K). 

:loopy belief propagation (loopy):  similar to mean field in that
    each node is optimized independently, but is has a non-monotonic
    energy trajectory. Works well in some applications, but not very 
    well in ours. Memory use is O(I \* T \* K). 

:product of chains (poc):   the entire chain from each species is 
    solved exactly, but different chains are optimized indepently.
    Memory use is O(I \* T \* K^2). This mode performed the best in
    our testing.

:Exact inference using a cliqued HMM (clique):  the entire graph is
    solved exactly using a naive cliquing approach. Memory use is
    O(T \* K^I). **This mode EATS memory** and is not recommended for 
    K > 10.

There are also a few more "experimental" modes: 

:Independent chains (indep):    the entire chain from each species is
    solved exactly, but there are no connections between different 
    chains (this is how ChromHMM handles joint inference). 
    Memory use is O(I \* T \* K^2).

:Exact inference using the Graphical Models Toolkit (gmtk):  exact
    inference using the Graphical Models Toolkit. If you have GMTK
    installed, this package provides an alternative to the clique 
    mode. We found it to be a bit slower clique, but has much more 
    reasonable memory usage. The easiest way to get GMTK might be to 
    follow the documentation of  
    `segway <http://noble.gs.washington.edu/proj/segway/>`_,  which
    requires it to run.


Changing the phylogeny
**********************
The phyologeny connecting each species is specified in `treehmm/static.py` and
is in the form of a python dictionary, each of whose keys are the child cell 
type and whose values are the parent cell type. For example, the default 
phylogeny used for the ENCODE data specifies that ``H1hesc`` is the parent of 
all other cell types::

    phylogeny = {'H1hesc':'H1hesc', 
                 'Huvec':'H1hesc',
                 'Hsmm':'H1hesc',
                 'Nhlf':'H1hesc',
                 'Gm12878':'H1hesc',
                 'K562':'H1hesc',
                 'Nhek':'H1hesc',
                 'Hmec':'H1hesc',
                 'Hepg2':'H1hesc'}

A few things to note:

-  Cell types that specify themselves as their own parents (like ``H1hesc``
   above) are considered root nodes (they have no parents) and use a different
   set of parameters than cell types with parents.
-  While each cell type is allowed to have zero or as many children as you want,
   each cell type is only allowed to have a single parent. This is enforced
   already since dictionaries can't have duplicate keys.
-  Connecting the tree in a loop (s.t. there is no root node) violates a
   fundamental assumption in Bayesian networks (they are supposed to be
   directed, Acyclic Graphs). The code may run okay, but you will probably get
   incorrect results.
-  You can have multiple roots in a graph (e.g., two independent trees or one 
   tree and a single, unrelated cell type). This might help in learning a global
   set of parameters (more data), but shouldn't affect the inference quality 
   within each tree. Cell types without parents or children will behave exactly
   like standard hidden Markov models.
-  We have preliminary code that uses a separate transition parameterization for
   each parent:child relationship.  While this mode should increase accuracy and
   may reveal some unique trends along each branch of your phylogeny, keep in 
   mind that the number of parameters is greatly increased and can lead to 
   overfitting.  You may want to reduce the number of states in use (K).  If
   you're interested in this mode, contact me/raise an issue on Github.
-  Internal "hidden" nodes are possible using the ``--mark_avail`` parameter
   (which in turn is allowing some marks/species to be missing). This mode 
   has some weird side-effects when run on heterogeneous mark combinations and
   wasn't pursued further.


Running on SGE vs. locally
**************************
By default, the inference stage in tree-hmm will try to submit jobs through the
SGE grid.  If it can't do so, it will fall back to running the jobs locally.
Here are a few tips to make things run faster:

-   If you haven't split your data into chunks using the ``split`` subcommand,
    you should set the ``--chunksize`` to 1.  That way, each chromosome will be
    handled by a different job.
-   For split data running on SGE, set the ``--chunksize`` fairly high (like 
    100, or even more if you have tens of thousands of chunks).  This will start
    fewer SGE jobs that will each run longer and save you from being yelled at 
    by your system administrators
-   If you're not using SGE, you should explicitly set ``--run_local`` since 
    it will use a more efficient message passing algorithm (inter-process 
    communication rather than writing parameters and results to disk).
-   If you are using SGE, be sure to clean up the (many, many) ``SGE_*`` files
    which serve as temporary messages outputs while the job is running.


Quick Example
*************
To infer K=18 states, but only do five M-step iterations, up to 10
E-step iterations per M-step, use the product-of-chains approximation on
the entire converted and split set of files, iterating until the change
in free energy is < 1e-5 in either the E-step or the M-step and running
in parallel locally rather than on an SGE grid::

    tree-hmm infer \
        18 \
        --max_iter 5 \
        --max_E_iter 10 \
        --approx poc \
        --epsilon 1e-5 \
        --epsilon_e 1e-5 \
        --run_local \
        "observations.chr*.chunk*.npy"

After running this, you'll find a new directory ``infer_out/mf/TIMESTAMP/``.

In this directory, you'll find several png formatted files showing the
free energy trajectory across iterations, the parameters as they are
learned (plotted at each iteration) as well as the Q distributions
(marginal probabilities of each node being in a particular state) and
the contributions of each job's inferred state to each parameter (i.e.,
the transition matrices alpha, beta, gamma, and theta as well as the
emission matrix e).


Post-processing
---------------
To make any sense of the actual genomic segmentation, you'll need to
convert the marginal Q probabilities into BED-formatted files. If you
used the ``split`` subcommand, you need to specify the
``start_positions_split.pkl`` file generated by that command::

    tree-hmm q_to_bed infer_out/mf/TIMESTAMP/ start_positions_split.pkl

If you did not use the ``split``, you may use the original pkl file::

    tree-hmm q_to_bed infer_out/mf/TIMESTAMP/ start_positions.pkl

These commands will find and output the most likely state assignment in
each bin for all species to a set of bed files 
``treehmm_states.{species}.state{k}.bed``.

Note that this is the maximum a posteriori (MAP) assignment, NOT the
most likely joint configuration. ChromHMM also outputs the MAP, whereas
Segway uses the most likely joint configuration or viterbi path. The
``gtmk`` inference mode can find the most likely joint configuration,
but downstream tools are lacking at the moment. If you're interested in
this, please raise an issue on Github.

You can also get the full probability matrix (not just most likely
state) by specifying ``--save_probs``. This step relies on the
``start_positions.pkl`` file generated during the ``split`` phase. You
may specify where that file is located via ``--start_positions``. If you
don't want to split your data beyond by-chromosome, I can modify this
step accordingly. Again, please raise an issue on Github if you're
interested.

Finally, you may want to check out `pybedtools <https://github.com/daler/pybedtools>`_ or 
`Galaxy <https://main.g2.bx.psu.edu/>`_ to do downstream analysis.
