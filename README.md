tree-hmm
========

A Tree hidden Markov model for learning epigenetic states in multiple cell types. This 
package is similar to the bioinformatics tools ChromHMM and Segway, except that it 
allows the user to explicitly model the relationships between cell types or species.


Please see our paper for further details! http://www.ncbi.nlm.nih.gov/pubmed/23734743


Quickstart
----------
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


    # download some sample .bam files from our server and histogram them to binary .npy
    mkdir data && cd data
    tree-hmm convert 
        --download_first \
        --base_url "https://cbcl.ics.uci.edu/public_data/tree-hmm-sample-data/%s" \
        --chromosomes chr19 \
        --bam_template "data/wgEncodeBroadHistone{species}{mark}StdAlnRep{repnum}.REF_chr19.bam" \
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
* Scipy
* Matplotlib (optional for plotting)
* PySAM (optional for converting SAM files to 0/1 Numpy matrices)
* Cython >= 0.15.1+ (required for converting .pyx => .c => .so)

Platform-specific installation instructions are available in the quickstart.  
Please note we don't support windows!


Installing tree-hmm
-------------------
* Easiest way:

    sudo pip install treehmm

* For the latest development version:

    sudo pip install -e git+https://github.com/uci-cbcl/tree-hmm.git#egg=tree_hmm

* To be able to hack on the code yourself:
    
    git clone https://github.com/uci-cbcl/tree-hmm.git
    cd tree-hmm
    sudo pip install -e .

* Or alternatively, if you don't have admin rights:

    git clone https://github.com/uci-cbcl/tree-hmm.git
    cd tree-hmm
    python setup.py build_ext --inplace
    export PYTHONPATH=`pwd`:$PYTHONPATH
    export PATH=`pwd`/bin:$PATH


Running the commands
--------------------
Use the tree-hmm command from the command line to perform the major commands, including:
* **convert** a set of SAM format mapped reads to numpy matrices
* **split** a chromosome into several variable-length chunks, determined via a 
    gaussian convolution of the raw read signal
* **infer** underlying chromatin states from a converted binary matrix
* **q_to_bed** convert the numpy probability matrices into BED files (maximum a posteriori state)

Each of these tasks has its own command-line help, accessible via:

    tree-hmm convert --help
    tree-hmm split --help
    tree-hmm infer --help
    tree-hmm q_to_bed --help


Data Conversion
---------------

    tree-hmm convert --download_first --marks Ctcf H3k4me2 --species H1hesc K562

We require at least one `.bam` input file for each species/mark combination.  
During this step, all of the .bam files will be scanned and histogrammed.
Replicants are pooled together and the reads are binarized using a poisson
background rate specific to each mark/species combination, i.e.,:

    bgrate_{il} = \frac{ \sum_{t=1}^T count_{itl} }{ T }
    markpresence_{il} = poisson.survival(count_{itl}, bg_rate_{il}) < max_pvalue

where max_pvalue can be specified by the user.  This simply imposes a threshold value that's
specific to a species/mark combination and should account for sequencing depth differences
and a variable number of replicates.  Finally, the histograms are split by chromosome and, 
by default, written to `observations.{chrom}.npy`.


The ENCODE histone data available on UCSC can be downloaded by specifying the `--download_first` 
option, or you can specify other datasets by changing `--base_url` and `--bam_template`, just
so long as the files are named systematically.  See the quickstart above for an example using 
sample ENCODE data from hg19's chr19.


The species and marks in use can be modified either by directly editing the treehmm/static.py
file or by specifying `--species` and/or `--marks` in  the convert step. Again, see the quickstart
for an example.

This step creates a new file `start_positions.pkl` which contains information about the original
coordinates and the species and marks used during the conversion. This file (or its twin created
by the `split` step) is required by the final `q_to_bed` step.

Note: there is already preliminary support for missing marks/species already present which I 
can make more user-friendly if there is interest.  Raise an issue on Github if you're interested.



Splitting the data
------------------
Since the majority of the genome will not have any signal in any of the marks, it is very useful 
to split up the histone data and only perform inference on the regions with adequate signal. This
step can also speed up inference as only a portion of the genome is used and each chunk is 
considered independent of all others.

We use a gaussian smoothing filter over the binarized data and cut out regions where there is 
no signal in any species or mark.  A histogram of the resulting read lengths is drawn to
help identify how much of the genome is retained for a given setting.  The defaults retained
about 50% of hg19 on the ENCODE data.

    tree-hmm split \
        observations.chr*.npy 
        start_positions.pkl \
        --min_reads .1 \
        --gauss_window_size 5 \
        --min_size 25

Note: the `--gauss_window_size` and `--min_size` are in terms of *bins*.  So if you want a 
smoothing window that acts over 10kb up and downstream (20kb total), and had specified a
`--window_size` of 200bp in the convert phase, you'd specify a `--gauss_window_size` of 50.

This step creates a new file `start_positions_split.pkl` which retains information about the 
original read coordinates. This file (or its twin created by the `convert` step) is required during the final `q_to_bed` step.



Inference
---------
For inference, you must specify the number of hidden states to infer and one or more input 
files.  There are also many parameters you can fine-tune with defaults as used in the paper.

Inference will try to submit jobs to an SGE cluster.  If you don't have such a cluster, 
make sure you specify the `--run_local` option.  If you do, you should clean up the 
`SGE_*` files when inference is complete.  Those files contain parameters and return
values job submissions.

There are several inference engines implemented for comparison including:
* **mf**: mean field approximation -- the simplest variational approximation, 
    with every node optimized independently. Memory use is O(I * T * K), that is, it scales 
    linearly with the number of species (I), the number of bins (T), and the number of states (K).
* **loopy**: loopy belief propagation -- similar to mean field in that each node is optimized
    independently, but is has a non-monotonic energy trajectory.  Works well in some applications,
    but not very well in ours.  Memory use is O(I * T * K).
* **poc**: product of chains -- the entire chain from each species is solved exactly, 
    but different chains are optimized indepently. Memory use is O(I * T * K^2).
* **clique**: the entire graph is solved exactly using a naive cliquing approach. Memory 
    use is O(T * K^I). **This mode EATS memory** and is not recommended for K > 10.

There are also a few more "experimental" modes:
* **indep**: independent chains -- the entire chain from each species is solved exactly, 
    but there are no connections between different chains (this is how ChromHMM 
    handles joint inference). Memory use is O(I * T * K^2).
* **gmtk**: exact inference using the Graphical Models Toolkit.  If you have GMTK installed, 
    this package provides an alternative to the clique mode.  We found it to be a bit slower
    clique, but has much more reasonable memory usage.  The easiest way to get GMTK might be to
    follow the documentation of segway, which requires it to run.
    http://noble.gs.washington.edu/proj/segway/



To infer, e.g., K=5 hidden states on the included test data, you can do:

    tree-hmm infer 5 test_data/*.npy

Or to infer 18 states, but only do five M-step iterations, up to 10 E-step 
iterations per M-step, use the product-of-chains approximation on the entire
converted and split set of files, iterating until the change in free energy 
is < 1e-5 in either the E-step or the M-step and running in parallel locally
rather than on an SGE grid:

    tree-hmm infer \
        18 \
        --max_iter 5 \
        --max_E_iter 10 \
        --approx poc \
        --epsilon 1e-5 \
        --epsilon_e 1e-5 \
        --run_local \
        "observations.chr*.chunk*.npy"


After running this, you'll find a new directory infer_out/mf/TIMESTAMP/.

In this directory, you'll find several png formatted files showing the 
free energy trajectory across iterations, the parameters as they are 
learned (plotted at each iteration) as well as the Q distributions 
(marginal probabilities of each node being in a particular state) and the 
contributions of each job's inferred state to each parameter (i.e., the 
transition matrices alpha, beta, gamma, and theta as well as the emission 
matrix e).


Post-processing
---------------
To make any sense of the actual genomic segmentation, you'll need to convert the marginal Q
probabilities into BED-formatted files.  If you used the `split` subcommand, you need
to specify the `start_positions_split.pkl` file generated by that command:

    tree-hmm q_to_bed infer_out/mf/TIMESTAMP/ start_positions_split.pkl

If you did not use the `split`, you may use the original pkl file:

    tree-hmm q_to_bed infer_out/mf/TIMESTAMP/ start_positions.pkl

These commands will find and output the most likely state assignment in each bin for 
all species to a set of bed files: `treehmm_states.{species}.state{k}.bed`.  

Note that this is the maximum a posteriori (MAP) assignment, NOT the 
most likely joint configuration. ChromHMM also outputs the MAP, whereas Segway uses the most 
likely joint configuration or  viterbi path.  The `gtmk` inference mode can find the most 
likely joint configuration, but downstream tools are lacking at the moment.  If you're
 interested in this, please raise an issue on Github.

You can also get the full probability matrix (not just most likely state) by specifying 
`--save_probs`.  This step relies on the `start_positions.pkl` file generated during the 
`split` phase.  You may specify where that file is located via `--start_positions`.  If you 
don't want to split your data beyond by-chromosome, I can modify this step accordingly.
Again, please raise an issue on Github if you're interested.


Finally, you may want to check out pybedtools or Galaxy to do downstream analysis.