tree-hmm
========

A Tree hidden Markov model for learning epigenetic states in multiple cell types



Prerequisites
-------------
* Scipy
* Matplotlib (optional for plotting)
* PySAM (optional for converting SAM files to 0/1 Numpy matrices)
* Cython >= 0.15.1+ (required for converting .pyx => .c => .so)


Compiling
---------
To generate the .so files for vb_mf and vb_prodc, run the command:

    python setup.py build_ext --inplace

from within the tree-hmm directory.  Add this directory to your PYTHONPATH.


Running
-------
Use the histone_tree_hmm.py file to perform the major commands, including:
* **convert** a set of SAM format mapped reads to numpy matrices
* **trim** a chromosome's ends off (telomeric regions; not used in the paper)
* **split** a chromosome into several variable-length chunks, determined via a 
    gaussian convolution of the raw read signal
* **infer** underlying chromatin states from a converted binary matrix

For example, to infer 5 hidden states on the test data, you can do:

    python histone_tree_hmm.py infer 5

Or do one M-step iteration, up to 50 E-step iterations using the 
product-of-chains approximation on a large set of files, iterating until
the change in free energy is < 1e-5 in either the E-step or the M-step and 
running in parallel locally rather than on a grid:

    python ~/Dropbox/histone_var_bayes/code/histone_vb_cython3.py infer 18 \
        --max_iter 1 --max_E_iter 50 --approx poc --epsilon 1e-5 \
        --epsilon_e 1e-5 --run_local \
        "/data/encode_histone/chr*_all.i*.chunk*.npy"

Each subcommand has several additional options available by specifying the -h 
option on the command-line

    python histone_tree_hmm.py convert --help


Post-processing
---------------
Use convert_Q_to_genomic_coordinates.py to recover the BED-format file from the
.npy Q matrices.


Example Run
-----------
Infer 5 states using mean-field approximation on the included data for 
chromosome 22:

    python histone_tree_hmm.py infer 5 "test_data/chr22_all.chunk*.npy" \
        --run_local --max_iter 5 --approx mf

After running this, you'll find a new directory infer_out/mf/TIMESTAMP/

In this directory, you'll find several png formatted files showing the 
free energy trajectory across iterations, the parameters as they are 
learned (plotted at each iteration) as well as the Q distributions 
(marginal probabilities of each node being in a particular state) and the 
contributions of each job's inferred state to each parameter (i.e., the 
transition matrices alpha, beta, gamma, and theta as well as the emission 
matrix e) 

