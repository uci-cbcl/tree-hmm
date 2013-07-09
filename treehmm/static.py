
import scipy as sp

#try:
#    from ipdb import set_trace as breakpoint
#except ImportError:
#    from pdb import set_trace as breakpoint



###################### ENCODE human ######################
valid_species = ['H1hesc', 'K562', 'Gm12878', 'Hepg2', 'Huvec', 'Hsmm', 'Nhlf', 'Nhek',
                'Hmec' ]  # 'Helas3' is extra
# valid_marks = ['Ctcf', 'H2az', 'H3k27ac', 'H3k27me3', 'H3k36me3', 'H3k4me1',
#            'H3k4me2', 'H3k4me3', 'H3k79me2', 'H3k9ac', 'H3k9me3', 'H4k20me1']
valid_marks = ['Ctcf', 'H3k27me3', 'H3k36me3', 'H4k20me1', 'H3k4me1', 'H3k4me2', 'H3k4me3', 'H3k27ac',
              'H3k9ac',  'Control',]  # no H2az or H3K9me3
# valid_marks = ['Ctcf', 'H3k27ac', 'H3k27me3', 'H3k36me3', 'H3k4me2', 'H3k4me3',
#               'H3k9ac', 'H4k20me1', 'Control',]  # H3K4me1 has an issue in Hepg2-- skipping for now

phylogeny = {'H1hesc':'H1hesc', 'Huvec':'H1hesc', 'Hsmm':'H1hesc',
            'Nhlf':'H1hesc', 'Gm12878':'H1hesc', 'K562':'H1hesc',
            'Nhek':'H1hesc', 'Hmec':'H1hesc', 'Hepg2':'H1hesc'}
mark_avail = sp.zeros((len(valid_species), len(valid_marks)), dtype=sp.int8)


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



# ############### ENCODE Human with RNA-seq ################
# valid_species = ['H1hesc', 'K562', 'Gm12878', 'Hepg2', 'Huvec', 'Hsmm', 'Nhlf', 'Nhek', 'Hmec' ]
# valid_marks = ['Ctcf', 'H2az', 'H3k27ac', 'H3k27me3', 'H3k36me3', 'H3k4me1',
#             'H3k4me2', 'H3k4me3', 'H3k79me2', 'H3k9ac', 'H3k9me3', 'H4k20me1']
# #valid_marks = ['Ctcf', 'H3k27ac', 'H3k27me3', 'H3k36me3', 'H3k4me2', 'H3k4me3',
# #               'H3k9ac', 'H4k20me1', 'Control',]  # H3K4me1 has an issue in Hepg2-- skipping for now
# valid_marks += ['CellLongnonpolya', 'CellPap', 'CellTotal'
#                 'CytosolLongnonpolya', 'CytosolPap',
#                 'NucleusLongnonpolya', 'NucleusPap',
#                 'NucleoplasmTotal',
#                 'NucleolusTotal',
#                 'ChromatinTotal',]
# mark_avail = sp.zeros((len(valid_species), len(valid_marks) * 3), dtype=sp.int8)
# phylogeny = {'H1hesc':'H1hesc', 'Huvec':'H1hesc', 'Hsmm':'H1hesc',
#              'Nhlf':'H1hesc', 'Gm12878':'H1hesc', 'K562':'H1hesc',
#              'Nhek':'H1hesc', 'Hmec':'H1hesc', 'Hepg2':'H1hesc'}




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


# use longdouble if getting underflow/divide by zero errors... not guaranteed to help though!
# also note longdouble's aren't as well supported in scipy (some versions don't support the necessary ufunc's)
# float_type = sp.longdouble  
float_type = sp.double
