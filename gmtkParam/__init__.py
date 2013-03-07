"""Tools for reading writing and manipulating GMTK parameter files

@author:    Arthur Kantor
@contact:       akantorREMOVE_THIS@uiuc.edu
@copyright: 2008
@license:       GPL version 3
@date:      11/18/2008
@version:       0.2

History:
This module is derived from a piece of the SVC library
which is Copyright (C) 2006-2008 by Jan Svec, honza.svec@gmail.com
@see: http://code.google.com/p/extended-hidden-vector-state-parser/

"""

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os.path
from types import StringTypes
from fnmatch import fnmatchcase
from copy import deepcopy
import subprocess

try:
    import numpy
except ImportError:
    pass
#TODO
#FIXME
#Arthur 11/18/2008
#1) children should not insert themselves into their parents
#2) replace dicts in Workspace and DTs class with NamedObjectCollection

_LEAF = -1

class DCPTDict(dict):
    def flat(self):
        for key in sorted(self.keys()):
            for i in self[key]:
                yield i

#class WordFile(file):
    #"""Class for reading file wordwise
    #"""
    #def __init__(self, *args):
        #super(WordFile, self).__init__(*args)
        #self._stack = []

    #def readWords(self):
        #while not self._stack:
            #line = self.readline()
            #if not line:
                #break
            #line = line.split('%', 1)[0]
            #self._stack.extend(line.split())
            #while self._stack:
                #yield self._stack.pop(0)


class ProbTable(object):
    def __init__(self, scard, pcard):
        super(ProbTable, self).__init__()
        if not issequence(scard):
            scard = [scard]
        if not issequence(pcard):
            pcard = [pcard]
        self._nvars = len(scard)
        self._npars = len(pcard)
        self._cvars = tuple(scard)
        self._cpars = tuple(pcard)
        self._table = self._initTable(scard, pcard)

    def __eq__(self, other):
        for attr in ('_cvars', '_cpars', '_eqTables'):
            if not hasattr(other, attr):
                return NotImplementedError()

        return (self._cvars == other._cvars) and \
               (self._cpars == other._cpars) and \
               self._eqTables(other)

    def _initTable(self, scard, pcard):
        raise TypeError('Abstract method called')

    def _eqTables(self, other):
        raise TypeError('Abstract method called')

    def getTable(self):
        return self._table

    table = property(getTable)

    def _convertSlices(self, item):
        var = []
        cond = []
        cur = var
        if not issequence(item):
            item = [item]
        for i in item:
            if isinstance(i, slice):
                if cur is cond:
                    raise ValueError("Malformed probability index")
                if i.step is not None:
                    raise ValueError("Malformed probability index")
                if i.start is not None:
                    var.append(i.start)
                if i.stop is None:
                    raise ValueError("Malformed probability index")
                else:
                    cond.append(i.stop)
                cur = cond
            else:
                cur.append(i)
        return tuple(var), tuple(cond)

    def _getMassFunction(self, parents):
        raise TypeError('Abstract method called')

    def _createMassFunction(self, parents):
        raise TypeError('Abstract method called')

    def _delMassFunction(self, parents):
        raise TypeError('Abstract method called')

    def __getitem__(self, item):
        var, cond = self._convertSlices(item)
        mf = self._getMassFunction(cond)
        if len(var) > 1:
            return mf[var]
        elif len(var) == 1:
            return mf[var[0]]
        else:
            return mf

    def __setitem__(self, item, value):
        var, cond = self._convertSlices(item)
        mf = self._createMassFunction(cond)
        if len(var) > 1:
            mf[var] = value
        elif len(var) == 1:
            mf[var[0]] = value
        elif len(mf) == len(value):
            #since dcpt[:1] returns the mf, it seems that dcpt[:1] =[.4 .6] should also work...
            mf[:]=value[:]
        else:
            raise ValueError('Cardinality is %d, but attempting to assign a mass function of cardinality %d.' % (len(mf) ,len(value)))

    def __delitem__(self, item):
        var, cond = self._convertSlices(item)
        if len(var) > 0:
            raise IndexError('You can delete only whole mass function, eg. [:2, 1, 3]')
        self._delMassFunction(cond)

    def getParentCards(self):
        return self._cpars

    parentCards=property(getParentCards)

    def getSelfCards(self):
        return self._cvars

    selfCards=property(getSelfCards)


class DenseTable(ProbTable):
    def _initTable(self, scard, pcard):
        if len(scard) != 1:
            raise ValueError("DenseTable currently supports only 1D variables")
        ret = DCPTDict()
        for p in self.possibleParents:
            ret[p] = [0.] * scard[0]
        return ret

    def _getMassFunction(self, parents):
        if len(parents) != self._npars:
            raise ValueError('Invalid count of parents')
        return self._table[parents]

    def _eqTables(self, other):
        return self._table == other._table

    _createMassFunction = _getMassFunction

    def changeParentCards(self, newParentCards):
        """reshape table to have new parents with newParentCards cardinalities.
           If new mass functions are added, they are set to 0 vectors """

        if not issequence(newParentCards):
            newParentCards = [newParentCards]

        if 0 in newParentCards:
            raise ValueError("cannot have 0 cardinality in parents")

        #check if some PMs need to be deleted
        toBeDeleted=[ range(newParentCards[p], self.parentCards[p]) for p in range(min(len(newParentCards), len(self.parentCards)))]
        toBeDeleted += [ range(self.parentCards[p]) for p in range(len(newParentCards), len(self.parentCards))]

        if len(newParentCards)<len(self.parentCards):
            raise NotImplementedError("the number of parents cannot decrease.  Not implemented because reordering parents might be confusing.")

        parCards=list(self._cpars)

        for i,p in enumerate(toBeDeleted):
            parRanges = [range(j) for j in parCards]
            parRanges[i]=p
            for pmKey in cartezian(*parRanges):
                del self.table[pmKey]
            if parCards[i]>newParentCards[i]:
                parCards[i]=newParentCards[i]

        #now check if some PMs need to be added
        toBeAdded=[range(self.parentCards[p], newParentCards[p]) for p in range(min(len(newParentCards), len(self.parentCards)))]
        toBeAdded += [range(newParentCards[p]) for p in range(len(self.parentCards), len(newParentCards))]

        for i,p in enumerate(toBeAdded):
            parRanges = [range(j) for j in parCards]
            parRanges[i]=p
            for pmKey in cartezian(*parRanges):
                self.table[pmKey] = [0.] * self.selfCards[0]
            if parCards[i]<newParentCards[i]:
                parCards[i]=newParentCards[i]

        self._cpars=tuple(parCards)


    def _delMassFunction(self, parents):
        if len(parents) != self._npars:
            raise ValueError('Invalid count of parents')
        del self._table[parents]


    def getPossibleParents(self):
        keys = [range(i) for i in self.parentCards]
        if keys:
            return cartezian(*keys)
        else:
            return [()]

    #trick to make getters and setters virtual -
    #if you override it in a derived class, it will be used
    #def _possibleParentsGet(self): return self.getPossibleParents()
    #possibleParents=property(_possibleParentsGet)

    possibleParents=property(getPossibleParents)

class _Object(object):
    """A named GMTK object.  The base class for all GMTK parameter objects"""
    def __init__(self, parent, name, *args):
        super(_Object, self).__init__(*args)
        self.name = name
        self.parent = parent
        self.parent[self.__class__] = self

    def __repr__(self):
        return '<%s object, name %r>' % (self.__class__.__name__, self.name)

    @classmethod
    def readFromFile(cls, parent, stream):
        """Read the object from stream.
            @param parent: Read object will be added to the container parent
            @type  stream: WorkspaceIO
            @param stream: Where the object will be read from

        """
        raise TypeError('Abstract method called')

    def writeToFile(self, stream):
        """Write the object to stream.
            @type  stream: WorkspaceIO
            @param stream: Where the object will be written to

        """
        raise TypeError('Abstract method called')

    def setName(self, name):
        self._name = name

    def getName(self):
        return self._name
    name = property(getName,setName)
    """Object name"""

    def setParent(self, parent):
        self._parent = parent

    def getParent(self):
        return self._parent

    parent = property(getParent, setParent)


class NameCollection(_Object, list):
    def __init__(self, parent, name):
        _Object.__init__(self,parent,name)

    @classmethod
    def readFromFile(cls, parent, stream):
        name = stream.readWord()
        coll = cls(parent, name)

        n_obj = stream.readInt()
        for i in range(n_obj):
            coll.append(stream.readWord())
        return coll

    def writeToFile(self, stream):
        stream.writelnWord(self.name)
#           ostr=UnnumberedOutStream(stream)
#           for i in self:
#               ostr.writeObject(i)
#           ostr.finalize()
        stream.writelnInt(len(self))
        for i in self:
            stream.writelnWord(i)

    def __eq__(self, other):
        return list(self) == list(other)


class TreeLeaf(object):
    def __init__(self, expression,):
        if isinstance(expression, int):
            self.eval = False
            self.expression = expression
            self.sexpression = str(expression)
        elif expression[0] == '{' and expression[-1] == '}':
            self.sexpression = expression
            expression = expression[1:-1].replace('!', ' not ')
            expression = expression.replace('&&', ' and ')
            expression = 'int(%s)' % expression.replace('||', ' or ')
            self.eval = True
            self.expression = compile(expression, '', 'eval')
        else:
            self.sexpression = expression
            self.eval = False
            self.expression = int(expression)

    def __call__(self, parents):
        if self.eval:
            ns = {}
            for i, value in enumerate(parents):
                ns['p%d' % i] = value
            return eval(self.expression, ns)
        else:
            return self.expression

    def __eq__(self, other):
        if not hasattr(other, 'expression'):
            return NotImplemented
        return self.expression == other.expression

    @classmethod
    def readFromFile(cls, stream):
        total = stream.readWord()
        if total[0] == '{':
            while total[-1] != '}':
                total += stream.readWord()
        leaf= cls(total)
        return leaf

    def writeToFile(self, stream, indent=''):
        stream.writeWord(self.sexpression)

class TreeBranch(object):
    def __init__(self, parent_id, default=0, comment=None):
        super(TreeBranch, self).__init__()
        self.parentId = parent_id
        self._questions = {}
        self.default = default
        self.comment =comment

    def __eq__(self, other):
        for attr in ('parentId', '_questions', 'default'):
            if not hasattr(other, attr):
                return NotImplemented
        return (self.parentId == other.parentId) and (self._questions == other._questions) and (self.default == other.default)

    def isLeaf(self):
        return self.parentId == _LEAF

    def vanish(self):
        while True:
            if self.isLeaf():
                return
            if len(self._questions) == 0:
                self.parentId = self.default.parentId
                self._questions = self.default._questions
                self.default = self.default.default
            else:
                break
        for branch in self._questions.values():
            branch.vanish()
        self.default.vanish()

    def __contains__(self, item):
        return item in self._questions

    def numQuestions(self):
        return len(self._questions)

    def __getitem__(self, item):
        if isinstance(item, (long, int)):
            if item in self._questions:
                return self._questions[item]
            else:
                return self.default
        else:
            raise TypeError('Bad index')

    def __setitem__(self, item, value):
        if isinstance(item, (long, int)):
            self._questions[item] = value
        else:
            raise TypeError('Bad index')

    def append(self, value):
        """Adds answer value to question numQuestions()
            WARNING: No checks are made to make sure that question does not already exist.
        """
        self._questions[self.numQuestions()]= value

    def __delitem__(self, item):
        if isinstance(item, (long, int)):
            try:
                del self._questions[item]
            except KeyError:
                raise ValueError("Tree hasn't branch for %r" % item)
        else:
            raise TypeError('Bad index')

    @classmethod
    def readFromFile(cls, stream):
        parent_id = stream.readInt()
        branch = cls(parent_id)
        if parent_id == _LEAF:
            branch.default = TreeLeaf.readFromFile(stream)
            branch.comment = stream.comment
            return branch
        else:
            n_quest = stream.readInt()
            questions = []

            while len(questions)< n_quest-1:
                w = stream.readWord()
                if w == '...':
                    questions.extend(range(questions[-1]+1,stream.readInt()+1))

                else:
                    questions.append(int(w))
            #
            #for x in questions:
            #       print str(x)+"\n"
            question = stream.readWord()
            if question != 'default':
                raise ValueError('Expected string "default", not %r' % question)
            #
            branch.comment = stream.comment
            for q in questions:
                answer = cls.readFromFile(stream)
                branch[q] = answer
            branch.default = cls.readFromFile(stream)
            return branch

    def writeToFile(self, stream, indent=''):
        if indent:
            stream.writeWord(indent)
        stream.writeInt(self.parentId)
        if self.isLeaf():
            self.default.writeToFile(stream)
            if self.comment:
                stream.writelnWord(' % '+self.comment)
            else:
                stream.writeNewLine()
        else:
            stream.writeInt(len(self._questions)+1)
            l = sorted(self._questions)
            if l == range(len(l)):
                stream.writeWord(self.makeRange(len(l)))
            else:
                for q in l:
                    stream.writeWord(q)

            stream.writeWord('default')
            if self.comment:
                stream.writelnWord(' % '+self.comment)
            else:
                stream.writeNewLine()
            for i in l:
                self._questions[i].writeToFile(stream, indent+' ')
            self.default.writeToFile(stream, indent+'       ')

    def makeRange(self,N):
        if N<0:
            raise ValueError("cannot make range into negative numbers")
        elif N == 0:
            return ''
        elif N == 1:
            return '0'
        else:
            return '0 ... '+str(N-1)

class DT(_Object):
    NullTree = TreeBranch(_LEAF, TreeLeaf(0))

    def __init__(self, parent, name, parentCount, tree):
        #self._tree = deepcopy(tree)
        self._tree = tree
        self._parentCount = parentCount
        super(DT, self).__init__(parent, name)

    def __eq__(self, other):
        for attr in ('_parentCount', '_tree'):
            if not hasattr(other, attr):
                return NotImplemented
        return (self._parentCount == other._parentCount) and (self._tree == other._tree)

    def getTree(self):
        return self._tree
    tree = property(getTree)

    def getParentCount(self):
        return self._parentCount
    parentCount = property(getParentCount)

    @classmethod
    def readFromFile(cls, parent, stream, readDTS=True):
        name = stream.readWord()
        w = stream.readWord()
        try:
            parentCount = int(w)
            per_utterance = False
        except ValueError:
            per_utterance = True

        if not per_utterance:
            tree = TreeBranch.readFromFile(stream)
            return cls(parent, name, parentCount, tree)
        else:
            return DTs(parent, w, readDTS)

    def writeToFile(self, stream):
        stream.writelnWord(self.name)
        stream.writelnInt(self.parentCount)
        self.tree.writeToFile(stream)

    def __getitem__(self, item):
        return self.answer(item)

    def answer(self, values):
        if len(values) != self.parentCount:
            raise ValueError('You must supply %d values' % self.parentCount)
        tree = self.tree
        while True:
            if tree.isLeaf():
                return tree.default(values)
            else:
                tree = tree[values[tree.parentId]]

    def __setitem__(self, item, value):
        self.store(item, value)

    def store(self, parents, value):
        if len(parents) != self.parentCount:
            raise ValueError('You must supply %d values' % self.parentCount)

        p_indexes = range(len(parents))

        if not isinstance(value, TreeLeaf):
            value = TreeLeaf(value)
        tree = self.tree
        while True:
            if tree.isLeaf():
                if p_indexes:
                    # Start branching in leaf, create new default branch as
                    # copy of this leaf
                    tree.default = deepcopy(tree)
                    tree.parentId = p_indexes.pop(0)
                    continue
                else:
                    # Overwrite stored value
                    tree.default = value
                    break
            else:
                p_id = tree.parentId
                p_val = parents[p_id]
                if p_val in tree:
                    # Descent in tree
                    tree = tree[p_val]
                    if p_id in p_indexes:
                        p_indexes.remove(p_id)
                else:
                    if p_indexes:
                        # Insert new subtree
                        new_p_id = p_indexes.pop(0)
                        new_default = deepcopy(tree.default)
                        new_tree = TreeBranch(new_p_id, new_default)
                        tree[p_val] = new_tree
                        tree = new_tree
                    else:
                        # Make leaf
                        tree[p_val] = TreeBranch(_LEAF, value)
                        break

    def __delitem__(self, item):
        self.delete(item)

    def delete(self, values):
        if len(values) != self.parentCount:
            raise ValueError('You must supply %d values' % self.parentCount)
        tree = self.tree
        old_tree = None
        while True:
            if tree.isLeaf():
                if old_tree is not None:
                    val = values[old_tree.parentId]
                    del old_tree[val]
                    old_tree.vanish()
                    break
                else:
                    raise ValueError("Cannot delete value in default branch of tree")
            else:
                val = values[tree.parentId]
                if val in tree:
                    tree, old_tree = tree[val], tree
                else:
                    tree = tree[val]
                    old_tree = None


class DTs(_Object):
    def __init__(self, parent, name, readDTS=True):
        gmtk_name = os.path.basename(name).rsplit('.',1)[0]
        super(DTs, self).__init__(parent, gmtk_name)
        self._trees = []
        self._readDTS=readDTS
        self.setDtsFilename(name)

    def __eq__(self, other):
        if not hasattr(other, '_trees'):
            return NotImplemented
        return (self._trees == other._trees)

    def getDtsFilename(self):
        return self._dtsFilename

    def setDtsFilename(self, name):
        self._dtsFilename = name
        if self._readDTS:
            self.readTrees()
    dtsFilename=property(getDtsFilename, setDtsFilename)

    def writeToFile(self, stream):
        stream.writeInt(len(self._trees))
        stream.writelnWord('%Number of DTs')
        for i in range(len(self._trees)):
            stream.writelnInt(i)
            self._trees[i].writeToFile(stream)
            stream.writeNewLine()

    def getTrees(self):
        return self._trees
    trees=property(getTrees)

    def discardTrees(self):
        trees = self.trees
        parent = self.parent
        while trees:
            t = trees.pop()
            del parent[DT, t.name]

    def readTrees(self):
        self.discardTrees()
        trees = self.trees
        io = self.parent.preprocessFile(self.dtsFilename)
        nobj = io.readInt()
        for i in range(nobj):
            ri = io.readInt()
            if i != ri:
                raise ValueError('Invalid object index, read %d, expected %d' % (ri, i))
            trees.append(DT.readFromFile(self.parent, io))
        io.close()


class _PMF(_Object):
    def __init__(self, parent, name, cardinality):
        super(_PMF, self).__init__(parent, name)
        self._initTable(cardinality)

    def _initTable(self, cardinality):
        raise TypeError('Abstract method called')

    def getCardinality(self):
        return len(self)
    cardinality = property(getCardinality)

class DPMF(_PMF, list):
    def _initTable(self, cardinality):
        self[:] = [0] * cardinality

    @classmethod
    def readFromFile(cls, parent, stream):
        name = stream.readWord()
        cardinality = stream.readInt()
        dpmf = cls(parent, name, cardinality)

        for i in range(cardinality):
            dpmf[i] = stream.readFloat()
        return dpmf

    def writeToFile(self, stream):
        stream.writeWord(self.name)
        stream.writeInt(self.cardinality)
        for i in self:
            stream.writeFloat(i)
        #stream.writeNewLine()

    def copy(self,parent,name):
        '''return a copy of this object, with new name and parent'''
        other=self.__class__(parent,name,self.cardinality)
        other[:]=self[:]
        return other

#MEAN is exactly the same as DPMF, only cardinality variable is replaced by dimentionality
class MEAN(_Object, list):
    def __init__(self, parent, name, dimensionality):
        _Object.__init__(self, parent, name)
        self._initTable(dimensionality)

    def _initTable(self, dimensionality):
        self[:] = [0] * dimensionality

    def getDimensionality(self):
        return len(self)
    dimensionality=property(getDimensionality)

    @classmethod
    def readFromFile(cls, parent, stream):
        name = stream.readWord()
        dimensionality = stream.readInt()
        vec = cls(parent, name, dimensionality)

        for i in range(dimensionality):
            vec[i] = stream.readFloat()
        return vec

    def writeToFile(self, stream):
        stream.writeWord(self.name)
        stream.writeInt(self.dimensionality)
        for i in self:
            stream.writeFloat(i)
        #stream.writeNewLine()

    def copy(self,parent,name):
        '''return a copy of this object, with new name and parent'''
        other=self.__class__(parent,name,self.dimensionality)
        other[:]=self[:]
        return other

#diagonal convariances are also the same as MEAN
class COVAR(MEAN):
    pass

#Gaussian Component
class GC(_Object):

    def __init__(self, parent, name, dimensionality, meanName, varName):
        super(GC, self).__init__(parent, name)
        self.dimensionality=dimensionality
        self.meanName=meanName
        self.varName=varName

    @classmethod
    def readFromFile(cls, parent, stream):
        dimensionality = stream.readInt()
        typ = stream.readInt()
        if typ != 0:
            raise ValueError('only GC type 0 is supported, but %d is specified' % typ)
        name = stream.readWord()
        meanName = stream.readWord()
        varName = stream.readWord()
        gc = cls(parent, name, dimensionality, meanName, varName)
        #print gc
        return gc

    def writeToFile(self, stream):
        stream.writeInt(self.dimensionality)
        stream.writeInt(0)
        stream.writeWord(self.name)
        stream.writeWord(self.meanName)
        stream.writelnWord(self.varName)



#Mixture of Gaussians
class MG(_Object):

    def __init__(self, parent, name, dimensionality, weightsDpmfName, gcNames):
        super(MG, self).__init__(parent, name)
        self.dimensionality=dimensionality
        self.weightsDpmfName=weightsDpmfName
        self.gcNames=gcNames

    def copy(self, parent, name):
        """returns a tied copy of self with a new parent and name"""
        return MG(parent, name, self.dimensionality, self.weightsDpmfName, self.gcNames)

    @classmethod
    def readFromFile(cls, parent, stream):
        dimensionality = stream.readInt()
        name = stream.readWord()
        numComponents = stream.readInt()
        weightsDpmfName = stream.readWord()
        gcNames=[]
        for i in range(numComponents):
            gcNames.append(stream.readWord())
        mg = cls(parent, name, dimensionality, weightsDpmfName, gcNames)
        #print mg
        return mg

    def writeToFile(self, stream):
        stream.writeInt(self.dimensionality)
        stream.writeWord(self.name)
        stream.writeInt(len(self.gcNames))
        stream.writelnWord(self.weightsDpmfName)
        for i in range(len(self.gcNames)):
            stream.writeWord(self.gcNames[i])

class DLINK_MAT(_Object):
    def __init__(self, parent, name):
        raise NotImplementedError()

    @classmethod
    def readFromFile(cls, parent, stream):
        raise NotImplementedError()

class WEIGHT_MAT(_Object):
    def __init__(self, parent, name, cardinality):
        raise NotImplementedError()

    @classmethod
    def readFromFile(cls, parent, stream):
        raise NotImplementedError()

class GSMG(_Object):
    def __init__(self, parent, name, cardinality):
        raise NotImplementedError()

    @classmethod
    def readFromFile(cls, parent, stream):
        raise NotImplementedError()

class LSMG(_Object):
    def __init__(self, parent, name, cardinality):
        raise NotImplementedError()

    @classmethod
    def readFromFile(cls, parent, stream):
        raise NotImplementedError()

class MSMG(_Object):
    def __init__(self, parent, name, cardinality):
        raise NotImplementedError()

    @classmethod
    def readFromFile(cls, parent, stream):
        raise NotImplementedError()


class SPMF(_PMF):
    def __init__(self, parent, name, cardinality, dpmfName):
        super(SPMF, self).__init__(parent, name, cardinality)
        self._dpmfName = dpmfName
        self._ptrs = {}

    def __eq__(self, other):
        for attr in ('_dpmfName', '_ptrs'):
            if not hasattr(other, attr):
                return NotImplemented
        return (self._dpmfName == other._dpmfName) and (self._ptrs == other._ptrs)

    def _initTable(self, cardinality):
        self._cardinality = cardinality

    def getDpmf(self):
        return self.parent[DPMF, self.dpmfName]
    dpmf=property(getDpmf)

    def getDpmfName(self):
        return self._dpmfName
    dpmfName=property(getDpmfName)

    def getPtrs(self):
        return self._ptrs
    ptrs=property(getPtrs)

    def __len__(self):
        return self._cardinality

    def __getitem__(self, item):
        dpmf = self.dpmf
        ptrs = self._ptrs
        l = len(self)
        if isinstance(item, (int, long)):
            if item < 0:
                item += l
            if not (0 <= item < l):
                raise IndexError('Index out of range')
            if item in ptrs:
                return dpmf[ptrs[item]]
            else:
                return 0.0
        else:
            raise TypeError('Bad index')

    def __setitem__(self, item, value):
        dpmf = self.dpmf
        ptrs = self._ptrs
        l = len(self)
        if isinstance(item, (int, long)):
            if item < 0:
                item += l
            if not (0 <= item < l):
                raise IndexError('Index out of range')
            if item in ptrs:
                dpmf[ptrs[item]] = value
            else:
                new_index = len(dpmf)
                ptrs[item] = new_index
                dpmf.append(value)
        else:
            raise TypeError('Bad index')

    def __delitem__(self, item):
        dpmf = self.dpmf
        ptrs = self._ptrs
        l = len(self)
        if isinstance(item, (int, long)):
            if item < 0:
                item += l
            if not (0 <= item < l):
                raise IndexError('Index out of range')
            if item in ptrs:
                ref = ptrs[item]
                del ptrs[item]
                del dpmf[ref]
                for key, value in ptrs.items():
                    if value > ref:
                        ptrs[key] = value-1
            else:
                pass
        else:
            raise TypeError('Bad index')

    @classmethod
    def readFromFile(cls, parent, stream):
        name = stream.readWord()
        cardinality = stream.readInt()

        ptrs = {}
        length = stream.readInt()
        for i in range(length):
            ptr = stream.readInt()
            ptrs[ptr] = i

        dpmfName = stream.readWord()
        spmf = cls(parent, name, cardinality, dpmfName)
        spmf.ptrs.update(ptrs)

        return spmf

    def writeToFile(self, stream):
        stream.writelnWord(self.name)
        stream.writelnInt(self.cardinality)

        t = [y[0] for y in sorted(self._ptrs.items(), key=lambda x: x[1])]

        stream.writelnInt(len(t))
        for n in t:
            stream.writeInt(n)
        stream.writeNewLine()
        stream.writelnWord(self.dpmfName)


class _CPT(_Object, ProbTable):
    def __init__(self, parent, name, parent_cards, self_card):
        super(_CPT, self).__init__(parent, name, [self_card], parent_cards)

    def getSelfCard(self):
        cards = self.selfCards
        assert len(cards) == 1
        return cards[0]

    selfCard=property(getSelfCard)

    def getTableAsNumpyArray(self):
        mat = numpy.zeros(self.parentCards + self.selfCards)
        for par_state in self.possibleParents:
            mat[par_state] = self._table[par_state]
        return mat
    array = property(getTableAsNumpyArray)


class DCPT(_CPT, DenseTable):
    def setTableFromNumpyArray(self, nparray):
        #if (nparray.shape[:-1] != self.parentCards or
        #    nparray.shape[-1] != self.selfCard):
        #    raise ValueError('nparray was not of the correct shape. Got %r '
        #                     'expected %r', self.parentCards + self.selfCards,
        #                     nparray.shape)
        for parval in self.getPossibleParents():
            self._table[parval] = list(nparray[parval])

    @classmethod
    def readFromFile(cls, parent, stream):
        name = stream.readWord()
        n_parents = stream.readInt()
        parent_cards = []
        total = 1
        for i in range(n_parents):
            card = stream.readInt()
            parent_cards.append(card)
            total *= card
        self_card = stream.readInt()
        total *= self_card

        dcpt = cls(parent, name, parent_cards, self_card)

        t = dcpt.table
        for key in sorted(t.keys()):
            for i in range(self_card):
                t[key][i] = stream.readFloat()

        return dcpt

    def writeToFile(self, stream):
        stream.writelnWord(self.name)
        stream.writeInt(len(self.parentCards))
        for c in self.parentCards:
            stream.writeInt(c)
        stream.writeNewLine()
        self_card = self.selfCard
        stream.writelnInt(self_card)
        for i, val in enumerate(self.table.flat()):
            if i > 0 and i % self_card == 0:
                stream.writeNewLine()
            stream.writeFloat(val)
        else:
            stream.writeNewLine()

class SCPT(_CPT):
    def __init__(self, parent, name, parent_cards, self_card, dt_name, coll_name):
        super(SCPT, self).__init__(parent, name, parent_cards, self_card)
        self._dtName = dt_name
        self._collName = coll_name

    def __eq__(self, other):
        return (super(SCPT, self).__eq__(other)) and \
               (self._dtName == other._dtName) and \
               (self._collName == other._collName)

    def _initTable(self, scard, pcard):
        return None

    def _eqTables(self, other):
        for attr in ('_dtName', '_collName'):
            if not hasattr(other, attr):
                return NotImplemented
        return (self._dtName == other._dtName) and \
               (self._collName == other._collName)

    def getDtName(self):
        return self._dtName
    dtName = property(getDtName)

    def getDt(self):
        return self.parent[DT, self.dtName]
    dt = property(getDt)

    def getCollName(self):
        return self._collName
    collName = property(getCollName)

    def getColl(self):
        return self.parent[NameCollection, self.collName]
    coll = property(getColl)

    @classmethod
    def readFromFile(cls, parent, stream):
        name = stream.readWord()
        n_parents = stream.readInt()
        parent_cards = []
        for i in range(n_parents):
            card = stream.readInt()
            parent_cards.append(card)
        self_card = stream.readInt()

        dtName = stream.readWord()
        collName = stream.readWord()

        scpt = cls(parent, name, parent_cards, self_card, dtName, collName)

        return scpt

    def writeToFile(self, stream):
        stream.writelnWord(self.name)
        stream.writeInt(len(self.parentCards))
        for c in self.parentCards:
            stream.writeInt(c)
        stream.writeNewLine()
        stream.writelnInt(self.selfCard)
        stream.writelnWord(self.dtName)
        stream.writelnWord(self.collName)

    @classmethod
    def create(cls, parent, name, parent_cards, self_card):
        collection = NameCollection(parent, name)
        collection.append(name+'00000')
        null_dpmf = DPMF(parent, name+'00000', self_card)
        null_spmf = SPMF(parent, name+'00000', self_card, name+'00000')
        dt = DT(parent, name, len(parent_cards), DT.NullTree)
        return cls(parent, name, parent_cards, self_card, name, name)

    def _getMassFunction(self, parents):
        index = self.dt[parents]
        spmf = self.parent[SPMF, self.coll[index]]
        return spmf

    def newMassFunction(self):
        """Create new SPMF (and its DPMF) and register it in collection

        @return:    Tuple (index, spmf), where `index` is index in NameCollection and spmf
                    is created function.
        """
        index = len(self.coll)
        new_name = '%s%05d' % (self.name, index)
        dpmf = DPMF(self.parent, new_name, 0)
        spmf = SPMF(self.parent, new_name, self.selfCard, dpmf.name)
        self.coll.append(new_name)
        return index, spmf

    def _createMassFunction(self, parents):
        tree_value = self.dt[parents]
        if tree_value != 0:
            return self._getMassFunction(parents)
        else:
            index, spmf = self.newMassFunction()
            self.dt[parents] = index
            return spmf

    def _delMassFunction(self, parents):
        del self.dt[parents]

class DetCPT(_CPT):
    def __init__(self, parent, name, parent_cards, self_card, dt_name):
        super(DetCPT, self).__init__(parent, name, parent_cards, self_card)
        self._dtName = dt_name

    def _eqTables(self, other):
        if not hasattr(other, '_dtName'):
            return NotImplemented
        return (self._dtName == other._dtName)

    def getDtName(self):
        return self._dtName
    dtName = property(getDtName)

    def getDt(self):
        return self.parent[DT, self.dtName]
    dt = property(getDt)

    def _initTable(self, scard, pcard):
        return None

    def _getMassFunction(self, parents):
        i = self.dt[parents]
        ret = [0.] * self.selfCard
        if not (0 <= i < self.selfCard):
            raise IndexError("DT %r returns value %d, which is out of range [0, %d]" % (self.dtName, i, self.selfCard-1))
        ret[i] = 1.0
        return ret

    @classmethod
    def readFromFile(cls, parent, stream):
        name = stream.readWord()
        n_parents = stream.readInt()
        parent_cards = []
        for i in range(n_parents):
            card = stream.readInt()
            parent_cards.append(card)
        self_card = stream.readInt()

        dtName = stream.readWord()

        detcpt = cls(parent, name, parent_cards, self_card, dtName)

        return detcpt

    def writeToFile(self, stream):
        stream.writelnWord(self.name)
        stream.writeInt(len(self.parentCards))
        for c in self.parentCards:
            stream.writeInt(c)
        stream.writeNewLine()
        stream.writelnInt(self.selfCard)
        stream.writelnWord(self.dtName)

class FeatureDefinition(set):
    def __init__(self, name, allowedValues):
        self.name = name
        self.update(allowedValues)

    @classmethod
    def readFromFile(cls, io):
        name = io.readWord()
        empty,firstVal= io.readWord().split('(')
        if not empty == '':
            raise ValueError("Cannot have chars preceding '(' in the same word on line %d"%io.line)
        self.add(firstVal)
        endSeen = False
        while not endSeen:
            val = io.readWord()
            try:
                lastVal, leftovers = val.split(')')
            except ValueError:
                self.add(val)
            else:
                endSeen = True
                if lastVal: #last char is )
                    self.add(lastVal)
                if leftovers:
                    raise ValueError("Was expecting a ')' by itself or at the end of a word on line %d"%io.line)
        return cls(name, featureValues)

    def writeToFile(self, io):
        io.writeWord(self.name)
        io.writeWord('(')
        for v in self:
            io.writeWord(v)
        io.writeWord(')')


class FeatureValues(object):
    def __init__(self, name, featureValues):
        self.featureValues=featureValues
        self.name = name

    @classmethod
    def readFromFile(cls, io):
        name = io.readWord()
        #FIXME actually read in the feature definitions to determine how many words to read
        #For now read all words to end of line
        featureValues = io.file.readLine().split()
        return cls(name, featureValues)

    def writeToFile(self, io):
        io.writeWord(self.name)
        for v in self.featureValues:
            io.writeWord(v)

class Question(set):
    def __init__(self, name, feature, values=set()):
        self.name = name
        self.feature =feature
        self.update(values)

    @classmethod
    def readFromFile(cls, io):
        name = io.readWord()
        feature = io.readWord()
        q =Question(name, feature)
        n = io.readInt()
        for i in range(n):
            q.add(io.readWord())
        return q

    def writeToFile(self, io):
        io.writeWord(self.name)
        io.writeWord(self.feature)
        io.writeInt(len(self))
        for v in sorted(self):
            io.writeWord(v)

class NamedObjectCollection(dict):
    "A generic named object collection"
    def __init__(self, obj_type):
        self.obj_type = obj_type #The class of the named objects in this collection

    def readFromIO(self, io):
        nobj = io.readInt()
        for i in range(nobj):
            ri = io.readInt()
            if i != ri:
                raise ValueError('Invalid object index, read %d, expected %d' % (ri, i))
            obj = self.obj_type.readFromFile(io)
            self[obj.name]=obj

    def writeToFile(self, io):
        items = sorted(self.items())
        ostr=NumberedObjectOutStream(io, self.obj_type)
        for (name, obj) in items:
            ostr.writeObject(obj)
        ostr.finalize()



class ObjectOutStream(object):
    def __init__(self, io):
        self._io=io
        self.objCount=0
        self._countBuf = self._io.file.tell()
        self._io.writeWord(' '*20)

    def finalize(self):
        curPos=self._io.file.tell()
        self._io.file.seek(self._countBuf)
        self._io.writeInt(self.objCount)
        self._io.file.seek(curPos)

    def writeComment(self,s):
        self._io.writelnWord(s)


class UnnumberedOutStream(ObjectOutStream):
    def __init__(self, io):
        super(UnnumberedOutStream,self).__init__(io)
        self._io.writelnWord('%  number of objects')
        self._io.writeNewLine()

    def writeObject(self,obj):
        obj.writeToFile(self._io)
        self.objCount +=1

class NumberedObjectOutStream(ObjectOutStream):
    '''An object which writes namedObjects to a stream on the fly,
    without storing them in memory - useful for large object collections'''
    def __init__(self, io, obj_type):
        try:
            obj_type = obj_type.__name__
        except AttributeError:
            pass #objtype is probably some string

        super(NumberedObjectOutStream,self).__init__(io)
        self._io.writelnWord('%  number of '+ "%s objects"  %  obj_type)
        self._io.writeNewLine()

    def writelnString(self,s):
        self._io.writeInt(self.objCount)
        self._io.writelnWord(s)
        self.objCount +=1

    def writeObject(self,obj):
        self._io.writelnInt(self.objCount)
        obj.writeToFile(self._io)
        self._io.writeNewLine()
        self._io.writeNewLine()
        self.objCount +=1

class NamedObjectList(list):
    "A generic named and ordered object list"
    def __init__(self, obj_type):
        self.obj_type = obj_type #The class of the named objects in this collection

    def readFromIO(self, io):
        """This differs from readFromFile because readFromFile is a @classMethod"""
        nobj = io.readInt()
        for i in range(nobj):
            ri = io.readInt()
            if i != ri:
                raise ValueError('Invalid object index, read %d, expected %d' % (ri, i))
            obj = self.obj_type.readFromFile(io)
            self[i]=obj

    def writeToFile(self, io):
        ostr=NumberedObjectOutStream(io, self.obj_type)
        for obj in self:
            ostr.writeObject(obj)
        ostr.finalize()


class FeatureValuesOutStream(NumberedObjectOutStream):
    def __init__(self,io,name, featureDefsName):
        io.writelnWord(name+" %  name of this feature values collection")
        io.writelnWord(featureDefsName+ " % feature definitions name")
        super(FeatureValuesOutStream,self).__init__(io,FeatureValues)


class FeatureValuesCollection(NamedObjectCollection,_Object):
    "A gmtkTie Feature Values collection"
    obj_type = FeatureValues

    def __init__(self, parent, name=None, featureDefsName=None, featureValuesList=[]):
        self.featureDefsName=featureDefsName
        self.update(featureValuesList)
        NamedObjectCollection.__init__(self,self.obj_type)
        _Object.__init__(self,parent, name)

    @classmethod
    def readFromFile(cls, parent, io):
        coll = cls(parent)
        coll.readFromIO(io)
        return coll

    def readFromIO(self, io):
        self.name = io.readWord()
        self.featureDefsName = io.readWord()
        super(FeatureValuesCollection,self).readFromIO(io)

    def getFeatureDefs(self):
        return self.parent[FeatureDefinitionList,self.featureDefsName]
    featureDefs= property(getFeatureDefs)

    def writeToFile(self, io):
        ostr=FeatureValuesOutStream(io, self.name,self.featureDefsName)
        for obj in self:
            ostr.writeObject(obj)
        ostr.finalize()

class QuestionCollection(NamedObjectCollection,_Object):
    "A gmtkTie questions collection"
    obj_type = Question

    def __init__(self, parent, name=None, featureDefsName=None, featureValuesName=None, questions={}):
        self.featureDefsName=featureDefsName
        self.featureValuesName=featureValuesName
        self.update(questions)
        NamedObjectCollection.__init__(self,self.obj_type)
        _Object.__init__(self,parent, name)

    @classmethod
    def readFromFile(cls, parent, io):
        coll = cls(parent)
        coll.readFromIO(io)
        return coll

    def readFromIO(self, io):
        self.name = io.readWord()
        self.featureDefsName = io.readWord()
        self.featureValuesName = io.readWord()
        super(FeatureValuesCollection,self).readFromIO(io)

    def getFeatureDefs(self):
        return self.parent[FeatureDefinitionList,self.featureDefsName]
    featureDefs= property(getFeatureDefs)


    def writeToFile(self, io):
        io.writelnWord(self.name+" %  name of this questions collection")
        io.writelnWord(self.featureDefsName+ " % feature definitions name")
        io.writelnWord(self.featureValuesName+ " % feature values name")
        super(QuestionCollection,self).writeToFile(io)

    def validate(self):
        "make sure all the questions are about existing features, and represent valid feature value sets"
        raise NotImplementedError()

class FeatureDefinitionList(NamedObjectList,_Object):
    "A gmtkTie Feature Definition collection"
    obj_type = FeatureDefinition

    def __init__(self, parent, name=None, featureDefList=[]):
        NamedObjectList.__init__(self,self.obj_type)
        _Object.__init__(self,parent, name)
        self.extend(featureDefList)

    @classmethod
    def readFromFile(cls, parent, io):
        coll = cls(parent)
        coll.readFromIO(io)
        return coll

    def readFromIO(self, io):
        self.name = io.readWord()
        super(FeatureDefinitionList,self).readFromFile(io)


    def writeToFile(self, io):
        io.writelnWord(self.name+" %  name of this feature definitions list")
        super(FeatureDefinitionList,self).writeToFile(io)



class Workspace(object):
    """ contains collections of all known objects, (including collections of some collections, e.g. DTs,
        FeatureValuesCollection and FeatureDefinitionCollection).  FIXME No effort is done to fix name collisions. So
        if two FeatureValues with the same name belong to different FeatureValuesCollection. self[FeatureValues] will have only one of them"""
    knownObjects = {
            'NAME_COLLECTION': NameCollection,
            'DPMF': DPMF,
            'MEAN': MEAN,
            'COVAR': COVAR,
            'GC': GC,
            'MG': MG,
            'SPMF': SPMF,
            'DENSE_CPT': DCPT,
            'SPARSE_CPT': SCPT,
            'DETERMINISTIC_CPT': DetCPT,
            'DT': DT,
            '____DTs': DTs,
            'DLINK_MAT' : DLINK_MAT,
            'WEIGHT_MAT' : WEIGHT_MAT,
            'GSMG' : GSMG,
            'LSMG' : LSMG,
            'MSMG' : MSMG,
            '____FeatureValuesCollection' : FeatureValuesCollection,    #gmtkTie feature values file: cannot be used in a Master file
            '____FeatureDefinitionList' : FeatureDefinitionList,        #gmtkTie feature definitions file: cannot be used in a Master file
            '____QuestionCollection' : QuestionCollection,              #gmtkTie questions file: cannot be used in a Master file
    }

    def __init__(self, cppOptions=None, readDTS=True):
        super(Workspace, self).__init__()
        self._objects = dict([(obj_type,{}) for obj_type in self.knownTypes])
        self._cpp = Preprocessor(cppOptions=cppOptions)
        self._readDTS = readDTS

    def __str__(self):
        print "%d object kinds" % len(self.objects)
        for obj_type in self.objects:
            print "%s obj_type : %d items" % (obj_type, len(self[obj_type]))
            for o in  sorted(self[obj_type].items()) :
                print o

    def getKnownTypes(self):
        return self.knownObjects.values()
    knownTypes = property(getKnownTypes)

    def getObjects(self):
        return self._objects
    objects = property(getObjects)

    def preprocessFile(self, fn):
        return WorkspaceIO(self._cpp.openFile(fn))

    def readMasterFile(self, mstr):
        IN_FILE = '_IN_FILE'
        INLINE = 'inline'
        ASCII = 'ascii'
        mstr_io = self.preprocessFile(mstr)
        #keep a list of open files, in case more than one parameter type is
        #stored in the same file
        file_ios={}
        while True:
            try:
                command = mstr_io.readWord()
            except IOError:
                break
            if not command.endswith(IN_FILE):
                raise ValueError('Invalid master file command: %s' % command)
            type_name = command[:-len(IN_FILE)]
            obj_type = self.knownObjects[type_name]
            fn = mstr_io.readWord()
            if fn == INLINE:
                self.readFromIO(obj_type,mstr_io)
            else:
                format = mstr_io.readWord()
                if format != ASCII:
                    raise ValueError('Format of %r not supported: %s' % (fn, format))
                if fn in file_ios:
                    file_io =file_ios[fn]
                else:
                    file_io = self.preprocessFile(fn)
                    file_ios[fn]=file_io
                self.readFromIO(obj_type,mstr_io)

    def writeMasterFile(self, mstr):
        OUT_FILE = '_OUT_FILE'
        ASCII = 'ascii'
        mstr_io = self.preprocessFile(mstr)
        file_ios={}
        while True:
            try:
                command = mstr_io.readWord()
            except IOError:
                break
            if not command.endswith(OUT_FILE):
                raise ValueError('Invalid master file command: %s' % command)
            type_name = command[:-len(OUT_FILE)]
            obj_type = self.knownObjects[type_name]
            fn = mstr_io.readWord()

            format = mstr_io.readWord()
            if format != ASCII:
                raise ValueError('Format of %r not supported: %s' % (fn, format))

            if fn in file_ios:
                file_io =file_ios[fn]
            else:
                fw = file(fn, 'w')
                file_io = WorkspaceIO(fw)
                file_ios[fn]=file_io

            self.writeToIO(obj_type,file_io)

        for f in file_ios.values():
            f.close()

    def readTrainableParamsFile(self, trainableFile):
        io = self.preprocessFile(trainableFile)
        for typ in [DPMF, SPMF, MEAN, COVAR,DLINK_MAT,WEIGHT_MAT, DCPT,GC, MG, GSMG, LSMG, MSMG, ]:
            self.readFromIO(typ,io)
        io.close()

    def writeTrainableParamsFile(self, trainableFile):
        io = WorkspaceIO(file(trainableFile,'w'))
        self.writeTrainableParamsIO(io)
        io.close()

    def writeTrainableParamsIO(self, io):
        for typ in [DPMF, SPMF, MEAN, COVAR,DLINK_MAT,WEIGHT_MAT, DCPT,GC, MG, GSMG, LSMG, MSMG, ]:
            self.writeToIO(typ,io)

    def readFromFile(self, obj_type, filename):
        f = self.preprocessFile(filename)
        try:
            self.readFromIO(obj_type,f)
        finally:
            f.close()


    def writeToFile(self, obj_type, filename):
        f = WorkspaceIO.withFile(filename, 'w')
        try:
            self.writeToIO(obj_type, f)
        finally:
            f.close()

    def readFromIO(self, obj_type, io):
            nobj = io.readInt()
            for i in range(nobj):
                    ri = io.readInt()
                    if i != ri:
                            raise ValueError('Invalid object index, read %d, expected %d' % (ri, i))
                    if obj_type == DT:
                            obj = obj_type.readFromFile(self, io, readDTS=self._readDTS)
                    else:
                            obj = obj_type.readFromFile(self, io)


    def writeToIO(self, obj_type, io):
            items = sorted(self[obj_type].items())
            io.writelnWord('%  '+ "%s objects"  %  obj_type.__name__)
            io.writeInt(len(items))
            io.writeNewLine()
            io.writeNewLine()
            for i, (name, obj) in enumerate(items):
                    io.writelnInt(i)
                    obj.writeToFile(io)
                    io.writeNewLine()

    def __contains__(self, (obj_type, name)):
        return name in self._objects[obj_type]

    def __getitem__(self, item):
        if not issequence(item):
            item = [item]
        if len(item) == 1:
            return self._objects[item[0]]
        elif len(item) == 2:
            return self._objects[item[0]][item[1]]
        else:
            raise IndexError('Invalid index: %r' % item)

    def __setitem__(self, obj_type, value):
        name = value.name
        if (obj_type, name) in self:
            raise ValueError('There is already %s object %r' % (obj_type.__name__, name))
        self._objects[obj_type][name] = value

    def __delitem__(self, (obj_type, name)):
        obj = self._objects[obj_type][name]
        del self._objects[obj_type][name]
        obj.parent = None

    def getObjLike(self, obj_type, mask):
        objs = self.objects[obj_type]
        ret = []
        for name, obj in objs.iteritems():
            if fnmatchcase(name, mask):
                ret.append(obj)
        return ret

    def delObjLike(self, obj_type, mask):
        objs = self.objects[obj_type]
        to_del = []
        for name, obj in objs.iteritems():
            if fnmatchcase(name, mask):
                to_del.append(name)
        for name in to_del:
            del self[obj_type, name]


class Preprocessor(object):
    def __init__(self, cppOptions=None):
        super(Preprocessor, self).__init__()
        if cppOptions is None:
            cppOptions = []
        self.cppOptions = cppOptions

    def createProcess(self, fn):
        p = subprocess.Popen(['cpp'] + self.cppOptions + ['-P', fn], stdout=subprocess.PIPE)
        return p

    def openFile(self, fn):
        p = self.createProcess(fn)
        return p.stdout


class WorkspaceIO(object):
    def __init__(self, fobj):
        super(WorkspaceIO, self).__init__()
        self._file = fobj
        self._line = 0 #counts lines read through this WorkspaceIO object
        self._stack = []
        self._ws = False
        self._wordsInFile = self.wordsGen()
        '''the comment on the current line in the input file'''
        self.comment=''

    @classmethod
    def withFile(cls, *args, **kwargs):
        f = file(*args, **kwargs)
        return cls(f)

    def getFile(self):
        return self._file
    file=property(getFile)

    @property
    def name(self):
        return self._file.name

    def readWord(self):
        return self._wordsInFile.next()

    def getLine(self):
        return self._line
    line=property(getLine)

    def wordsGen(self):
        for line in self._file:
            self._line += 1
            l = line.split('%', 2)
            line=l[0]
            self.comment = (len(l)>1 and l[1] or '').strip()
            #print "%s, %s" %(line, self.comment)
            for w in line.split():
                yield w
        raise IOError('End of file')

    def readInt(self):
        return int(self.readWord())

    def readFloat(self):
        return float(self.readWord())

    def writeWord(self, w):
        if self._ws:
            self._file.write(' ')
        self._file.write('%s' % w)
        self._ws = True

    def writelnWord(self, w):
        self.writeWord(w)
        self.writeNewLine()

    def writeInt(self, i):
        if self._ws:
            self._file.write(' ')
        self._file.write('%d' % i)
        self._ws = True

    def writelnInt(self, w):
        self.writeInt(w)
        self.writeNewLine()

    def writeFloat(self, f):
        if self._ws:
            self._file.write(' ')
        self._file.write('%.10g' % f)
        self._ws = True

    def writelnFloat(self, w):
        self.writeFloat(w)
        self.writeNewLine()

    def writeNewLine(self):
        self._file.write('\n')
        self._ws = False

    def close(self):
        self._file.close()

#some utility functions
def cartezian(*vectors):
    """Compute Cartesian product of passed arguments
    """
    ret = ret_old = [(v,) for v in vectors[0]]
    for vec in vectors[1:]:
        ret = []
        for v in vec:
            for r in ret_old:
                ret.append(r+(v,))
        ret_old = ret
    return ret

def issequence(obj):
    """Return True if `obj` is sequence, but not string

    @rtype: bool
    """
    if isinstance(obj, StringTypes):
        return False
    else:
        try:
            len(obj)
            return True
        except TypeError:
            return False

def createWorkspaceFromNumpyArrays(np_arrays):
    """Create a new workspace with Dense CPT's loaded from the dict np_arrays"""
    w = Workspace()
    for name, array in np_arrays.items():
        elem = DCPT(w, name, array.shape[:-1], array.shape[-1])
        elem.setTableFromNumpyArray(array)
    return w
