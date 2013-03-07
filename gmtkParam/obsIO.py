#!/usr/bin/python
# -*- coding: utf-8 -*-
"""IO library for reading/writing GMTK Observation files.

For now there are 
classes for read/writing of the gmtk binary files. 
classes for read/writing of the pfile ascii files 
of form <uttId> <frameId> <featureVector> and they are unrelated
would be nice to combine them to both use  numpy
module it knows nothing about utterances and only   

@author: 	Arthur Kantor
@contact: 	akantorREMOVE_THIS@uiuc.edu
@copyright: 2009
@license: 	GPL version 3
@date: 		3/27/2009
@version: 	0.1

The tests follow

>>> a= numpy.array([[1,2,3],[4,5,6]])
>>> a
array([[1, 2, 3],
       [4, 5, 6]])

>>> o=GmtkBinaryObsOutStream('obsIOunitTest.bin', 3, 0, True)
>>> o.writeFrames(a)
>>> o.close()
>>> i=GmtkBinaryObsInStream('obsIOunitTest.bin', 3, 0, True)
>>> b=i.readFrames()
>>> i.close()
>>> b
array([[1, 2, 3],
       [4, 5, 6]])

>>> numpy.alltrue(b == a)
True


"""
import os.path
from ctypes import c_int, c_float, sizeof
import array
import numpy

class GmtkBinaryStream(object):
	def __init__(self, f, nint, nfloat, bswap):
		self.nint=nint
		self.nfloat=nfloat
		self._bswap=bswap
		self._f= f
		if self.nint>0 and self.nfloat>0:
			raise ValueException("nint and nfloat cannot both be greater than 0")
		if(self.nint>0):
			intSiz=sizeof(c_int)
			self._arrayType='i'
			self._stride=intSiz*self.nint
			self._cols=self.nint
		else:
			floatSiz=sizeof(c_float)
			self._arrayType='f'
			self._stride=floatSiz*self.nfloat
			self._cols=self.nfloat

	def close(self):
		self._f.close()	

	
class GmtkBinaryObsInStream(GmtkBinaryStream):
	def __init__(self, fname, nint, nfloat, bswap):
		super(GmtkBinaryObsInStream, self).__init__(file(fname), nint, nfloat, bswap)
		self._frame=0 #the  frame to be read next
		bytes =os.path.getsize(fname)
		self._totalFrames=bytes/(self._stride)
	
	def readFrames(self,count=None):
		'''@param count: The number of frames is None, read to the end of file.
		   @return a numpy 2d array of size countX(nint+nfloat)
		   
		   floats and longs are assumed to be whatever c would use on the given system'''
		if count==None:
			count = self._totalFrames - self._frame 
			
		a=array.array(self._arrayType)
		a.fromfile(self._f,count*self._cols)
		if self._bswap:
			a.byteswap()
		na=numpy.array(a)
		na.shape=count,self._cols
		self._frame += count
		return na

class GmtkBinaryObsOutStream(GmtkBinaryStream):
	def __init__(self, fname, nint, nfloat, bswap):
		'''open fname for writing, clearing it out if it exists'''
		super(GmtkBinaryObsOutStream, self).__init__(file(fname,'w'), nint, nfloat, bswap)
		self._frame=0 #the  frame to be read next
	
	def writeFrames(self,arr):
		'''@param arr: Append arr to file.'''
		   
		if arr.shape[1] != self._cols:
			raise ValueError("stream expects, %d columns, but arr has %d columns",(self._cols,arr.shape[1]))

		cnt=arr.shape[0]*arr.shape[1]
		b=arr.reshape(cnt,)
		a=array.array(self._arrayType,b.tolist())
		if self._bswap:
			a.byteswap()
		a.write(self._f)
		self._frame += arr.shape[0]
	

class ObsSink(object):
	'''
	writes Observations to a file.  The provided 
	uttIds are ignored and new uttIds are assigned sequentially

	>>> of = open('asciiObs.txt','w')
	>>> obsSink = ObsSink(of)
	>>> uttObs1=[('0','0', '.1 -.1 .01'),('0','1', '.2 -.2 .02')]
	>>> uttObs2=[('333','0', '.3 -.3 .03'),('333','1', '.4 -.4 .04')]
	>>> obsSink.writeObs(uttObs1)
	>>> obsSink.writeObs(uttObs2)
	>>> of.close()

	>>> ifile = open('asciiObs.txt')
	>>> oSource = obsSource(ifile)
	>>> uttObs3= oSource.next()
	>>> uttObs4= oSource.next()
	>>> ifile.close()

	>>> uttObs1 == uttObs3
	True

	>>> [('1','0', '.3 -.3 .03'),('1','1', '.4 -.4 .04')] == uttObs4
	True

	'''
	def __init__(self,ofile):
		self._f = ofile
		self.uttCount=0
		
	def writeObs(self, obs):
		s=str(self.uttCount)
		for frame in obs:
			self._f.write(' '.join((s, frame[1],frame[2])))
			self._f.write('\n')
		
		self.uttCount += 1
			
def obsSource(obsF):			
	l = obsF.next()
	frame= tuple(x for x in l.strip().split(None, 2))
	curUtt= [frame]
	prevUttId=int(frame[0])
	for l in obsF:
		frame= tuple(x for x in l.strip().split(None, 2))
		if(int(frame[0]) == prevUttId):
			curUtt.append(frame)
		else:
			yield curUtt
			curUtt= [frame]
			prevUttId=int(frame[0])
		
	yield curUtt

if __name__ == "__main__":
	import doctest
	doctest.testmod()

