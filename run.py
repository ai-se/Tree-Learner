#! /Users/rkrsn/miniconda/bin/python
from __future__ import print_function, division

from os import environ, getcwd
from os import walk
from pdb import set_trace
import sys
from bdb import set_trace

# Update PYTHONPATH
HOME = environ['HOME']
axe = HOME + '/git/axe/axe/'  # AXE
pystat = HOME + '/git/pystats/'  # PySTAT
cwd = getcwd()  # Current Directory
sys.path.extend([axe, pystat, cwd])

from Planning import *
from Prediction import *
from _imports import *
from abcd import _Abcd
from cliffsDelta import cliffs
from dEvol import tuner
from demos import cmd
from methods1 import *
from sk import rdivDemo
import numpy as np
import pandas as pd
import csv
from numpy import sum


class run():

  def __init__(
          self,
          pred=rforest,
          _smoteit=True,
          _n=-1,
          _tuneit=False,
          dataName=None,
          reps=1,
          extent=0.5,
          fSelect=False,
          Prune=False,
          infoPrune=0.75):
    self.pred = pred
    self.dataName = dataName
    self.out, self.out_pred = [self.dataName], []
    self._smoteit = _smoteit
    self.train, self.test = self.categorize()
    self.reps = reps
    self._n = _n
    self.tunedParams = None if not _tuneit else tuner(
        self.pred, self.train[_n])

  def categorize(self):
    dir = './Data'
    self.projects = [Name for _, Name, __ in walk(dir)][0]
    self.numData = len(self.projects)  # Number of data
    one, two = explore(dir)
    data = [one[i] + two[i] for i in xrange(len(one))]

    def withinClass(data):
      N = len(data)
      return [(data[:n], [data[n]]) for n in range(1, N)]

    def whereis():
      for indx, name in enumerate(self.projects):
        if name == self.dataName:
          return indx

    return [
        dat[0] for dat in withinClass(data[whereis()])], [
        dat[1] for dat in withinClass(data[whereis()])]  # Train, Test

  def go(self):

    for _ in xrange(self.reps):
      predRows = []
      train_DF = createTbl(self.train[self._n], isBin=True, bugThres=1)
      test_df = createTbl(self.test[self._n], isBin=True, bugThres=1)
      actual = Bugs(test_df)
      before = self.pred(train_DF, test_df,
                         tunings=self.tunedParams,
                         smoteit=True)

      for predicted, row in zip(before, test_df._rows):
        tmp = row.cells
        tmp[-2] = predicted
        if predicted > 0:
          predRows.append(tmp)

      predTest = clone(test_df, rows=predRows)
      newTab = treatments(
          train=self.train[
              self._n], test_DF=predTest).main()

      after = self.pred(train_DF, newTab,
                        tunings=self.tunedParams,
                        smoteit=True)

      self.out_pred.append(_Abcd(before=actual, after=before))
      delta = cliffs(lst2=Bugs(predTest), lst1=after).delta()
      frac = sum(after) / sum(before)
      self.out.append(frac)
    print(self.out)

  def deltas():
    orig = createTbl(self.test[self._n], isBin=True, bugThres=1)
    rows = 
  
def _test(file='ant'):
  for file in ['ivy', 'lucene', 'poi', 'jedit', 'ant']:
    print('##', file)
    R = run(dataName=file, reps=10).go()


def rdiv():
  lst = []

  def striplines(line):
    listedline = line.strip().split(',')  # split around the = sign
    listedline[0] = listedline[0][2:-1]
    lists = [listedline[0]]
    for ll in listedline[1:-1]:
      lists.append(float(ll))
    return lists

  f = open('./data.txt')
  for line in f:
    lst.append(striplines(line[:-1]))

  rdivDemo(lst, isLatex='True')
  set_trace()


if __name__ == '__main__':
  #   _test(file='ant')
  rdiv()
#   eval(cmd())
