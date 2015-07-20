import csv
from numpy import array, mean, median, percentile
from run import run
from pdb import set_trace
from methods1 import createTbl
from Prediction import rforest


def eDist(row1, row2):
  "Euclidean Distance"
  return sum([(a * a - b * b)**0.5 for a, b in zip(row1[:-1], row2[:-1])])

class node():
  """
  A data structure to hold all the rows in a cluster.
  Also return an exemplar: the centroid.
  """

  def __init__(self, rows):
    self.rows = []
    for r in rows:
      self.rows.append(r.cells[:-1])

  def exemplar(self, what='centroid'):
    if what == 'centroid':
      return median(array(self.rows), axis=0)
    elif what == 'mean':
      return mean(array(self.rows), axis=0)


class contrast():
  "Identify the nearest enviable node."

  def __init__(self, clusters):
    self.clusters = clusters

  def closest(self, testCase):
    return sorted([f for f in self.clusters],
                  key=lambda F: eDist(F.exemplar(), testCase.cells[:-1]))[0]

  def envy(self, testCase, alpha=0.5):
    me = self.closest(testCase)
    others = [o for o in self.clusters if not me == o]
    betters = [f for f in others if f.exemplar()[-1] <= me.exemplar()[-1]]
    return sorted([f for f in betters],
                  key=lambda F: eDist(F.exemplar(), me.exemplar()))[0]


class patches():
  "Apply new patch."

  def __init__(self, test, clusters):
    self.train = createTbl(train, isBin=True) 
    self.test = createTbl(test, isBin=True)
    self.pred = rforest(train, test, smoteit=True, duplicate=True)
    self.clusters = clusters

  def delta(self, node1, node2):
    return [el1 - el2 for el1,
            el2 in zip(node1.exemplar()[:-1], node2.exemplar()[:-1])]

  def delta(self, t):
    C = contrast(self.clusters)
    closest = C.closest(t)
    better = C.envy(t, alpha=1)
    return array(self.delta(closest, better))

  def newRow(self, t):
    return (array(t.cells[:-2]) + self.delta(t)).tolist()

  def newTable(self):
    oldRows = [r for r, p in zip(self.test._rows, self.pred) if p>0]
    newRows = [self.newRow(t) for t in oldRows]
    return clone(self.test, rows=newRows)

  def deltas(self, name='ant'):
    "Changes"
    header = array([h.name[1:] for h in self.test.headers[:-2]])
    oldRows = [r for r, p in zip(self.test._rows, self.pred) if p>0]
    delta = array([self.delta(t) for t in oldRows])
    y = median(delta, axis=0)
    yhi, ylo = percentile(delta, q=[75, 25], axis=0)
    dat1 = sorted([(h, a, b, c) for h, a, b, c in zip(header, y, ylo, yhi)]
                  , key=lambda F: F[1])
    dat = np.asarray([(d[0], n, d[1], d[2], d[3])
                      for d, n in zip(dat1, range(1, 21))])
    with open('/Users/rkrsn/git/GNU-Plots/rkrsn/errorbar/%s.csv' % (name), 'w') as csvfile:
      writer = csv.writer(csvfile, delimiter=' ')
      for el in dat[()]:
        writer.writerow(el)
    # new = [self.newRow(t) for t in oldRows]


class strawman():

  def __init__(self, name = "ant"):
    self.dir = './Jureczko'
    self.name = name

  def nodes(self, rowObject):
    clusters = set([r.cells[-1] for r in rowObject])
    for id in clusters:
      cluster = []
      for row in rowObject:
        if row.cells[-1] == id:
          cluster.append(row)
      yield node(cluster)

  def main(self):
    train, test = run(dataName='ant').categorize()
    train_DF = createTbl(train[-1], isBin=False)._rows
    clusters = [c for c in self.nodes(train_DF)]
    newTbl = patches(train[-1], test[-1], clusters).newTable()
    # -------- DEBUG --------
    set_trace()

if __name__ == '__main__':
  strawman().main()
