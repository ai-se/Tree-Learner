from numpy import array, mean, median
from run import run
from pdb import set_trace
from methods1 import createTbl


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
      return median(array(self.rows), axis=1)
    elif what == 'mean':
      return mean(array(self.rows), axis=1)


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
  ""


class strawman():

  def __init__(self):
    self.dir = './Jureczko'
    pass

  def delta(self, test):
    pass

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
    test_df = createTbl(test[-1], isBin=False)._rows
    clusters = [c for c in self.nodes(train_DF)]
    C = contrast(clusters)
    for t in test_df:
      closest = C.envy(t)

      # -------- DEBUG --------
      set_trace()

if __name__ == '__main__':
  strawman().main()
