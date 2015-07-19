from numpy import array, mean, median
from run import run
from pdb import set_trace
from methods1 import createTbl


class node():

  def __init__(self, rows):
    self.rows = array([])
    for r in rows:
      self.rows.append(r.cells)

  def centroid(self):
    return median(self.rows)


class contrast():

  def __init__(self):
    self.dir = './Jureczko'
    pass

  def delta(self, test):
    pass

  def nodes(self, rowObject):
    clusters = set([r.cells[-1] for r in rowObject])
    for row in rowObject:
      cluster = []
      for id in clusters:
        if row.cells[-1] == id:
          cluster.append(row)
      yield node(cluster)

  def main(self):
    train, test = run(dataName='ant').categorize()
    train_DF = createTbl(train[-1], isBin=False)._rows
    test_df = createTbl(test[-1], isBin=False)._rows
    clusters = [c for c in self.nodes(train_DF)]
    # -------- DEBUG --------
    set_trace()

if __name__ == '__main__':
  contrast().main()
