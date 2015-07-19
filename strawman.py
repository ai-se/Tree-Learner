from numpy import array, mean, median


class exemplar():

  def __init__(self, rows):
    self.rows = array([])
    for r in rows:
      self.rows.append(array(r.cells))

  def centroid(self):
    return median(self.rows)
  
class contrast():
  def __init__(self):
    pass  
  def delta(self, test):
    
    