from functools import wraps
from time import time

def organization_activity2wz08section(label : str):
    return group2section(label.split('.')[0])

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

def group2section(label : str):

  labelz = int(label)
  if labelz == 0:
      label="[UNKNOWN]"
  elif labelz <= 3:
      label = "A"
  elif labelz <= 9:
      label = "B"
  elif labelz <= 33:
      label = "C"
  elif labelz <= 35:
      label = "D"
  elif labelz <= 39:
      label = "E"
  elif labelz <= 43:
      label = "F"
  elif labelz <= 47:
      label = "G"
  elif labelz <= 53:
      label = "H"
  elif labelz <= 56:
      label = "I"
  elif labelz <= 63:
      label = "J"
  elif labelz <= 66:
      label = "K"
  elif labelz <= 68:
      label = "L"
  elif labelz <= 75:
      label = "M"
  elif labelz <= 82:
      label = "N"
  elif labelz <= 84:
      label = "O"
  elif labelz <= 85:
      label = "P"
  elif labelz <= 88:
      label = "Q"
  elif labelz <= 93:
      label = "R"
  elif labelz <= 96:
      label = "S"
  elif labelz <= 98:
      label = "T"
  elif labelz <= 99:
      label = "U"

  return label