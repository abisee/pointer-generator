"""
Simple script that checks if a checkpoint is corrupted with any inf/NaN values. Run like this:
  python inspect_checkpoint.py model.12345
"""

import tensorflow as tf
import sys
import numpy as np


if __name__ == '__main__':
  if len(sys.argv) != 2:
    raise Exception("Usage: python inspect_checkpoint.py <file_name>\nNote: Do not include the .data .index or .meta part of the model checkpoint in file_name.")
  file_name = sys.argv[1]
  reader = tf.train.NewCheckpointReader(file_name)
  var_to_shape_map = reader.get_variable_to_shape_map()

  finite = []
  all_infnan = []
  some_infnan = []

  for key in sorted(var_to_shape_map.keys()):
    tensor = reader.get_tensor(key)
    if np.all(np.isfinite(tensor)):
      finite.append(key)
    else:
      if not np.any(np.isfinite(tensor)):
        all_infnan.append(key)
      else:
        some_infnan.append(key)

  print "\nFINITE VARIABLES:"
  for key in finite: print key

  print "\nVARIABLES THAT ARE ALL INF/NAN:"
  for key in all_infnan: print key

  print "\nVARIABLES THAT CONTAIN SOME FINITE, SOME INF/NAN VALUES:"
  for key in some_infnan: print key

  print ""
  if not all_infnan and not some_infnan:
    print "CHECK PASSED: checkpoint contains no inf/NaN values"
  else:
    print "CHECK FAILED: checkpoint contains some inf/NaN values"
