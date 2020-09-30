import numpy as np
import scipy.sparse
import os.path
import sys

# In SciPy 0.7, the sparse eigensolver was scipy.sparse.linalg.eigen.arpack.eigen
# In SciPy 0.8, it is scipy.sparse.linalg.eigen
# In SciPy 0.9, it is scipy.sparse.linalg.eigs OR scipy.sparse.linalg.eigen.arpack.eigs
try:  
  import scipy.sparse.linalg.eigen.arpack as arpack
  try:
      scipy.sparse.linalg.eigen = arpack.eigen  
  except:
      scipy.sparse.linalg.eigen = arpack.eigs      
except:
  import scipy.sparse.linalg.eigen



def tba(traj, clip=True):
    """Take a trajectory of states, and fill in gaps of negative state number by neighboring states
       using Transition Based Assignments.
       """
    traj = np.asarray(traj)
    assert len(traj.shape) == 1, "Trajectory must be a one-dimensional sequence."
    
    if not clip:
        raise NotImplementedError

    if traj.size == 0:
        return traj

    valid = traj>=0
    change_indices = np.where(np.logical_xor(valid[:-1],valid[1:]))[0]

    if not valid[0]:                               # If we start with an invalid state...
        if change_indices.size == 0:               #   ... if we never switch to a valid state, return empty array
            return  np.array([])                  
        else:                                      #   ... otherwise clip beginning of trajectory
            traj = traj[change_indices[0]+1:]
            valid = valid[change_indices[0]+1:]
            change_indices = change_indices[1:]-(change_indices[0]+1)

    if not valid[-1]:                              # If we end with an invalid state...
        assert change_indices.size > 0             #   ... we know from above that at some point it switched from a valid one
        traj = traj[:change_indices[-1]+1]         # Clip ending of trajectory
        valid = valid[:change_indices[-1]+1]
        change_indices = change_indices[:-1]

    assert change_indices.size % 2 == 0                 # we can now assume we have an even number of transition (valid - invalid - valid)

    valid_to_invalid = change_indices[::2]         # determine indices of valid-invalid transitions
    invalid_to_valid = change_indices[1::2]

    for (b,e) in zip(valid_to_invalid, invalid_to_valid):
        if (e-b) % 2 == 0:
            m = b + (e-b)/2
        else:
            m = b + (e-b)/2 + int(round(np.random.random()))
        traj[int(b+1):int(m+1)] = traj[b]
        traj[int(m+1):int(e+1)] = traj[int(e+1)]
    
    return traj


# Run Length Encoding of 1-d arrays: https://gist.github.com/nvictus/66627b580c13068589957d6ab0919e66
#     x=randint(10,size=10)
#     l, v = msm.rlencode(x)
#     np.savez_compressed("blub.npz", lengths=l, values=v)
#     b = np.load("blub.npz")
#     y = msm.rldecode(b['lengths'], b['values'])

def rlencode(x):
    """
    Run length encoding.
    Based on http://stackoverflow.com/a/32681075, which is based on the rle 
    function from R.
    
    Parameters
    ----------
    x : 1D array_like
        Input array to encode
    
    Returns
    -------
    run lengths, run values
    
    """
#    where = np.flatnonzero
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return (np.array([], dtype=int), 
                np.array([], dtype=x.dtype))

    starts = np.r_[0, np.flatnonzero(x[1:] != x[:-1])+1]
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]
    
    return lengths, values

