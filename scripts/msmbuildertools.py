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

def save_sparse_matrix(filename, m):
    """Saves sparse matrix m to specified file."""

    m = m.tocsr()
    np.savez(filename, shape=m.shape, data = m.data, indices=m.indices, indptr=m.indptr)

def load_sparse_matrix(filename):
    """Loads sparse matrix as CSR file."""

    content = np.load(filename)
    return scipy.sparse.csr_matrix((content["data"], content["indices"], content["indptr"]), shape=content["shape"])

def save_matrix(filename, m):
    """Saves (sparse or dense) matrix m to file of specified name."""

    if scipy.sparse.isspmatrix(m):
        save_sparse_matrix(filename, m)
    else:
        np.save(filename, m)

def load_matrix(filename):
    """Loads (sparse or dense) matrix from file."""

    content = np.load(filename)

    if isinstance(content, np.ndarray):
        return content

    try:
        return scipy.sparse.csr_matrix((content["data"], content["indices"], content["indptr"]), shape=content["shape"])
    except:
        raise IOError("Unable to load file " + filename + " as a matrix.")

def parse_integer_specifier(spec):
    """Parses a string that specifies a sequence of integer numbers, and returns that sequence.

    Examples:
    ---------

    parse_integer_specifier("1,5:8,10:20:5") = [1, 5, 6, 7, 8, 10, 15, 20]    
    """
    
    result = []
    for s in spec.split(","):
        u = s.split(":")
        if len(u) == 1:
            result.append(int(u[0]))
        elif len(u) == 2:
            result.extend(list(range(int(u[0]),int(u[1])+1)))
        elif len(u) == 3:
            result.extend(list(range(int(u[0]),int(u[1])+1,int(u[2]))))
    return result

def read_generator(filename):
    """Reads a MSMBuilder generator file."""
    
    with open(filename) as f:
        # skip the first 3 lines
        f.readline()
        f.readline()
        f.readline()

        # read number of atoms
        natoms = int(f.readline())        

        # read list of (x1,x2,x3,...,y1,y2,y3,...,z1,z2,z3...) atom coordinates,
        # reshape it into a (N,3) array, and create a copy to turn it into
        # a C-contiguous array
        return np.loadtxt(f, dtype=np.single).reshape((natoms,3), order="F").copy()

def iterloadtxt(filenames, progressbar="auto", **kwargs):
    """Generator to iterate over a sequence of arrays stored in files.

    Example
    -------
    for a in iterloadtxt(["file1.npy", "file2.npy"]):
        print a.shape
    
    Parameters
    ----------
    filenames : string or array
        A single file name or a sequence of file names.

    progressbar : {True, False, "auto"}
        If True, a progress bar will be displayed to give a visual indication
        of the iteration process. If "auto", then the progress bar will be
        displayed if stderr is a terminal and the list of filenames contains
        more than one element.

    **kwargs : keyword arguments
        All remaining keyword arguments are passed through to numpy.loadtxt

    Returns
    -------
    Iterator over numpy arrays.
"""
    
    # in only a single file name is given, turn it into a sequence of file names of length 1
    if isinstance(filenames, str):
        filenames = (filenames,)
        
    if progressbar == "auto":
        try:
            progressbar = sys.stderr.isatty() and len(filenames) > 1
        except:
            progressbar = False
    assert type(progressbar) == bool, "Internal error: progressbar is of type " + str(type(progressbar)) + " and has value " + str(progressbar)

    if progressbar:
        try:
            pbar = pb.ProgressBar(maxval=len(filenames)).start()
        except:
            progressbar = False
    try:
        for i in range(len(filenames)):
            yield np.loadtxt(filenames[i], **kwargs)
            if progressbar:
                pbar.update(i)
    finally:
        if progressbar:
            pbar.finish()

def assignmentfilenames(trajfilename="trajlist", assigndir="assignments/"):
    """Returns list of filenames of assignment files.

    Parameters
    ----------

    trajfilename : string
        Name of the traj file.

    assigndir : string
        Name of the directory in which the assignment files live.
        
    Returns
    -------
    List of filenames of assignment files
    """
    
    with open(trajfilename) as tf:
        filenames = tf.readlines()
    filenames = [os.path.join(assigndir, f.strip()) for f in filenames]

    return filenames

def iterassignments(trajfilename="trajlist", assigndir="assignments/", column=0, progressbar="auto", dtype=np.int):
    """Generator to iterate over a sequence of assignment files.

    Parameters
    ----------

    trajfilename : string
        Name of the traj file.

    assigndir : string
        Name of the directory in which the assignment files live.

    column : integer
        Which column of the assignment files to use.

    progressbar : {True, False, "auto"}
        Specifies whether a progress bar will be displayed.

    dtype : data type
        The data type of the column to be read.
        
    Returns
    -------
    Iterator over assignment arrays.
    """
    
    with open(trajfilename) as tf:
        filenames = tf.readlines()
    filenames = [os.path.join(assigndir, f.strip()) for f in filenames]

    return iterloadtxt(filenames, progressbar, dtype=dtype, usecols=(column,))

def get_transition_count_matrix_sparse(states, numstates = None, lagtime = 1, slidingwindow = True):
  """Computes the transition count matrix for a sequence of states.

  Parameters
  ----------
  states : array
      A one-dimensional array of integers representing the sequence of states.
      These integers must be in the range [0, numstates[

  numstates : integer
      The total number of states. If not specified, the largest integer in the
      states array plus one will be used.

  lagtime : integer
      The time delay over which transitions are counted

  slidingwindow : bool

  Returns
  -------
  C : sparse matrix of integers
      The computed transition count matrix. C[i,j] is the number of observed
      transitions from i to j.
  """
  
  if not isinstance(states, np.ndarray):
      states = np.array(states)

  if not numstates:
    numstates = np.max(states)+1

  if slidingwindow:
    from_states = states[:-lagtime:1]
    to_states = states[lagtime::1]
  else:
    from_states = states[:-lagtime:lagtime]
    to_states = states[lagtime::lagtime]
  assert from_states.shape == to_states.shape

  transitions = np.row_stack((from_states,to_states))
  counts = np.ones(transitions.shape[1], dtype=int)
        
  try:
    C = scipy.sparse.coo_matrix((counts, transitions),shape=(numstates,numstates))
  except ValueError:
    # Lutz: if we arrive here, there was probably a state with index -1
    # we try to fix it by ignoring transitions in and out of those states
    # (we set both the count and the indices for those transitions to 0)
    mask = transitions < 0
    counts[mask[0,:] | mask[1,:]] = 0
    transitions[mask] = 0
    C = scipy.sparse.coo_matrix((counts, transitions),shape=(numstates,numstates))
            
  return C

def get_transition_count_matrix_dense(states, numstates = None, lagtime = 1, slidingwindow = True):
  """Computes the transition count matrix for a sequence of states.

  Parameters
  ----------
  states : array
      A one-dimensional array of integers representing the sequence of states.
      These integers must be in the range [0, numstates[

  numstates : integer
      The total number of states. If not specified, the largest integer in the
      states array plus one will be used.

  lagtime : integer
      The time delay over which transitions are counted

  slidingwindow : bool

  Returns
  -------
  C : two-dimensional ndarray
      The computed transition count matrix. C[i,j] is the number of observed
      transitions from i to j.
  """

  if not isinstance(states, np.ndarray):
      states = np.array(states)
      
  if not numstates:
    numstates = np.max(states)+1
  else:
    if np.max(states) >= numstates:
        raise RuntimeError("Encountered state " + str(np.max(states)) + ", should be less than " + str(numstates) + ".")
    
  if slidingwindow:
    from_states = states[:-lagtime:1]
    to_states = states[lagtime::1]
  else:
    from_states = states[:-lagtime:lagtime]
    to_states = states[lagtime::lagtime]
  assert from_states.shape == to_states.shape

  try:
    C = np.bincount(numstates * from_states + to_states)
    C.resize((numstates,numstates), refcheck=False)
  except ValueError:
    # Lutz: if we arrive here, there was probably a state with index -1
    # We can't fix that and still use bincount, because it throws an
    # exception for negative values even if their weight is 0.
    # Instead, we construct a sparse matrix first, which is still much
    # faster than manually looping over the arrays (but it is memory intensive).
    transitions = np.row_stack((from_states,to_states))
    counts = np.ones(transitions.shape[1], dtype=int)
    mask = transitions < 0
    counts[mask[0,:] | mask[1,:]] = 0
    transitions[mask] = 0
    C = scipy.sparse.coo_matrix((counts, transitions),shape=(numstates,numstates)).todense()
  
  return C

def estimate_transition_matrix(tCount, make_symmetric = True):
    """Naive Maximum Likelihood estimator of transition matrix.

    Parameters
    ----------
    tCount : array / sparse matrix
        A square matrix of transition counts

    make_symmetric : bool
        If true, make transition count matrix symmetric

    Returns
    -------
    tProb : array / sparse matrix
        Estimate of transition probability matrix

    Notes
    -----
    The transition count matrix will not be altered by this function. Its elemnts can
    be either of integer of floating point type.
    """

    # Lutz: We have be a little careful here, because:
    # 1. The transition count matrix might be integer of floating point.
    #    We convert explicitly to floating point.
    # 2. We do not want to alter the tCount object. When generating the
    #    tProb array, we have to copy data at least once, but we don't
    #    want to do it more often than that. When symmetrizing the matrix,
    #    the addition creates a new object. But when not, we have to make
    #    sure that tProb is decoupled from tCount.

    if scipy.sparse.isspmatrix(tCount):
        if make_symmetric:
            tProb = (tCount + tCount.transpose()).tocsr().asfptype()
        else:
            tProb = tCount.tocsr().asfptype()
            if tProb is tCount:           # if tProb is still the same object as the function parameter tCount (even after CSR and FP conversion),
                tProb = tCount.copy()     # make a copy so that we don't change the tCount object in the calling code
        weights = np.asarray(tProb.sum(axis=1)).flatten()            
        weights[weights==0] = 1.
            
        # now we know that tProb is a floating point CSR matrix, and weights is an array of floating point numbers

        for i in range(tProb.shape[0]):                                          # for every row:
            tProb.data[tProb.indptr[i]:tProb.indptr[i+1]] /= weights[i]           #   divide each matrix element by the weight of the row

    else:
        if make_symmetric:
            tProb = np.asfarray(tCount + tCount.transpose())
        else:
            tProb = np.asarray(tCount.astype(float))                              # astype creates a copy, so tProb is decoupled from tCount
        weights = tProb.sum(axis=1)
        weights[weights==0] = 1.

        tProb = tProb / weights.reshape((weights.shape[0],1))
        
    return tProb

def eigensystem(tprob, k = None, **kwargs):
    """Returns the k largest (by magnitude) eigenvalues and left eigenvectors. Additional arguments passed to the (sparse or dense) eigensolver.
    """

    if k is None:
        k = tprob.shape[0]

    if scipy.sparse.isspmatrix(tprob):
        # test if transition probability matrix has empty rows (for CSR matrices, this causes consecutive equal entries in the indptr array)
        tprob = tprob.tocsr()
        if np.any(tprob.indptr[:-1] == tprob.indptr[1:]):
            print("Warning: transition probability matrix countains rows with all zeros.", file=sys.stderr)                    
        # compute the largest (by magnitude) eigenvalues
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigen(tprob.transpose(), k, **kwargs)
    else:
        # test if transition probability matrix has empty rows
        if np.any(np.alltrue(tprob == 0,axis=-1)):
            print("Warning: transition probability matrix countains rows with all zeros.", file=sys.stderr)
        # compute all eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(tprob.transpose(), **kwargs)

    # now we have the eigenvalues and left eigenvectors
    # if the matrix was dense, then we have to sort the eigenvalues by descreasing magnitude
    # if the matrix was sparse, then the eigenvalues should already by sorted, but sometimes they are not
    eigenvalues = np.real_if_close(eigenvalues)
    sortorder = np.abs(eigenvalues).argsort()[-1:-1-k:-1]
    eigenvalues = eigenvalues[sortorder]
    eigenvectors = np.real_if_close(eigenvectors[:,sortorder])

    return eigenvalues, eigenvectors

def corr(seq1, seq2 = None):
    """Computes the correlation function for two sequences, for example time series.

    For two sequences s1, s2 of the same length N, it computes

           c[t] = sum_{i=0}^{i+t<N} s1[i]*s2[i+t] / sum_{i=0}^{i+t<N} 1

    If only one sequence is specified, it computes the autocorrelation function
    
       c[t] = sum_{i=0}^{i+t<N} s[i]*s[i+t] / sum_{i=0}^{i+t<N} 1
    """
    
    N = len(seq1)
    if seq2 == None:
        seq2 = seq1
    else:
        assert(len(seq2) == N)

    r=np.correlate(seq1,seq2,mode="full")
    r=r[r.size/2:] / np.arange(N,0,-1,dtype=float)    # take only the positive half of the function (it's symmetric), and normalize
    return r


def corr2(seq1, seq2 = None, times=None, slidingwindow = True):
    """Same as corr, but hand-coded, and with the option to turn off sliding windows."""
   
    N = len(seq1)
    if seq2 == None:
        seq2 = seq1
    else:
        assert(len(seq2) == N)

    if times == None:
        times = range(N)

    erg = np.zeros(len(times))

    if slidingwindow:
        dt = 1        
        for i,t in enumerate(times):
            if t == 0:
                erg[i] = np.mean(seq1 * seq2)
            else:
                erg[i] = np.mean(seq1[:-t] * seq2[t:])
    else:
        for i,t in enumerate(times):
            if t == 0:
                erg[i] = np.mean(seq1 * seq2)
            else:
                erg[i] = np.mean(seq1[:-t:t] * seq2[t::t])

    return erg

def lifetimes(seq, lifetimes = None, clip=True):

    if not clip:
      raise NotImplementedError

    if lifetimes == None:
        lifetimes = {}

    start = 0;
    i = 1
    while i < len(seq):
        if seq[i] != seq[i-1]:
            if start > 0:
                try:
                    lifetimes[seq[i-1]].append(i - start)
                except KeyError:
                    lifetimes[seq[i-1]] = [i - start]
            start = i
        i += 1
    return lifetimes


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

def rldecode(lengths, values, minlength=None):
    """
    Decode a run-length encoding of a 1D array.
    
    Parameters
    ----------
    starts, lengths, values : 1D array_like
        The run-length encoding.
    minlength : int, optional
        Minimum length of the output array.
    
    Returns
    -------
    1D array. Missing data will be filled with NaNs.
    
    """
    lengths, values = list(map(np.asarray, (lengths, values)))
    starts = np.r_[0, np.cumsum(lengths[:-1])]
    # TODO: check validity of rle
    ends = starts + lengths
    n = ends[-1]
    if minlength is not None:
        n = max(minlength, n)
    x = np.empty(n, dtype=values.dtype)
    for lo, hi, val in zip(starts, ends, values):
        x[lo:hi] = val
    return x
