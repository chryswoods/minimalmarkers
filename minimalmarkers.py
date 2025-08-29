import os as _os

try:
    from dataclasses import dataclass as _dataclass
    from typing import List as _List
    from typing import Dict as _Dict
    from typing import Tuple as _Tuple
except Exception:
    raise ImportError(
        "This script requires newer features of Python, as provided "
        "by the dataclasses and typing modules. Please upgrade to "
        "at least Python 3.6 and try again. If you are using "
        "Python 3.6 then make sure that the 'dataclasses' module "
        "has been installed."
    )


try:
    if "NO_NUMBA" in _os.environ and _os.environ["NO_NUMBA"] == "1":
        # used for debugging - disable numba
        raise ImportError("Disabling numba")

    import numba as _numba

    _MIN_CHUNK_SIZE = 1000
except Exception:
    print(
        "WARNING: numba is not available. This is used to accelerate "
        "this script. We really recommend installing numba, "
        "e.g. via `pip install numba` or `conda install numba` "
        "as this will make this script run significantly (hundreds "
        "of times!) faster."
    )

    _MIN_CHUNK_SIZE = 10

    # This is a fake numba module which will enable the numba'd code
    # to run without requiring the module. Note that it won't be
    # compiled, so will run really slowly!
    class _numba:
        @staticmethod
        def jit(**kwargs):
            def wrapped(func):
                return func

            return wrapped

        @staticmethod
        def prange(*args, **kwargs):
            return range(*args, **kwargs)

        class CONFIG:
            NUMBA_NUM_THREADS = 1

        config = CONFIG()


try:
    import numpy as _np
except Exception:
    raise ImportError(
        "This script requires numpy to accelerate key parts. "
        "Please install numpy (e.g. via 'conda install numpy' or "
        "'pip install numpy' and try again."
    )


def _no_progress_bar(x, **kwargs):
    return x


try:
    if "NO_PROGRESS" in _os.environ and _os.environ["NO_PROGRESS"] == "1":
        # Option to completely disable progress bars (some people
        # really dislike them)
        raise ImportError("Disabling progress bars")

    from tqdm import tqdm as _progress_bar

    # This will fail on older versions of tqdm
    _p = _progress_bar(range(1, 10), unit="patterns", delay=1)
except Exception:
    _progress_bar = _no_progress_bar


@_dataclass
class Patterns:
    """This class holds all of the data relating to the patterns
    that distinguish between varieties that will be searched
    """

    """numpy 2D array of integers holding the patterns to be searched"""
    patterns = None

    """The IDs of the patterns, in the same order as the rows in
       the numpy 2D array"""
    ids: _List[str] = None

    """The varieties to be distinguised, in the same order as the
       columns in the numpy 2D array"""
    varieties: _List[str] = None

    """The minor allele frequency for each pattern, in the same order
       as the rows in the numpy 2D array. Note that this array should
       be sorted in order of decreasing MAF"""
    mafs: _List[float] = None

    """If there are any duplicate patterns, then this dictionary
       contains the ID of the canonical pattern as the key,
       with the value being the IDs of all of the other duplicates"""
    duplicates: _Dict[str, _List[str]] = None

    def __init__(self, patterns, ids, varieties, mafs, duplicates):
        self.patterns = patterns
        self.ids = ids
        self.varieties = varieties
        self.mafs = mafs
        self.duplicates = duplicates

    def get_pattern_as_string(self, i):
        """Return the ith pattern as a string"""
        return _get_pattern_from_array(self.patterns[i])


def _get_pattern_from_array(array) -> str:
    """Return a pattern encoded as an integer array
    into a string - this converts values less than
    0 into x
    """

    # Common formats are A, B for homozygotes (AA and BB) and AB for
    # heterozygotes or 0,1,2 for hom AA, het AB and hom BB.
    # We're going to convert A AB B to 0,1,2 format from here on

    values = {
        "0": "0",
        "1": "1",
        "2": "2",
        0: "0",
        1: "1",
        2: "2",
        "AB": "1",
        "A": "0",
        "B": "2",
    }

    pattern = [values.get(x, "x") for x in array]
    return "".join(pattern)


def convert_vcf_to_genotypes(input_file: str) -> str:
    """Convert the passed input file in VCF format into the genotypes
    format required by this program. This will write the output
    file into a new file called '{input_file}.genotypes', and will
    return that filename

    input_file: str  The full path to the input file to convert

    returns:
         The full path to the converted file
    """
    import re

    lines = open(input_file, "r").readlines()

    i = 0

    for line in lines:
        line = line.strip()

        if line.startswith("#CHR"):
            # we have found the CHROM line
            break

        i += 1

    if not line.startswith("#CHR"):
        print(f"WARNING: This does not look like a valid VCF file!")
        return None

    parts = line.split("\t")

    if len(parts) < 10:
        print(f"WARNING: Corruption on line {i + 1} of the VCF file!")
        return None

    head = "\t".join(parts[9:])

    output_file = f"{input_file}.genotypes"

    FILE = open(output_file, "w")

    FILE.write(f"Marker\t{head}\n")

    i += 1

    while i < len(lines):
        line = lines[i].strip()

        parts = line.split("\t")

        if len(parts) < 10:
            print(f"WARNING: Corruption on line {i + 1} of the VCF file!")
            FILE.close()
            os.unlink(output_file)
            return None

        chr = parts[0]
        pos = parts[1]
        data = parts[9:]

        FILE.write(f"{chr}_{pos}")

        for cell in data:
            field = cell.split(":")[0]

            m = re.search(r"^(\d)[\/\|](\d)", field)

            if m:
                a = int(m.groups()[0])
                b = int(m.groups()[1])

                if a == 0 and b == 1:
                    FILE.write("\t1")
                elif a == 0 and b == 0:
                    FILE.write("\t0")
                elif a == 1 and b == 1:
                    FILE.write("\t2")
                else:
                    print(f"WARNING: Unrecognised pattern {field} on line {i + 1}.")
                    FILE.write("\t-1")
            else:
                print(
                    f"WARNING: Unrecognised pattern {field} on "
                    f"line {i + 1}. Expecting 0/1, 1|1, 0/0 or equivalent."
                )
                FILE.write("\t-1")

        FILE.write("\n")
        i += 1

    FILE.close()

    return output_file


@_numba.jit(nopython=True, nogil=True, fastmath=True, parallel=True, cache=True)
def _calculate_mafs(
    data,
    min_call_rate: float = 0.9,
    min_maf: float = 0.001,
    print_progress: bool = False,
):
    """Now loop over the distinct SNP patterns to calculate their
    Minor Allele Frequency (MAF) score.

     min_maf: MAF is minor allele frequency.  This is set to a low level
              to include as many markers as possible but exclude rare
              error calls. It probably needs optimising for your data.

    This will return a numpy array of the scores for all of the patterns.
    in the order they appear in the data. A score of 0 is given for
    any skipped or invalid patterns. Note that, for speed, the score is
    returned as an array of integers. You need to divide this by the number
    of varieties to get the proper MAF.
    """
    if data is None:
        return None

    nrows: int = data.shape[0]
    ncols: int = data.shape[1]

    if nrows == 0 or ncols == 0:
        return None

    mafs = _np.zeros((nrows), _np.int64)

    # Go through in parallel and calculate maf for
    # each pattern. Patterns that should be skipped
    # will be given a maf of 0
    for i in _numba.prange(0, nrows):
        fails: int = 0
        zeros: int = 0
        ones: int = 0
        twos: int = 0

        for j in range(0, ncols):
            x: int = data[i, j]

            if x == -1:
                fails += 1
            elif x == 0:
                zeros += 1
            elif x == 1:
                ones += 1
            else:
                twos += 1

        nalleles: int = (zeros != 0) + (ones != 0) + (twos != 0)
        call_rate: float = float(ncols - fails) / ncols

        if nalleles <= 1 or call_rate < min_call_rate:
            mafs[i] = 0
        else:
            # Logic steps to work out which is the second most common
            # call, which we'll define as the minor allele.
            if ones >= zeros and zeros >= twos:
                minor: int = zeros
            elif zeros >= ones and ones >= twos:
                minor: int = ones
            elif zeros >= twos and twos >= ones:
                minor: int = twos
            elif ones >= twos and twos >= zeros:
                minor: int = twos
            elif twos >= ones and ones >= zeros:
                minor: int = ones
            elif twos >= zeros and zeros >= ones:
                minor: int = zeros
            else:
                print("PROGRAM BUG!!! INVALID CONDITION!")
                minor: int = 0

            if minor < min_maf * ncols:
                # eliminate patterns with too low a maf
                minor = 0

            mafs[i] = minor

    return mafs


@_numba.jit(nopython=True, nogil=True, fastmath=True, cache=True)
def _sort_patterns(data, mafs, max_markers: int, print_progress: bool = False):
    """Return a numpy array of indicies that would represent
    the array of data sorted by maf. Note that this will
    remove duplicates and anything that has a maf score
    of zero
    """
    nrows: int = len(mafs)
    ncols: int = data.shape[1]

    if data.shape[0] != nrows:
        print("CORRUPT DATA!")
        return (None, None, None)

    if nrows == 0:
        return (None, None, None)

    # Get the indicies of the sorted mafs. Numpy sorts in increasing
    # order, but we need decreasing order (hence the [::-1])
    sorted_idxs = _np.argsort(mafs)[::-1]

    # We have to assume that the data is already sorted - need
    # to find and remove duplicates, plus ones where the
    # maf is zero.
    if mafs[sorted_idxs[0]] == 0:
        return (None, None, None)

    order = _np.full(nrows, fill_value=-1, dtype=_np.int64)
    duplicates = _np.full(nrows, fill_value=-1, dtype=_np.int64)

    order[0] = sorted_idxs[0]
    npatterns: int = 1
    nduplicates: int = 0

    for i in range(1, nrows):
        if npatterns >= max_markers:
            if i != nrows - 1:
                print("Maximum marker count reached! Ignoring further markers.")
                break

        idx: int = sorted_idxs[i]
        maf: int = mafs[idx]

        if maf == 0:
            continue

        # is this a duplicate of any that are above with the same
        # maf score?
        new_pattern: int = 1

        for j in range(i - 1, -1, -1):
            last_idx: int = sorted_idxs[j]

            if maf != mafs[last_idx]:
                # there are no more patterns with the same maf score
                break
            elif duplicates[last_idx] == -1:
                # is this equal to any of the previous patterns
                # that has the same maf score?
                all_same: int = 1

                for k in range(0, ncols):
                    if data[idx, k] != data[last_idx, k]:
                        all_same = 0
                        break

                if all_same == 1:
                    # this is a duplicate pattern
                    new_pattern = 0
                    duplicates[idx] = last_idx
                    break

        if new_pattern == 1:
            order[npatterns] = idx
            npatterns += 1
        else:
            nduplicates += 1

    # remove invalid patterns
    order = order[order != -1]

    npatterns: int = len(order)

    # now copy the patterns, in order, to the output array
    patterns = _np.zeros((npatterns, ncols), _np.int64)

    for i in range(0, npatterns):
        idx = order[i]

        for j in range(0, ncols):
            patterns[i, j] = data[idx, j]

    return (patterns, order, duplicates)


def load_patterns(
    input_file: str,
    min_call_rate: float = 0.9,
    min_maf: float = 0.001,
    max_markers: int = 1000000000000,
    print_progress: bool = False,
) -> Patterns:
    """Load all of the patterns from the passed file.
    The patterns will be converted to the correct format,
    including cleaning / conversion of A, B, AB converted
    to 0, 1, 2 format.

    min_call_rate: Ignore markers with less than this proportion
                   of valid (0, 1 of 2 ) calls.

    min_maf: Ignore patterns with a MAF below this value

    max_markers: Ignore more than this number of patterns
                 (extra patterns with lower MAFs will be ignored)

    print_progress: Print the progress of reading / processing
                    the patterns to the screen. If False then
                    this function will not print anything.

    This will return a valid Patterns object that will contain
    all of the data for the patterns, sorted in decreasing
    MAF order
    """
    # Read in the data - assume this is comma separated for now. Can
    # easily add a test to change to tab separated if needed
    if print_progress:
        progress = _progress_bar
        print(f"Loading '{input_file}'...")
    else:
        progress = _no_progress_bar

    import csv

    lines = open(input_file, "r").readlines()
    dialect = csv.Sniffer().sniff(lines[0], delimiters=[" ", ",", "\t"])

    # the varieties are the column headers (minus the first column
    # which is the ID code for the pattern)
    varieties = []

    for variety in list(csv.reader([lines[0]], dialect=dialect))[0][1:]:
        varieties.append(variety.lstrip().rstrip())

    ids = []
    nrows = len(lines) - 1
    ncols = len(varieties)

    data = _np.full((nrows, ncols), -1, _np.int64)

    if print_progress:
        print(f"Reading {nrows} patterns for {ncols} varieties...")
        progress = _progress_bar
    else:
        progress = _no_progress_bar

    npatterns = 0

    irow = _np.full(ncols, -1, _np.int64)

    for i in progress(range(1, nrows + 1), unit="patterns", delay=1):
        parts = list(csv.reader([lines[i]], dialect=dialect))[0]

        if len(parts) != ncols + 1:
            print(f"WARNING - invalid row! '{parts}' : {len(parts)} vs {ncols}")
        else:
            ids.append(parts[0])
            row = _np.asarray(parts[1:], _np.bytes_)
            pattern = data[npatterns]
            pattern[(row == b"0") | (row == b"A")] = 0
            pattern[(row == b"1") | (row == b"AB")] = 1
            pattern[(row == b"2") | (row == b"B")] = 2

            npatterns += 1

    if print_progress:
        print(f"Successfully read {npatterns} patterns.\n")
        print(f"Calculating MAFs, removing duplicates and sorting patterns...")

    mafs = _calculate_mafs(
        data,
        min_call_rate=min_call_rate,
        min_maf=min_maf,
        print_progress=print_progress,
    )

    if print_progress:
        num_eliminated: int = len(mafs[mafs == 0])
        print(
            "\nNumber of eliminated patterns (by min_call_rate, min_maf or "
            f"requiring more than one allele) is {num_eliminated}."
        )

    (patterns, order, dups) = _sort_patterns(
        data, mafs, max_markers=max_markers, print_progress=print_progress
    )

    if patterns is None:
        if print_progress:
            print(
                "\nWARNING! There are no patterns that exceeded "
                f"the required criteria (min_maf = {min_maf}, "
                f"min_call_rate = {min_call_rate}).\n"
            )
        return None

    sorted_ids = []
    sorted_mafs = []
    duplicates = {}

    for idx in order:
        sorted_ids.append(ids[idx])
        sorted_mafs.append(mafs[idx] / ncols)

    for i, dup in enumerate(dups):
        if dup != -1:
            same = _np.array_equal(data[dup], data[i])

            if not same:
                print(
                    "WARNING: Program bug. Two patterns which are "
                    "flagged as identical aren't the same! "
                    f"{dup} and {i}"
                )
                assert same

            canonical = ids[dup]

            if canonical in duplicates:
                duplicates[canonical].append(ids[i])
            else:
                duplicates[canonical] = [ids[i]]

    if print_progress:
        print(
            f"\nLoaded marker data for {patterns.shape[0]} "
            "distinct patterns that have a sufficiently high "
            "MAF and call rate to be worth including."
        )

    assert len(sorted_ids) == patterns.shape[0]
    assert len(varieties) == patterns.shape[1]

    return Patterns(
        patterns=patterns,
        ids=sorted_ids,
        varieties=varieties,
        mafs=sorted_mafs,
        duplicates=duplicates,
    )


@_numba.jit(nopython=True, nogil=True, fastmath=True, parallel=True, cache=True)
def _calculate_best_possible_score(patterns, start: int, end: int):
    """Calculate the best possible score that could be achieved
    using all of the patterns
    """
    npatterns: int = patterns.shape[0]

    score: int = 0  # reduction variable
    # see http://numba.pydata.org/numba-doc/0.12.1/prange.html?highlight=parallel
    # for variable privatization rules
    for i in _numba.prange(start, end):
        for j in range(0, i):
            for p in range(0, npatterns):
                ival: int = patterns[p, i]
                jval: int = patterns[p, j]
                if ival != -1 and jval != -1 and ival != jval:
                    score += 1
                    break  # stop early if cell contains a one

    return score


def calculate_best_possible_score(
    patterns: Patterns, print_progress: bool = False
) -> int:
    """Calculate the best possible score for the passed Patterns
    object.

    patterns: The Patterns object containing the patterns to search

    print_progress: Whether or not to print any progress status
                    to output

    This returns the best possible score
    """

    if type(patterns) != Patterns:
        raise TypeError("This function requires a valid Patterns object!")

    patterns = patterns.patterns

    ncols: int = patterns.shape[1]

    if print_progress:
        chunk_size = min(_MIN_CHUNK_SIZE, ncols)
        progress = _progress_bar
    else:
        chunk_size = ncols
        progress = _no_progress_bar

    nchunks: int = int(ncols / chunk_size)

    while nchunks * chunk_size < ncols:
        nchunks += 1

    if nchunks < 4:
        nchunks = 1
        chunk_size = ncols
        progress = _no_progress_bar

    score: int = 0

    for i in progress(
        range(0, nchunks), delay=1, unit="variety", unit_scale=chunk_size
    ):
        start: int = i * chunk_size
        end: int = min((i + 1) * chunk_size, ncols)
        score += _calculate_best_possible_score(patterns, start, end)

    return score


@_numba.jit(nopython=True, nogil=True, fastmath=True, parallel=True, cache=True)
def _chunked_first_score_patterns(patterns, skip_patterns, scores, start, end):
    ncols: int = patterns.shape[1]

    for p in _numba.prange(start, end):
        if not skip_patterns[p]:
            score: int = 0

            # If we were to sort the rows by value the resulting scoring
            # matrix would be a matrix of ones, with a block diagonal of
            # zeros, each block being having the dimension corresponding to
            # the count of the identifer. We can therefore skip the O(n^2)
            # visiting each element and instead calculate the score directly.
            bins = _np.zeros(4, _np.int64)
            for i in range(0, ncols):
                # Here we take advantage of the identifiers being -1, 0, 1, 2 to
                # use them to directly index the counter. We take advanage of
                # wrap-around indexing so that -1 ends up in index 3.
                # We could instead have used numpy.bincount, but then
                # we would have to recode or remove -1 values first.
                bins[patterns[p, i]] += 1
            # Examining the matrix we can see that the score can be calculated directly as:
            score = bins[0] * bins[1] + (bins[0] + bins[1]) * bins[2]
            # If we wanted to handle more than three groups this formula could be extended
            # We skip the invalid -1 code by ignoring the last bin value in this
            # calculation.
            scores[p] = score

            if score == 0:
                skip_patterns[p] = 1


def _first_score_patterns(patterns, skip_patterns, print_progress: bool = True):
    """Do the work of scoring all of the passed patterns against
    the current value of the matrix. This returns a tuple
    of the best score and the index of the pattern with
    that best score
    """
    npatterns: int = patterns.shape[0]
    scores = _np.zeros(npatterns, _np.int64)

    if print_progress:
        chunk_size = min(_MIN_CHUNK_SIZE, npatterns)
        progress = _progress_bar
    else:
        chunk_size = npatterns
        progress = _no_progress_bar

    nchunks: int = int(npatterns / chunk_size)

    while nchunks * chunk_size < npatterns:
        nchunks += 1

    if nchunks < 4:
        nchunks = 1
        chunk_size = npatterns
        progress = _no_progress_bar

    for i in progress(
        range(0, nchunks), delay=1, unit="patterns", unit_scale=chunk_size
    ):
        start: int = i * chunk_size
        end: int = min((i + 1) * chunk_size, npatterns)
        _chunked_first_score_patterns(patterns, skip_patterns, scores, start, end)

    return scores


@_numba.jit(nopython=True, cache=True)
def _unmatched_element_indices(pattern, expected_size):
    """Calculate the co-ordinates of the elements with value
    zero in the matrix corresponding to the passed in pattern.
    Returns
    =======
         indices : N x 2 numpy array containing row/column of each zero
    """
    indices = _np.empty((expected_size, 2), dtype=_np.int64)
    nele: int = pattern.size
    idx: int = 0  # shared across iterations, so can't parallelise the loops
    for i in range(0, nele):
        ival: int = pattern[i]
        for j in range(0, i):
            jval: int = pattern[j]
            if ival == -1 or jval == -1 or ival == jval:
                indices[idx, 0] = i
                indices[idx, 1] = j
                idx += 1

    assert idx == expected_size

    return indices


@_numba.jit(nopython=True, nogil=True, fastmath=True, parallel=True, cache=True)
def _chunked_rescore_patterns(
    patterns,
    indices,
    skip_patterns,
    scores,
    sorted_idxs,
    best_score: int,
    start: int,
    end: int,
):
    ncols: int = patterns.shape[1]
    nidx: int = indices.shape[0]

    nthreads: int = _numba.config.NUMBA_NUM_THREADS

    best_scores = _np.zeros(nthreads, _np.int64)

    for thread_id in _numba.prange(0, nthreads):
        my_best_score: int = best_score

        for chunk in range(start, end, nthreads):
            idx: int = chunk + thread_id

            if idx >= end:
                break

            p: int = sorted_idxs[idx]

            if skip_patterns[p]:
                continue

            current_score: int = scores[p]

            if current_score < my_best_score:
                # there are no more patterns with a better score
                break

            # this pattern could be the best scoring pattern...
            score: int = 0

            for idx in range(0, nidx):
                i: int = indices[idx, 0]
                j: int = indices[idx, 1]
                ival: int = patterns[p, i]
                jval: int = patterns[p, j]
                if ival != -1 and jval != -1 and ival != jval:
                    score += 1

            scores[p] = score

            if score == 0:
                skip_patterns[p] = 1

            if score > my_best_score:
                my_best_score = score

        best_scores[thread_id] = my_best_score

    best_score = 0

    for score in best_scores:
        if score > best_score:
            best_score = score

    return best_score


def _rescore_patterns(
    patterns, indices, skip_patterns, scores, sorted_idxs, print_progress: bool = False
):
    """Do the work of scoring all of the passed patterns against
    the current value of the matrix. This returns a tuple
    of the best score and the index of the pattern with
    that best score
    """
    npatterns: int = patterns.shape[0]

    if print_progress:
        chunk_size = min(_MIN_CHUNK_SIZE, npatterns)
        progress = _progress_bar
    else:
        chunk_size = npatterns
        progress = _no_progress_bar

    nchunks: int = int(npatterns / chunk_size)

    while nchunks * chunk_size < npatterns:
        nchunks += 1

    if nchunks < 4:
        nchunks = 1
        chunk_size = npatterns
        progress = _no_progress_bar

    best_score: int = 0

    for i in progress(
        range(0, nchunks), delay=1, unit="patterns", unit_scale=chunk_size
    ):
        start: int = i * chunk_size
        end: int = min((i + 1) * chunk_size, npatterns)
        best_score: int = _chunked_rescore_patterns(
            patterns,
            indices,
            skip_patterns,
            scores,
            sorted_idxs,
            best_score,
            start,
            end,
        )

    return _np.argsort(scores)[::-1]


@_numba.jit(nopython=True, cache=True)
def _remove_matched_indices(data, indices, num_fewer):
    """Removes indices from the input array corresponding to
    ones in the matrix generated from the data array
    Returns
    =======
         new_indices : N x 2 numpy array containing the new indices
    """
    num_old: int = indices.shape[0]
    new_size: int = num_old - num_fewer
    nrows: int = data.shape[0]
    new_indices = _np.empty((new_size, 2), dtype=_np.int64)
    new_idx: int = 0  # shared accross iterations, so can't parallelise the loop
    for old_idx in range(0, num_old):
        i: int = indices[old_idx, 0]
        j: int = indices[old_idx, 1]
        ival: int = data[i]
        jval: int = data[j]
        if ival == -1 or jval == -1 or ival == jval:
            new_indices[new_idx] = [i, j]
            new_idx += 1

    assert new_idx == new_size

    return new_indices


@_numba.jit(nopython=True, fastmath=True, nogil=True, parallel=True, cache=True)
def _create_matrix(pattern, matrix):
    """Create the stencil matrix for this pattern, based on the
    current value of the combined stencil matrix (i.e. only
    cover up holes that are not already covered
    """
    n = len(pattern)

    m = _np.zeros((n, n), _np.int64)

    for i in _numba.prange(0, n):
        ival = pattern[i]

        if ival != -1:
            for j in range(i + 1, n):
                jval = pattern[j]

                # If this cell in the matrix is currently set to zero,
                # i.e. this pair of varieties (i and j) are unresolved,
                # and their genotypes are valid and different, then we can
                # set this cell in the test matrix to 1 (= resolved) -
                # otherwise it remains set to zero.
                if jval != -1 and ival != jval and matrix[i, j] == 0:
                    m[i, j] = 1

    return m


@_numba.jit(nopython=True, fastmath=True, nogil=True, parallel=True, cache=True)
def _calculate_score(matrix):
    ncols: int = matrix.shape[1]

    test_score: int = 0

    for i in _numba.prange(0, ncols):
        for j in range(i + 1, ncols):
            test_score += matrix[i, j]

    return test_score


@_numba.jit(nopython=True, fastmath=True, nogil=True, parallel=False, cache=True)
def _get_unresolved(matrix):
    ncols: int = matrix.shape[1]

    unresolved = _np.zeros((ncols, 2), _np.int64)
    num_unresolved: int = 0

    for i in range(0, ncols):
        for j in range(i + 1, ncols):
            if matrix[i, j] == 0:
                unresolved[num_unresolved, 0] = i
                unresolved[num_unresolved, 1] = j
                num_unresolved += 1

    return (num_unresolved, unresolved)


def get_unresolved(
    patterns: Patterns,
    best_patterns: _List[_Tuple[int, int]],
    print_progress: bool = False,
):
    """Return the IDs of any of the varieties that are
    not resolved by the set of best patterns output
    by the 'find_best_patterns' function.

    patterns: Patterns
        The input data to process

    best_patterns: List[Tuple[int, int]]
        The collection of best patterns output from the
        find_best_patterns function

    returns: List[Tuple[str, str]]
        The lists of pairs of varieties that cannot be
        distinguished from one another. The names
        of the varieties are returned
    """
    ncols: int = len(patterns.varieties)

    # first, build the matrix of what is resolved
    matrix = _np.zeros((ncols, ncols), _np.int64)

    if print_progress:
        print("Identifying unresolved varieties...")
        progress = _progress_bar
    else:
        progress = _no_progress_bar

    for i in progress(range(0, len(best_patterns)), delay=1, unit="patterns"):
        (pattern, score) = best_patterns[i]
        matrix += _create_matrix(patterns.patterns[pattern], matrix)

    # now find the number and ID of unresolved varieties from this matrix
    (num_unresolved, unresolved) = _get_unresolved(matrix)

    unresolved_varieties = {}

    result = []

    for idx in range(0, num_unresolved):
        i: int = unresolved[idx, 0]
        j: int = unresolved[idx, 1]

        var1 = patterns.varieties[i]
        var2 = patterns.varieties[j]

        unresolved_varieties[i] = 1
        unresolved_varieties[j] = 1

        result.append((var1, var2))

    return result


def find_best_patterns(
    patterns: Patterns, print_progress: bool = False
) -> _List[_Tuple[int, int]]:
    """This is the main function where we iterate through all of the
    available rows of SNP data and find the one that adds the most
    new "1s" to the overall scoring matrix. This function will return
    when the current_score value is zero - i.e. adding
    another row doesn't add anything to the overall matrix score.

    This returns the best patterns with their associated score,
    as a list of tuples, e.g.

    [ (index of best pattern, score of best pattern),
      (index of second best pattern, score of second best pattern),
      ...
      (index of last pattern, score of last pattern)
    ]
    """

    # create the scoring matrix
    ncols: int = len(patterns.patterns[0])

    perfect_score: int = int((ncols * (ncols - 1)) / 2)

    # estimate the best score...
    if print_progress:
        print("\nCalculating the best possible score...")

    best_possible_score: int = calculate_best_possible_score(
        patterns, print_progress=print_progress
    )

    if print_progress:
        print(
            f"The best possible score is {best_possible_score}. This would "
            f"resolve {100.0 * best_possible_score / perfect_score:.4f}% "
            "of varieties."
        )

    if print_progress:
        print(
            f"\n{perfect_score} varietal comparisons of which "
            f"{best_possible_score} can be resolved."
        )
        print("\nIteration\tCumulativeResolved\tProportion\tMarkerID\tPattern")

    # now find the intrinsic scores of all of the patterns
    skip_patterns = _np.zeros(len(patterns.patterns), _np.int64)

    scores = _first_score_patterns(patterns.patterns, skip_patterns, print_progress)

    # get the sorted list of these scores - this is an array of indicies
    # into the scores / patterns lists
    sorted_idxs = _np.argsort(scores)[::-1]

    # the first pattern is the best one
    iteration: int = 1
    best_pattern = sorted_idxs[0]
    best_score: int = scores[best_pattern]
    best_patterns = [(best_pattern, best_score)]
    cumulative_score: int = best_score
    current_score: int = best_score
    proportion_resolved = cumulative_score / perfect_score

    # patterns which have been used are given a score of zero
    scores[best_pattern] = 0
    skip_patterns[best_pattern] = 1

    # number of elements that can still distinguish varieties
    nremain = perfect_score - best_score

    # create a list indicies representing a sparse matrix containing the elements
    # can be still be used to distinguish varieties
    remaining_indices = _unmatched_element_indices(
        patterns.patterns[best_pattern], nremain
    )

    if print_progress:
        pattern = _get_pattern_from_array(patterns.patterns[best_pattern])
        pattern_id = patterns.ids[best_pattern]

        print(
            f"1\t{cumulative_score}\t"
            f"{100.0 * proportion_resolved:.4f}%\t"
            f"{pattern_id}\t{pattern}"
        )
        print(f"Best score from this iteration is {best_score}.")

    # we know that scores can only go down, so iterate through the remaining
    # patterns in descending score order, updating their score based on
    # the current matrix,
    while current_score > 0:
        iteration += 1

        sorted_idxs = _rescore_patterns(
            patterns.patterns,
            remaining_indices,
            skip_patterns,
            scores,
            sorted_idxs,
            print_progress,
        )

        best_pattern = sorted_idxs[0]
        best_score = scores[best_pattern]

        if best_score > 0:
            cumulative_score += best_score
            proportion_resolved = cumulative_score / perfect_score
            best_patterns.append((best_pattern, cumulative_score))

            scores[best_pattern] = 0
            skip_patterns[best_pattern] = 1

            remaining_indices = _remove_matched_indices(
                patterns.patterns[best_pattern], remaining_indices, best_score
            )
            if print_progress:
                pattern = _get_pattern_from_array(patterns.patterns[best_pattern])
                pattern_id = patterns.ids[best_pattern]

                print(
                    f"{iteration}\t{cumulative_score}\t"
                    f"{100.0 * proportion_resolved:.4f}%\t"
                    f"{pattern_id}\t{pattern}"
                )
                print(f"Best score from this iteration is {best_score}.")

            if cumulative_score == best_possible_score:
                break

        current_score: int = best_score

    if cumulative_score != best_possible_score:
        print(
            "\n\nWARNING: The algorithm should have been able to find a set "
            f"of patterns that resolved {best_possible_score} varieties "
            f"but was only able to resolve {cumulative_score} varieties. "
            "This suggests a bug or error in the program!\n"
        )

    # validate that the selected best patterns do indeed distinguish
    # the correct number of varieties
    matrix = _np.zeros((ncols, ncols), _np.int64)

    last_score: int = 0

    if print_progress:
        print("\n\nValidating that the result is correct...")
        progress = _progress_bar
    else:
        progress = _no_progress_bar

    for i in progress(range(0, len(best_patterns)), delay=1, unit="patterns"):
        (pattern, score) = best_patterns[i]

        matrix += _create_matrix(patterns.patterns[pattern], matrix)

        test_score = _calculate_score(matrix)

        if test_score - last_score != score:
            print(
                f"WARNING: {patterns.ids[pattern]} {test_score - last_score} vs {score}"
            )

    if test_score != best_possible_score:
        print(
            "\n\nWARNING: The patterns selected by the algorithm do not "
            f"give a total score ({test_score}) that is equal to the "
            "best possible score that should have been attainable "
            f"({best_possible_score}). This suggests a bug or error "
            "in the program!\n"
        )

    elif print_progress:
        print("All good!")

        ncomparisons: int = int(len(best_patterns) * ncols * (ncols - 1) / 2)

        print(f"\n{ncomparisons} comparisons.")

        (num_unresolved, unresolved) = _get_unresolved(matrix)

        unresolved_varieties = {}

        for idx in range(0, num_unresolved):
            i: int = unresolved[idx, 0]
            j: int = unresolved[idx, 1]

            var1 = patterns.varieties[i]
            var2 = patterns.varieties[j]

            unresolved_varieties[i] = 1
            unresolved_varieties[j] = 1

            print(f"{var1} not resolved from {var2}")

        if len(unresolved_varieties) == 0:
            print(f"All {ncols} varieties resolved.")
        else:
            print(f"{len(unresolved_varieties)} of {ncols} varieties unresolved.")

    return best_patterns


def write_markers_to_file(
    filename: str, patterns: Patterns, best_patterns: _List[_Tuple[int, int]]
):
    """Write the selected markers in 'best_patterns' from 'patterns'
    to the output file 'filename'.
    """
    with open(filename, "w") as FILE:
        FILE.write("CumulativeResolved\tMarkerID\tGenotypePattern\n")

        for idx, score in best_patterns:
            pid = patterns.ids[idx]
            pattern = patterns.get_pattern_as_string(idx)
            FILE.write(f"{score}\t{pid}\t{pattern}\n")


def write_markers_with_headers_to_file(
    filename: str, patterns: Patterns, best_patterns: _List[_Tuple[int, int]]
):
    """Write the selected markers in 'best_patterns' from 'patterns'
    to the output file 'filename'
    """
    with open(filename, "w") as FILE:
        header = "\t".join(patterns.varieties)
        FILE.write(f"ID\t{header}\n")

        for idx, _ in best_patterns:
            FILE.write(f"{patterns.ids[idx]}\t")
            FILE.write(
                "\t".join([str(x) if x >= 0 else "x" for x in patterns.patterns[idx]])
            )
            FILE.write("\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        prog="select_minimal_markers.py",
        description="Find the minimal set "
        "of markers needed to distinguish between "
        "varieties.",
    )

    parser.add_argument(
        "input_file",
        metavar="input_file",
        type=str,
        nargs=1,
        help="The input file containing the markers to be processed.",
    )

    parser.add_argument(
        "--min_call_rate",
        nargs="?",
        type=float,
        default=0.9,
        const=1,
        help="The minimum call rate needed of a pattern "
        "to make it worth processing. "
        f"Default value is 0.9.",
    )

    parser.add_argument(
        "--min_maf",
        nargs="?",
        type=float,
        default=0.001,
        const=1,
        help="The minimum Minor Allele Frequency of a "
        "pattern needed to make it worth processing. "
        f"Default value is 0.001.",
    )

    parser.add_argument(
        "--max_markers",
        nargs="?",
        type=int,
        default=1000000000000,
        const=1,
        help="The maximum number of markers to process. "
        "Markers are first sorted by MAF, so any "
        "markers with a low MAF after max_markers "
        "will be discarded before processing. Default "
        f"value is 1000000000000.",
    )

    parser.add_argument(
        "--vcf",
        action="store_true",
        help="Read the input file in VCF format. Note that "
        "this will first convert the file into "
        "the internal format, save that to disk "
        "using the filename '{input_file}.genotypes' "
        "and will run the program against the "
        "converted file.",
    )

    parser.add_argument(
        "--silent",
        action="store_true",
        help="Silence all output written to the console.",
    )

    args = parser.parse_args()

    input_file = args.input_file[0]

    if args.vcf:
        input_file = convert_vcf_to_genotypes(input_file)

        if input_file is None:
            import sys

            sys.exit(-1)

    if args.silent:
        print_progress = False
    else:
        print_progress = True

    patterns = load_patterns(
        input_file,
        min_call_rate=args.min_call_rate,
        min_maf=args.min_maf,
        max_markers=args.max_markers,
        print_progress=print_progress,
    )

    if patterns is None:
        best_patterns = []
        matrix = None
        import sys

        sys.exit(-1)
    else:
        best_patterns = find_best_patterns(patterns, print_progress=print_progress)

    if print_progress:
        print(f"\nProcessing complete! Writing output")

    import os

    base = os.path.splitext(input_file)[0]

    write_markers_to_file(f"{base}_minimal_markers.txt", patterns, best_patterns)

    write_markers_with_headers_to_file(
        f"{base}_selected_markers_with_headers.txt", patterns, best_patterns
    )
