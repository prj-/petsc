"""
Demo: Creating a MATHTOOL hierarchical matrix using a Python kernel functor.

This example creates a hierarchical matrix arising from the kernel function

   K(x, y) = 1 / (0.01 + ||x - y||_2)

where x and y are points in R^3.  The kernel is defined as a Python callable
and passed together with point coordinates to `Mat.createHtoolFromKernel`.

The example demonstrates:
  * Defining a kernel functor in Python.
  * Passing coordinate spaces for target and source points.
  * Using `HtoolGetPermutationSource`/`HtoolGetPermutationTarget`.
  * Converting the hierarchical matrix to a dense matrix for verification.
"""

import sys

import numpy

import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc  # noqa: E402

# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------

N = 100   # number of points (global)
dim = 3   # spatial dimension

# Generate N points in R^dim distributed uniformly along a line.
comm = PETSc.COMM_WORLD
rank = comm.rank
size = comm.size

# Each MPI rank owns a contiguous block of rows/columns.
# Distribute N rows as evenly as possible across MPI ranks.
def _local_size(total, nproc, iproc):
    return total // nproc + (1 if iproc < total % nproc else 0)

local_rows = _local_size(N, size, rank)
coords = numpy.linspace(
    (1.0, 2.0, 3.0),
    (10.0, 20.0, 30.0),
    N,
    dtype=PETSc.RealType,
)
# Compute global offset for this rank.
offset = sum(_local_size(N, size, r) for r in range(rank))
# Local coordinates for this rank.
local_coords = coords[offset:offset + local_rows]

# ---------------------------------------------------------------------------
# Kernel functor
# ---------------------------------------------------------------------------
# The kernel is a Python callable with signature:
#   kernel(sdim, M, N, rows, cols, v, ctx)
#
# Parameters
# ----------
# sdim : int
#     Spatial dimension of the coordinates.
# M, N : int
#     Number of target/source points in this submatrix block.
# rows : ndarray of int, shape (M,)
#     Global indices of the target points.
# cols : ndarray of int, shape (N,)
#     Global indices of the source points.
# v : ndarray, shape (M, N)
#     Output array to be filled with kernel values (Fortran order).
# ctx : object
#     User context; here it is the full global coordinate array.

def kernel(sdim, M, N, rows, cols, v, ctx):
    """Evaluate 1 / (0.01 + ||x - y||) for all (target, source) pairs."""
    gcoords = ctx  # global coordinate array, shape (N_global, sdim)
    for i in range(M):
        xi = gcoords[rows[i]]
        for j in range(N):
            yj = gcoords[cols[j]]
            diff = xi - yj
            v[i, j] = 1.0 / (0.01 + numpy.sqrt(numpy.dot(diff, diff)))


# ---------------------------------------------------------------------------
# Create the MATHTOOL matrix
# ---------------------------------------------------------------------------
A = PETSc.Mat()
A.createHtoolFromKernel(
    [[local_rows, N], [local_rows, N]],   # size: ((local_rows, global_rows), (local_cols, global_cols))
    dim,
    local_coords,        # local target coordinates
    local_coords,        # local source coordinates (same for square matrix)
    kernel,
    coords,              # kernelctx: full global coordinates for index lookup
    comm=comm,
)
A.setFromOptions()
A.assemble()

PETSc.Sys.Print(f"Created MATHTOOL matrix of size {A.getSize()}")
PETSc.Sys.Print(f"  local size: {A.getLocalSize()}")

# ---------------------------------------------------------------------------
# Retrieve Htool permutations
# ---------------------------------------------------------------------------
iss = A.HtoolGetPermutationSource()
ist = A.HtoolGetPermutationTarget()
PETSc.Sys.Print(f"Source permutation size: {iss.getSize()}")
PETSc.Sys.Print(f"Target permutation size: {ist.getSize()}")

# ---------------------------------------------------------------------------
# Verify by comparing mat-vec product with dense matrix
# ---------------------------------------------------------------------------
x, y_htool = A.createVecs()
x.setRandom()

A.mult(x, y_htool)

D = A.convert('dense')
y_dense = D.createVecLeft()
D.mult(x, y_dense)

y_htool.axpy(-1.0, y_dense)
rel_err = y_htool.norm() / y_dense.norm()
PETSc.Sys.Print(f"Relative error ||y_htool - y_dense|| / ||y_dense|| = {rel_err:.2e}")
assert rel_err < 1.0e-4, f"Relative error too large: {rel_err}"
PETSc.Sys.Print("Verification passed.")
D.destroy()

A.destroy()
