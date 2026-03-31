"""Python equivalent of src/ksp/ksp/tutorials/ex82.c

Solves a linear system with a MATHTOOL hierarchical matrix using PCHPDDM
(when ``-pc_type hpddm`` is supplied) or another solver.

The kernel is

   K(x, y) = 1 / (0.01 + ||x - y||_2)

where x and y are random points in R^dim.

Usage::

    mpiexec -n 4 python ex82.py -ksp_view -ksp_converged_reason \\
        -mat_htool_epsilon 1e-2 -m_local 200 -pc_type hpddm \\
        -pc_hpddm_define_subdomains -pc_hpddm_levels_1_sub_pc_type lu \\
        -pc_hpddm_levels_1_eps_nev 1 -pc_hpddm_coarse_pc_type lu \\
        -pc_hpddm_levels_1_eps_gen_non_hermitian -symmetric 0 -overlap 2
"""

import sys

import numpy

import petsc4py

petsc4py.init(sys.argv)

from petsc4py import PETSc  # noqa: E402

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
OptDB   = PETSc.Options()
m       = OptDB.getInt('m_local', 100)    # local number of rows/cols per rank
dim     = OptDB.getInt('dim', 3)           # spatial dimension
overlap = OptDB.getInt('overlap', 1)       # subdomain overlap for HPDDM
sym     = OptDB.getBool('symmetric', False)

comm = PETSc.COMM_WORLD
size = comm.size
rank = comm.rank
M    = OptDB.getInt('M', size * m)         # global number of rows/cols

# ---------------------------------------------------------------------------
# Generate random local coordinates, then assemble global coordinate array.
# Each rank generates m*dim random values; the global array is built with
# MPI_Exscan (prefix sum) and MPI_Allreduce, mirroring the C tutorial.
# ---------------------------------------------------------------------------
rng    = numpy.random.default_rng(seed=rank)
coords = rng.random((m, dim)).astype(PETSc.RealType)

from mpi4py import MPI  # noqa: E402

mpi_comm = comm.tompi4py()

# Prefix sum: compute the starting point (in number of points) for this rank.
begin     = numpy.zeros(1, dtype=numpy.intp)
local_cnt = numpy.array([m], dtype=numpy.intp)
mpi_comm.Exscan(local_cnt, begin)
begin = int(begin[0])

# Build the global coordinate array on every rank via a zero-initialised
# buffer where each rank fills its own block, followed by Allreduce(SUM).
gcoords               = numpy.zeros((M, dim), dtype=PETSc.RealType)
gcoords[begin:begin + m] = coords
mpi_comm.Allreduce(MPI.IN_PLACE, gcoords, op=MPI.SUM)

# ---------------------------------------------------------------------------
# Kernel: K(x, y) = 1 / (0.01 + ||x - y||_2)
# ---------------------------------------------------------------------------
def kernel(sdim, nrows, ncols, rows, cols, v, ctx):
    """Evaluate the kernel for a submatrix block."""
    gc = ctx  # global coordinate array, shape (M, sdim)
    for i in range(nrows):
        xi = gc[rows[i]]
        for j in range(ncols):
            yj     = gc[cols[j]]
            diff   = xi - yj
            v[i, j] = 1.0 / (1.0e-2 + numpy.sqrt(numpy.dot(diff, diff)))


# ---------------------------------------------------------------------------
# Create MATHTOOL matrix
# ---------------------------------------------------------------------------
A = PETSc.Mat()
A.createHtoolFromKernel(
    [[m, M], [m, M]],
    dim,
    coords,
    coords,
    kernel,
    gcoords,
    comm=comm,
)
A.setOption(PETSc.Mat.Option.SYMMETRIC, sym)
A.setFromOptions()
A.assemble()

# ---------------------------------------------------------------------------
# Set up and solve the linear system
# ---------------------------------------------------------------------------
b, x = A.createVecs()

rdm = PETSc.Random().create(comm=comm)
rdm.setUp()
b.setRandom(rdm)
rdm.destroy()

ksp = PETSc.KSP().create(comm=comm)
ksp.setOperators(A)
ksp.setFromOptions()

# If PCHPDDM is chosen, set up the local auxiliary (subdomain) matrix.
pc = ksp.getPC()
if pc.getType() == PETSc.PC.Type.HPDDM:
    row_start, row_end = A.getOwnershipRange()
    n_local = row_end - row_start
    is_ = PETSc.IS().createStride(n_local, row_start, 1, comm=PETSc.COMM_SELF)
    A.increaseOverlap(is_, overlap)
    n_aug = is_.getLocalSize()
    # Trivial local identity matrix — tests that HPDDM plumbing is in place.
    aux = PETSc.Mat().createDense([n_aug, n_aug], comm=PETSc.COMM_SELF)
    aux.setOption(PETSc.Mat.Option.SYMMETRIC, sym)
    aux.assemble()
    aux.shift(1.0)
    pc.setHPDDMAuxiliaryMat(is_, aux)
    is_.destroy()
    aux.destroy()

ksp.solve(b, x)

# ---------------------------------------------------------------------------
# Clean up
# ---------------------------------------------------------------------------
ksp.destroy()
b.destroy()
x.destroy()
A.destroy()
