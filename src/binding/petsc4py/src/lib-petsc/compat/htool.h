#if !defined(PETSC4PY_COMPAT_HTOOL_H)
#define PETSC4PY_COMPAT_HTOOL_H

#if !defined(PETSC_HAVE_HTOOL)

typedef PetscErrorCode MatHtoolKernelFn(PetscInt, PetscInt, PetscInt, const PetscInt *, const PetscInt *, PetscScalar *, void *);
typedef MatHtoolKernelFn *MatHtoolKernel;

typedef enum {
  MAT_HTOOL_COMPRESSOR_SYMPARTIAL_ACA,
  MAT_HTOOL_COMPRESSOR_FULL_ACA,
  MAT_HTOOL_COMPRESSOR_SVD
} MatHtoolCompressorType;

typedef enum {
  MAT_HTOOL_CLUSTERING_PCA_REGULAR,
  MAT_HTOOL_CLUSTERING_PCA_GEOMETRIC,
  MAT_HTOOL_CLUSTERING_BOUNDING_BOX_1_REGULAR,
  MAT_HTOOL_CLUSTERING_BOUNDING_BOX_1_GEOMETRIC
} MatHtoolClusteringType;

#define PetscMatHTOOLError do { \
    PetscFunctionBegin; \
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "%s() requires Htool", PETSC_FUNCTION_NAME); \
    PetscFunctionReturn(PETSC_ERR_SUP); \
    } while (0)

PetscErrorCode MatCreateHtoolFromKernel(PETSC_UNUSED MPI_Comm a, PETSC_UNUSED PetscInt b, PETSC_UNUSED PetscInt c, PETSC_UNUSED PetscInt d, PETSC_UNUSED PetscInt e, PETSC_UNUSED PetscInt f, PETSC_UNUSED const PetscReal g[], PETSC_UNUSED const PetscReal h[], PETSC_UNUSED MatHtoolKernelFn *i, PETSC_UNUSED void *j, PETSC_UNUSED Mat *k) {PetscMatHTOOLError;}
PetscErrorCode MatHtoolSetKernel(PETSC_UNUSED Mat a, PETSC_UNUSED MatHtoolKernelFn *b, PETSC_UNUSED void *c) {PetscMatHTOOLError;}
PetscErrorCode MatHtoolGetPermutationSource(PETSC_UNUSED Mat a, PETSC_UNUSED IS *b) {PetscMatHTOOLError;}
PetscErrorCode MatHtoolGetPermutationTarget(PETSC_UNUSED Mat a, PETSC_UNUSED IS *b) {PetscMatHTOOLError;}
PetscErrorCode MatHtoolUsePermutation(PETSC_UNUSED Mat a, PETSC_UNUSED PetscBool b) {PetscMatHTOOLError;}
PetscErrorCode MatHtoolUseRecompression(PETSC_UNUSED Mat a, PETSC_UNUSED PetscBool b) {PetscMatHTOOLError;}
PetscErrorCode MatHtoolGetEpsilon(PETSC_UNUSED Mat a, PETSC_UNUSED PetscReal *b) {PetscMatHTOOLError;}
PetscErrorCode MatHtoolSetEpsilon(PETSC_UNUSED Mat a, PETSC_UNUSED PetscReal b) {PetscMatHTOOLError;}
PetscErrorCode MatHtoolGetEta(PETSC_UNUSED Mat a, PETSC_UNUSED PetscReal *b) {PetscMatHTOOLError;}
PetscErrorCode MatHtoolSetEta(PETSC_UNUSED Mat a, PETSC_UNUSED PetscReal b) {PetscMatHTOOLError;}
PetscErrorCode MatHtoolGetMaxLeafSize(PETSC_UNUSED Mat a, PETSC_UNUSED PetscInt *b) {PetscMatHTOOLError;}
PetscErrorCode MatHtoolSetMaxLeafSize(PETSC_UNUSED Mat a, PETSC_UNUSED PetscInt b) {PetscMatHTOOLError;}
PetscErrorCode MatHtoolGetMinTargetDepth(PETSC_UNUSED Mat a, PETSC_UNUSED PetscInt *b) {PetscMatHTOOLError;}
PetscErrorCode MatHtoolSetMinTargetDepth(PETSC_UNUSED Mat a, PETSC_UNUSED PetscInt b) {PetscMatHTOOLError;}
PetscErrorCode MatHtoolGetMinSourceDepth(PETSC_UNUSED Mat a, PETSC_UNUSED PetscInt *b) {PetscMatHTOOLError;}
PetscErrorCode MatHtoolSetMinSourceDepth(PETSC_UNUSED Mat a, PETSC_UNUSED PetscInt b) {PetscMatHTOOLError;}
PetscErrorCode MatHtoolGetBlockTreeConsistency(PETSC_UNUSED Mat a, PETSC_UNUSED PetscBool *b) {PetscMatHTOOLError;}
PetscErrorCode MatHtoolSetBlockTreeConsistency(PETSC_UNUSED Mat a, PETSC_UNUSED PetscBool b) {PetscMatHTOOLError;}
PetscErrorCode MatHtoolGetCompressorType(PETSC_UNUSED Mat a, PETSC_UNUSED MatHtoolCompressorType *b) {PetscMatHTOOLError;}
PetscErrorCode MatHtoolSetCompressorType(PETSC_UNUSED Mat a, PETSC_UNUSED MatHtoolCompressorType b) {PetscMatHTOOLError;}
PetscErrorCode MatHtoolGetClusteringType(PETSC_UNUSED Mat a, PETSC_UNUSED MatHtoolClusteringType *b) {PetscMatHTOOLError;}
PetscErrorCode MatHtoolSetClusteringType(PETSC_UNUSED Mat a, PETSC_UNUSED MatHtoolClusteringType b) {PetscMatHTOOLError;}

#undef PetscMatHTOOLError

#endif /* !PETSC_HAVE_HTOOL */

#endif /* PETSC4PY_COMPAT_HTOOL_H */
