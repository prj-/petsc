static char help[] = "Tests MatCreateSubMatrices() for MATMPISBAIJ.\n\n"
                     "With an unsorted IS, MATMPISBAIJ silently returned wrong values because\n"
                     "the diagonal block is stored as MATSEQSBAIJ (upper triangle only), so\n"
                     "entries that belong to the upper triangle of the submatrix but to the lower\n"
                     "triangle of the original matrix were not sent and were missing from the result.\n"
                     "The fix adds the same sorted-IS requirement that MatCreateSubMatrix() already enforces.\n\n";

#include <petscmat.h>

/* Build a symmetric N-by-N banded matrix: diagonal=4, super-diagonals 1 and 2 = -1 */
static PetscErrorCode BuildMat(MPI_Comm comm, const char *type, PetscInt N, Mat *A)
{
  PetscInt rstart, rend, i;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, N, N));
  PetscCall(MatSetType(*A, type));
  PetscCall(MatSetUp(*A));
  PetscCall(MatGetOwnershipRange(*A, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    PetscCall(MatSetValue(*A, i, i, 4.0, INSERT_VALUES));
    if (i + 1 < N) PetscCall(MatSetValue(*A, i, i + 1, -1.0, INSERT_VALUES));
    if (i + 2 < N) PetscCall(MatSetValue(*A, i, i + 2, -1.0, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat         S, A, *submats_s, *submats_a, Cs;
  IS          is[1];
  PetscInt    N = 8, idx[4] = {0, 2, 5, 7};
  PetscBool   equal, test_unsorted = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &N, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_unsorted", &test_unsorted, NULL));

  PetscCall(BuildMat(PETSC_COMM_WORLD, MATMPISBAIJ, N, &S));
  PetscCall(BuildMat(PETSC_COMM_WORLD, MATMPIAIJ, N, &A));

  if (!test_unsorted) {
    /* Sorted IS: MatCreateSubMatrices() must agree with the AIJ reference */
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 4, idx, PETSC_COPY_VALUES, &is[0]));
    PetscCall(MatCreateSubMatrices(A, 1, is, is, MAT_INITIAL_MATRIX, &submats_a));
    PetscCall(MatCreateSubMatrices(S, 1, is, is, MAT_INITIAL_MATRIX, &submats_s));

    PetscCall(MatConvert(submats_s[0], MATSEQAIJ, MAT_INITIAL_MATRIX, &Cs));
    PetscCall(MatEqual(submats_a[0], Cs, &equal));
    PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "SBAIJ and AIJ submatrices differ for sorted IS");
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Sorted IS: SBAIJ submatrix matches AIJ submatrix\n"));

    PetscCall(MatDestroy(&Cs));
    PetscCall(MatDestroySubMatrices(1, &submats_a));
    PetscCall(MatDestroySubMatrices(1, &submats_s));
    PetscCall(ISDestroy(&is[0]));
  } else {
    /*
     * Unsorted IS: before the fix, MatCreateSubMatrices() for MATMPISBAIJ silently
     * returned wrong results (missing entries) because the SBAIJ diagonal block only
     * stores the upper triangle.  The fix raises PETSC_ERR_ARG_INCOMP instead.
     */
    PetscInt unsorted[4] = {7, 0, 5, 2};

    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 4, unsorted, PETSC_COPY_VALUES, &is[0]));
    PetscCall(MatCreateSubMatrices(S, 1, is, is, MAT_INITIAL_MATRIX, &submats_s));
    PetscCall(MatDestroySubMatrices(1, &submats_s));
    PetscCall(ISDestroy(&is[0]));
  }

  PetscCall(MatDestroy(&S));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2
      output_file: output/ex309.out

TEST*/
