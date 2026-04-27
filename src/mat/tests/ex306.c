static char help[] = "Test MatCreateSubMatrices() for MATNEST.\n\n";

#include <petscmat.h>

static PetscErrorCode CreateDiagMat(MPI_Comm comm, PetscInt N, PetscScalar val, Mat *A)
{
  PetscInt istart, iend, i;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, N, N));
  PetscCall(MatSetFromOptions(*A));
  PetscCall(MatSetUp(*A));
  PetscCall(MatGetOwnershipRange(*A, &istart, &iend));
  for (i = istart; i < iend; i++) PetscCall(MatSetValue(*A, i, i, val, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Check that each diagonal entry of a sequential submatrix has the expected value */
static PetscErrorCode CheckDiag(Mat S, PetscScalar expected)
{
  Vec       diag;
  PetscInt  n, i;
  PetscBool ok = PETSC_TRUE;

  PetscFunctionBeginUser;
  PetscCall(MatCreateVecs(S, &diag, NULL));
  PetscCall(MatGetDiagonal(S, diag));
  PetscCall(VecGetLocalSize(diag, &n));
  for (i = 0; i < n; i++) {
    PetscScalar v;
    PetscCall(VecGetValues(diag, 1, &i, &v));
    if (v != expected) {
      ok = PETSC_FALSE;
      break;
    }
  }
  PetscCall(VecDestroy(&diag));
  PetscCheck(ok, PETSC_COMM_SELF, PETSC_ERR_PLIB, "diagonal value mismatch: expected %g", (double)PetscRealPart(expected));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat      A00, A01, A10, A11, nest, *submats;
  Mat      mats[4];
  IS       rows[2], cols[2], irow[4], icol[4];
  PetscInt i, j, n = 4;
  MPI_Comm comm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  /* Create four diagonal matrices with distinct diagonal values */
  PetscCall(CreateDiagMat(comm, 3, 1.0, &A00));
  PetscCall(CreateDiagMat(comm, 3, 2.0, &A01));
  PetscCall(CreateDiagMat(comm, 3, 3.0, &A10));
  PetscCall(CreateDiagMat(comm, 3, 4.0, &A11));

  /* Build a 2x2 nest */
  mats[0] = A00;
  mats[1] = A01;
  mats[2] = A10;
  mats[3] = A11;
  PetscCall(MatCreateNest(comm, 2, NULL, 2, NULL, mats, &nest));
  PetscCall(MatSetUp(nest));
  PetscCall(MatAssemblyBegin(nest, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(nest, MAT_FINAL_ASSEMBLY));

  /* Get global row/column ISes from the nest */
  PetscCall(MatNestGetISs(nest, rows, cols));

  /* Request all four (i,j) block combinations */
  for (i = 0; i < 2; i++) {
    for (j = 0; j < 2; j++) irow[i * 2 + j] = rows[i];
  }
  for (i = 0; i < 2; i++) {
    for (j = 0; j < 2; j++) icol[i * 2 + j] = cols[j];
  }

  /* Initial extraction - returns sequential submatrices */
  PetscCall(MatCreateSubMatrices(nest, n, irow, icol, MAT_INITIAL_MATRIX, &submats));

  /* Verify each extracted block has the correct diagonal value */
  PetscCall(CheckDiag(submats[0], 1.0));
  PetscCall(CheckDiag(submats[1], 2.0));
  PetscCall(CheckDiag(submats[2], 3.0));
  PetscCall(CheckDiag(submats[3], 4.0));

  /* Reuse extraction */
  PetscCall(MatCreateSubMatrices(nest, n, irow, icol, MAT_REUSE_MATRIX, &submats));

  PetscCall(CheckDiag(submats[0], 1.0));
  PetscCall(CheckDiag(submats[1], 2.0));
  PetscCall(CheckDiag(submats[2], 3.0));
  PetscCall(CheckDiag(submats[3], 4.0));

  PetscCall(MatDestroySubMatrices(n, &submats));
  PetscCall(MatDestroy(&nest));
  PetscCall(MatDestroy(&A00));
  PetscCall(MatDestroy(&A01));
  PetscCall(MatDestroy(&A10));
  PetscCall(MatDestroy(&A11));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      output_file: output/empty.out

   test:
      suffix: 2
      nsize: 2
      output_file: output/empty.out

TEST*/
