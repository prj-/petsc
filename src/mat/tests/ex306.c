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

int main(int argc, char **argv)
{
  Mat     A00, A01, A10, A11, nest, *submats;
  Mat     mats[4];
  IS      rows[2], cols[2], irow[4], icol[4];
  PetscInt i, j, n = 4;
  PetscBool equal;
  MPI_Comm  comm;

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
    for (j = 0; j < 2; j++) {
      irow[i * 2 + j] = rows[i];
      icol[i * 2 + j] = cols[j];
    }
  }

  /* Initial extraction */
  PetscCall(MatCreateSubMatrices(nest, n, irow, icol, MAT_INITIAL_MATRIX, &submats));

  /* Verify each extracted block matches the original */
  PetscCall(MatEqual(submats[0], A00, &equal));
  PetscCheck(equal, comm, PETSC_ERR_PLIB, "submats[0] != A00");
  PetscCall(MatEqual(submats[1], A01, &equal));
  PetscCheck(equal, comm, PETSC_ERR_PLIB, "submats[1] != A01");
  PetscCall(MatEqual(submats[2], A10, &equal));
  PetscCheck(equal, comm, PETSC_ERR_PLIB, "submats[2] != A10");
  PetscCall(MatEqual(submats[3], A11, &equal));
  PetscCheck(equal, comm, PETSC_ERR_PLIB, "submats[3] != A11");

  /* Reuse extraction */
  PetscCall(MatCreateSubMatrices(nest, n, irow, icol, MAT_REUSE_MATRIX, &submats));

  PetscCall(MatEqual(submats[0], A00, &equal));
  PetscCheck(equal, comm, PETSC_ERR_PLIB, "reuse: submats[0] != A00");
  PetscCall(MatEqual(submats[1], A01, &equal));
  PetscCheck(equal, comm, PETSC_ERR_PLIB, "reuse: submats[1] != A01");
  PetscCall(MatEqual(submats[2], A10, &equal));
  PetscCheck(equal, comm, PETSC_ERR_PLIB, "reuse: submats[2] != A10");
  PetscCall(MatEqual(submats[3], A11, &equal));
  PetscCheck(equal, comm, PETSC_ERR_PLIB, "reuse: submats[3] != A11");

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
