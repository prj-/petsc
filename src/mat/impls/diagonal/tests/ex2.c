static const char help[] = "Test MatPtAP with MATDIAGONAL\n";

#include <petscmat.h>

/* KOKKOS: Following two cases will fail, as MatDiagonalScale_{Seq,MPI}AIJKOKKOS does
   not support CPU diagonal vector against AIJ KOKKOS.
   -amat_type diagonal -pmat_type aijkokkos -adiag_vec_type standard
   -amat_type aijkokkos -pmat_type diagonal -pdiag_vec_type standard */
static PetscErrorCode CreateTestMatrix(MPI_Comm comm, const char type[], const char prefix[], PetscInt m, PetscInt n, PetscRandom rand, Mat *M)
{
  PetscBool isdiag, isaij, isdense;
  PetscBool isaijkokkos, isaijcusparse, isaijhipsparse;
  PetscBool isdensecuda, isdensehip;

  PetscFunctionBeginUser;
  PetscCall(PetscStrcmp(type, MATDIAGONAL, &isdiag));
  PetscCall(PetscStrcmp(type, MATAIJ, &isaij));
  PetscCall(PetscStrcmp(type, MATDENSE, &isdense));
  PetscCall(PetscStrcmp(type, MATAIJKOKKOS, &isaijkokkos));
  PetscCall(PetscStrcmp(type, MATAIJCUSPARSE, &isaijcusparse));
  PetscCall(PetscStrcmp(type, MATAIJHIPSPARSE, &isaijhipsparse));
  PetscCall(PetscStrcmp(type, MATDENSECUDA, &isdensecuda));
  PetscCall(PetscStrcmp(type, MATDENSEHIP, &isdensehip));
  if (isdiag) {
    Vec d;

    PetscCall(VecCreate(comm, &d));
    if (prefix) PetscCall(VecSetOptionsPrefix(d, prefix));
    PetscCall(VecSetSizes(d, PETSC_DECIDE, m));
    PetscCall(VecSetFromOptions(d));
    PetscCall(VecSetRandom(d, rand));
    PetscCall(MatCreateDiagonal(d, M));
    PetscCall(VecDestroy(&d));
    PetscCall(MatAssemblyBegin(*M, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*M, MAT_FINAL_ASSEMBLY));
  } else if (isaij || isaijkokkos || isaijcusparse || isaijhipsparse) {
    PetscCall(MatCreate(comm, M));
    PetscCall(MatSetSizes(*M, PETSC_DECIDE, PETSC_DECIDE, m, n));
    PetscCall(MatSetType(*M, MATAIJ));
    PetscCall(MatSeqAIJSetPreallocation(*M, n, NULL));
    PetscCall(MatMPIAIJSetPreallocation(*M, n, NULL, n, NULL));
    PetscCall(MatSetUp(*M));
    PetscCall(MatSetRandom(*M, rand));
    PetscCall(MatAssemblyBegin(*M, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*M, MAT_FINAL_ASSEMBLY));
    if (isaijkokkos) PetscCall(MatConvert(*M, MATAIJKOKKOS, MAT_INPLACE_MATRIX, M));
    else if (isaijcusparse) PetscCall(MatConvert(*M, MATAIJCUSPARSE, MAT_INPLACE_MATRIX, M));
    else if (isaijhipsparse) PetscCall(MatConvert(*M, MATAIJHIPSPARSE, MAT_INPLACE_MATRIX, M));
  } else if (isdense || isdensecuda || isdensehip) {
    PetscCall(MatCreate(comm, M));
    PetscCall(MatSetSizes(*M, PETSC_DECIDE, PETSC_DECIDE, m, n));
    PetscCall(MatSetType(*M, MATDENSE));
    PetscCall(MatSetUp(*M));
    PetscCall(MatSetRandom(*M, rand));
    PetscCall(MatAssemblyBegin(*M, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*M, MAT_FINAL_ASSEMBLY));
    if (isdensecuda) PetscCall(MatConvert(*M, MATDENSECUDA, MAT_INPLACE_MATRIX, M));
    else if (isdensehip) PetscCall(MatConvert(*M, MATDENSEHIP, MAT_INPLACE_MATRIX, M));
  } else SETERRQ(comm, PETSC_ERR_USER, "Unsupported matrix type %s", type);
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat         A, P, C;
  MPI_Comm    comm;
  PetscInt    m = 10, n = 8;
  PetscRandom rand;
  PetscBool   flg, flg2, isdiag;
  const char *atype_default = MATDIAGONAL;
  const char *ptype_default = MATDIAGONAL;
  char        atype[256]    = "";
  char        ptype[256]    = "";

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscOptionsBegin(comm, "", help, "none");
  PetscCall(PetscOptionsFList("-amat_type", "A matrix type", "", MatList, atype_default, atype, 256, &flg));
  PetscCall(PetscOptionsFList("-pmat_type", "P matrix type", "", MatList, ptype_default, ptype, 256, &flg2));
  PetscCall(PetscOptionsInt("-m", "m size", "", m, &m, NULL));
  PetscCall(PetscOptionsInt("-n", "n size", "", n, &n, NULL));
  PetscOptionsEnd();

  if (!flg) PetscCall(PetscStrcpy(atype, atype_default));
  if (!flg2) PetscCall(PetscStrcpy(ptype, ptype_default));

  PetscCall(PetscStrcmp(ptype, MATDIAGONAL, &isdiag));
  if (isdiag) n = m;

  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(CreateTestMatrix(comm, atype, "adiag_", m, m, rand, &A));
  PetscCall(CreateTestMatrix(comm, ptype, "pdiag_", m, n, rand, &P));

  /* Initial PtAP */
  PetscCall(MatPtAP(A, P, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatPtAPMultEqual(A, P, C, 10, &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "MAT_INITIAL_MATRIX: MatPtAPMultEqual failed");

  /* Reuse with modified A */
  PetscCall(MatScale(A, 2.0));
  PetscCall(MatPtAP(A, P, MAT_REUSE_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatPtAPMultEqual(A, P, C, 10, &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "MAT_REUSE_MATRIX (modified A): MatPtAPMultEqual failed");

  /* Reuse with modified P */
  PetscCall(MatScale(P, 0.5));
  PetscCall(MatPtAP(A, P, MAT_REUSE_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatPtAPMultEqual(A, P, C, 10, &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "MAT_REUSE_MATRIX (modified P): MatPtAPMultEqual failed");

  /* Reuse with modified A */
  PetscCall(MatScale(A, 1.1));
  PetscCall(MatPtAP(A, P, MAT_REUSE_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatPtAPMultEqual(A, P, C, 10, &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "MAT_REUSE_MATRIX (modified A): MatPtAPMultEqual failed");

  /* Reuse with modified P */
  PetscCall(MatScale(P, 3.7));
  PetscCall(MatPtAP(A, P, MAT_REUSE_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatPtAPMultEqual(A, P, C, 10, &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "MAT_REUSE_MATRIX (modified P): MatPtAPMultEqual failed");

  /* Modify both A and P */
  PetscCall(MatScale(A, 0.23));
  PetscCall(MatScale(P, 1.43));
  PetscCall(MatPtAP(A, P, MAT_REUSE_MATRIX, PETSC_DETERMINE, &C));
  PetscCall(MatPtAPMultEqual(A, P, C, 10, &flg));
  PetscCheck(flg, comm, PETSC_ERR_PLIB, "MAT_REUSE_MATRIX (modified P): MatPtAPMultEqual failed");

  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&P));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: diag_diag
    nsize: {{1 2}}
    args: -amat_type diagonal -pmat_type diagonal
    output_file: output/empty.out

  test:
    suffix: diag_cpu
    nsize: {{1 2}}
    args: -amat_type diagonal -pmat_type {{aij dense}}
    output_file: output/empty.out

  test:
    suffix: cpu_diag
    nsize: {{1 2}}
    args: -amat_type {{aij dense}} -pmat_type diagonal
    output_file: output/empty.out

  # --- Kokkos: diagonal with kokkos matrix/vec types ---
  test:
    suffix: diag_diag_kokkos
    nsize: {{1 2}}
    requires: kokkos_kernels
    args: -amat_type diagonal -pmat_type diagonal -adiag_vec_type kokkos -pdiag_vec_type {{kokkos standard}}
    output_file: output/empty.out

  test:
    suffix: diag_standard_diag_kokkos
    nsize: {{1 2}}
    requires: kokkos_kernels
    args: -amat_type diagonal -pmat_type diagonal -adiag_vec_type standard -pdiag_vec_type kokkos
    output_file: output/empty.out

  test:
    suffix: diag_aijkokkos
    nsize: {{1 2}}
    requires: kokkos_kernels
    args: -amat_type diagonal -pmat_type aijkokkos -adiag_vec_type kokkos
    output_file: output/empty.out

  test:
    suffix: aijkokkos_diag
    nsize: {{1 2}}
    requires: kokkos_kernels
    args: -amat_type aijkokkos -pmat_type diagonal -pdiag_vec_type kokkos
    output_file: output/empty.out

  test:
    suffix: diag_diag_cuda
    nsize: {{1 2}}
    requires: cuda
    args: -amat_type diagonal -pmat_type diagonal -adiag_vec_type cuda -pdiag_vec_type {{cuda standard}}
    output_file: output/empty.out

  test:
    suffix: diag_standard_diag_cuda
    nsize: {{1 2}}
    requires: cuda
    args: -amat_type diagonal -pmat_type diagonal -adiag_vec_type standard -pdiag_vec_type cuda
    output_file: output/empty.out

  test:
    suffix: diag_cuda_mat
    nsize: {{1 2}}
    requires: cuda
    args: -amat_type diagonal -pmat_type {{aijcusparse densecuda}} -adiag_vec_type {{cuda standard}}
    output_file: output/empty.out

  test:
    suffix: cuda_mat_diag
    nsize: {{1 2}}
    requires: cuda
    args: -amat_type {{aijcusparse densecuda}} -pmat_type diagonal -pdiag_vec_type {{cuda standard}}
    output_file: output/empty.out

  test:
    suffix: diag_diag_hip
    nsize: {{1 2}}
    requires: hip
    args: -amat_type diagonal -pmat_type diagonal -adiag_vec_type hip -pdiag_vec_type {{hip standard}}
    output_file: output/empty.out

  test:
    suffix: diag_standard_diag_hip
    nsize: {{1 2}}
    requires: hip
    args: -amat_type diagonal -pmat_type diagonal -adiag_vec_type standard -pdiag_vec_type hip
    output_file: output/empty.out

  test:
    suffix: diag_hip_mat
    nsize: {{1 2}}
    requires: hip
    args: -amat_type diagonal -pmat_type {{aijhipsparse densehip}} -adiag_vec_type {{hip standard}}
    output_file: output/empty.out

  test:
    suffix: hip_mat_diag
    nsize: {{1 2}}
    requires: hip
    args: -amat_type {{aijhipsparse densehip}} -pmat_type diagonal -pdiag_vec_type {{hip standard}}
    output_file: output/empty.out

TEST*/
