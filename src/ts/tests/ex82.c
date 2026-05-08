static char help[] =
  "Test to check the PETSc machinery for the TS\n"
  "Explicit (RK):  ./ex82 -ts_monitor_error -ts_dt 0.1 -ts_type rk -ts_adapt_type none\n"
  "Implicit (BDF): ./ex82 -ts_monitor_error -ts_dt 0.01 -ts_type bdf -ts_bdf_order 2 -ts_adapt_type none -condition_system -complex_fd -pc_type none\n\n";

#include <petscdmshell.h>
#include <petscts.h>

typedef enum {
  TIME_INT_IMPLICIT = 0,
  TIME_INT_EXPLICIT
} TimeInt;

typedef struct {
  PetscBool     ts_implicit;
  PetscBool     condition_system;
  PetscBool     complex_fd;
  PetscReal     shift;
  PetscReal     epsilon;
  Vec           u_curr;
  PetscComplex *u_plus_iv;
  PetscComplex *complex_fun_eval;
} AppCtx;

static void TildeToUArray(PetscInt n, const PetscScalar u_tilde[], PetscScalar u[])
{
  PetscInt i;

  for (i = 0; i < n; ++i) u[i] = PetscExpScalar(u_tilde[i]);
}

static void UToTildeArray(PetscInt n, const PetscScalar u[], PetscScalar u_tilde[])
{
  PetscInt i;

  for (i = 0; i < n; ++i) u_tilde[i] = PetscLogScalar(u[i]);
}

static void TildeToUComplexArray(PetscInt n, const PetscComplex u_tilde[], PetscComplex u[])
{
  PetscInt i;

  for (i = 0; i < n; ++i) u[i] = PetscExpComplex(u_tilde[i]);
}

static void MyRHSArray(TimeInt time_int, const PetscScalar u[], PetscScalar f[])
{
  PetscScalar c_type = time_int == TIME_INT_EXPLICIT ? -1.0 : 1.0;

  f[0] = c_type * (u[0] - u[1]);
  f[1] = c_type * (u[1] - u[2]);
  f[2] = c_type * u[2];
}

static void MyRHSComplexArray(TimeInt time_int, const PetscComplex u[], PetscComplex f[])
{
  PetscComplex c_type = time_int == TIME_INT_EXPLICIT ? PetscCMPLX(-1.0, 0.0) : PetscCMPLX(1.0, 0.0);

  f[0] = c_type * (u[0] - u[1]);
  f[1] = c_type * (u[1] - u[2]);
  f[2] = c_type * u[2];
}

static PetscErrorCode TildeToU(Vec U_tilde, Vec U)
{
  PetscInt           n;
  const PetscScalar *u_tilde;
  PetscScalar       *u;

  PetscFunctionBeginUser;
  if (U_tilde == U) {
    PetscCall(VecExp(U));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecGetLocalSize(U_tilde, &n));
  PetscCheck(n == 3, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Expected a vector of length 3, got %" PetscInt_FMT, n);
  PetscCall(VecGetArrayRead(U_tilde, &u_tilde));
  PetscCall(VecGetArrayWrite(U, &u));
  TildeToUArray(n, u_tilde, u);
  PetscCall(VecRestoreArrayRead(U_tilde, &u_tilde));
  PetscCall(VecRestoreArrayWrite(U, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode UToTilde(Vec U, Vec U_tilde)
{
  PetscInt           n;
  const PetscScalar *u;
  PetscScalar       *u_tilde;

  PetscFunctionBeginUser;
  if (U == U_tilde) {
    PetscCall(VecLog(U));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(VecGetLocalSize(U, &n));
  PetscCheck(n == 3, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Expected a vector of length 3, got %" PetscInt_FMT, n);
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArrayWrite(U_tilde, &u_tilde));
  UToTildeArray(n, u, u_tilde);
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayWrite(U_tilde, &u_tilde));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PrimitiveToConservative(TS ts, Vec in, Vec out, PetscCtx ctx)
{
  PetscFunctionBeginUser;
  PetscCall(TildeToU(in, out));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec V_in, Vec V_out, PetscCtx ctx)
{
  PetscInt           n;
  const PetscScalar *u;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(V_in, &n));
  PetscCheck(n == 3, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Expected a vector of length 3, got %" PetscInt_FMT, n);
  PetscCall(VecGetArrayRead(V_in, &u));
  PetscCall(VecGetArrayWrite(V_out, &f));
  MyRHSArray(TIME_INT_EXPLICIT, u, f);
  PetscCall(VecRestoreArrayRead(V_in, &u));
  PetscCall(VecRestoreArrayWrite(V_out, &f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode Solution(TS ts, PetscReal t, Vec U, PetscCtx ctx)
{
  PetscScalar *u;
  PetscReal    et = PetscExpReal(-t);

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayWrite(U, &u));
  u[0] = (1.0 + t + 0.5 * t * t) * et;
  u[1] = (1.0 + t) * et;
  u[2] = et;
  PetscCall(VecRestoreArrayWrite(U, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SolutionTildeVar(TS ts, PetscReal t, Vec U, PetscCtx ctx)
{
  PetscFunctionBeginUser;
  PetscCall(Solution(ts, t, U, ctx));
  PetscCall(UToTilde(U, U));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComplexStepDeriv_Private(Mat J, Vec V, Vec W, PetscBool change_of_variable)
{
  AppCtx            *ctx;
  PetscInt           i, n;
  const PetscScalar *v;
  const PetscScalar *u;
  PetscScalar       *w;
  PetscReal          h, sigma;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(J, &ctx));
  PetscCall(VecGetLocalSize(V, &n));
  PetscCheck(n == 3, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Expected a vector of length 3, got %" PetscInt_FMT, n);
  PetscCall(VecGetArrayRead(V, &v));
  PetscCall(VecGetArrayRead(ctx->u_curr, &u));
  PetscCall(VecGetArrayWrite(W, &w));
  sigma = ctx->shift;
  h     = ctx->epsilon;
  for (i = 0; i < n; ++i) ctx->u_plus_iv[i] = PetscCMPLX(PetscRealPart(u[i]), PetscRealPart(v[i]) * h);
  if (change_of_variable) TildeToUComplexArray(n, ctx->u_plus_iv, ctx->u_plus_iv);
  MyRHSComplexArray(TIME_INT_EXPLICIT, ctx->u_plus_iv, ctx->complex_fun_eval);
  for (i = 0; i < n; ++i) w[i] = PetscImaginaryPart(ctx->complex_fun_eval[i] + sigma * ctx->u_plus_iv[i]) / h;
  PetscCall(VecRestoreArrayRead(V, &v));
  PetscCall(VecRestoreArrayRead(ctx->u_curr, &u));
  PetscCall(VecRestoreArrayWrite(W, &w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComplexStepDeriv_NoChangeOfVariable(Mat J, Vec V, Vec W)
{
  PetscFunctionBeginUser;
  PetscCall(ComplexStepDeriv_Private(J, V, W, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComplexStepDeriv_ChangeOfVariable(Mat J, Vec V, Vec W)
{
  PetscFunctionBeginUser;
  PetscCall(ComplexStepDeriv_Private(J, V, W, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PreJVEval(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal shift, Mat J, Mat P, PetscCtx ctx_void)
{
  AppCtx *ctx = (AppCtx *)ctx_void;

  PetscFunctionBeginUser;
  PetscCall(VecCopy(U, ctx->u_curr));
  ctx->shift = shift;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FunImplicit_Private(TS ts, PetscReal t, Vec U, Vec U_t, Vec F, PetscCtx ctx_void, PetscBool change_of_variable)
{
  AppCtx            *ctx = (AppCtx *)ctx_void;
  PetscInt           n;
  const PetscScalar *u;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  if (change_of_variable) {
    PetscCall(TildeToU(U, ctx->u_curr));
    PetscCall(VecGetArrayRead(ctx->u_curr, &u));
  }
  else PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetLocalSize(U, &n));
  PetscCheck(n == 3, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Expected a vector of length 3, got %" PetscInt_FMT, n);
  PetscCall(VecGetArrayWrite(F, &f));
  MyRHSArray(TIME_INT_IMPLICIT, u, f);
  PetscCall(VecRestoreArrayWrite(F, &f));
  if (change_of_variable) PetscCall(VecRestoreArrayRead(ctx->u_curr, &u));
  else PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecAXPY(F, 1.0, U_t));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FunImplicit_NoChangeOfVariable(TS ts, PetscReal t, Vec U, Vec U_t, Vec F, PetscCtx ctx_void)
{
  PetscFunctionBeginUser;
  PetscCall(FunImplicit_Private(ts, t, U, U_t, F, ctx_void, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FunImplicit_ChangeOfVariable(TS ts, PetscReal t, Vec U, Vec U_t, Vec F, PetscCtx ctx_void)
{
  PetscFunctionBeginUser;
  PetscCall(FunImplicit_Private(ts, t, U, U_t, F, ctx_void, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  TS          ts;
  Vec         u_sol;
  Vec         u_exact;
  Mat         j_shell = NULL;
  DM          dm;
  AppCtx      ctx     = {PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, 0.0, 1e-30, NULL, NULL, NULL};
  TSType      ts_type;
  PetscInt    prob_size = 3;
  PetscMPIInt size;
  PetscReal   final_time, error, tol;
  PetscBool   same;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCheck(!PetscDefined(USE_COMPLEX), PETSC_COMM_WORLD, PETSC_ERR_SUP, "This example requires a real-scalar PETSc build");
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Only for sequential runs");

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-condition_system", &ctx.condition_system, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-complex_fd", &ctx.complex_fd, NULL));

  PetscCall(VecCreateSeq(PETSC_COMM_SELF, prob_size, &u_sol));
  PetscCall(VecSet(u_sol, 1.0));
  PetscCall(VecDuplicate(u_sol, &ctx.u_curr));
  PetscCall(PetscMalloc2(prob_size, &ctx.u_plus_iv, prob_size, &ctx.complex_fun_eval));

  PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetTimeStep(ts, 0.01));
  PetscCall(TSSetMaxTime(ts, 0.1));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetSolutionFunction(ts, Solution, &ctx));
  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSGetType(ts, &ts_type));
  PetscCall(PetscStrcmp(ts_type, TSBDF, &same));
  if (same) ctx.ts_implicit = PETSC_TRUE;
  PetscCall(PetscStrcmp(ts_type, TSTHETA, &same));
  if (same) ctx.ts_implicit = PETSC_TRUE;
  PetscCall(PetscStrcmp(ts_type, TSALPHA, &same));
  if (same) ctx.ts_implicit = PETSC_TRUE;
  PetscCall(PetscStrcmp(ts_type, TSBEULER, &same));
  if (same) ctx.ts_implicit = PETSC_TRUE;
  PetscCall(PetscStrcmp(ts_type, TSCN, &same));
  if (same) ctx.ts_implicit = PETSC_TRUE;
  PetscCall(PetscStrcmp(ts_type, TSARKIMEX, &same));
  if (same) ctx.ts_implicit = PETSC_TRUE;
  PetscCall(PetscStrcmp(ts_type, TSROSW, &same));
  if (same) ctx.ts_implicit = PETSC_TRUE;

  if (ctx.ts_implicit) {
    PetscCall(TSSetIFunction(ts, NULL, ctx.condition_system ? FunImplicit_ChangeOfVariable : FunImplicit_NoChangeOfVariable, &ctx));
    PetscCall(TSGetDM(ts, &dm));
    PetscCall(DMShellSetGlobalVector(dm, u_sol));

    if (ctx.complex_fd) {
      PetscCall(MatCreateShell(PETSC_COMM_SELF, prob_size, prob_size, prob_size, prob_size, &ctx, &j_shell));
      if (ctx.condition_system) PetscCall(MatShellSetOperation(j_shell, MATOP_MULT, (PetscErrorCodeFn *)ComplexStepDeriv_ChangeOfVariable));
      else PetscCall(MatShellSetOperation(j_shell, MATOP_MULT, (PetscErrorCodeFn *)ComplexStepDeriv_NoChangeOfVariable));
      PetscCall(TSSetIJacobian(ts, j_shell, NULL, PreJVEval, &ctx));
    }

    if (ctx.condition_system) {
      PetscCall(TSSetTransientVariable(ts, PrimitiveToConservative, &ctx));
      PetscCall(UToTilde(u_sol, u_sol));
      PetscCall(TSSetSolutionFunction(ts, SolutionTildeVar, &ctx));
    }
  } else {
    PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, &ctx));
    if (ctx.complex_fd || ctx.condition_system) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: complex_fd and condition_system options are ignored in explicit stepping.\n"));
  }

  PetscCall(TSSetSolution(ts, u_sol));
  PetscCall(TSSetUp(ts));
  PetscCall(TSSolve(ts, u_sol));

  if (ctx.condition_system) PetscCall(TildeToU(u_sol, u_sol));

  PetscCall(TSGetTime(ts, &final_time));
  PetscCall(VecDuplicate(u_sol, &u_exact));
  PetscCall(Solution(ts, final_time, u_exact, &ctx));
  PetscCall(VecAXPY(u_exact, -1.0, u_sol));
  PetscCall(VecNorm(u_exact, NORM_2, &error));
  tol = ctx.ts_implicit ? 1e-3 : 1e-6;
  PetscCheck(error < tol, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Solution error %g exceeds tolerance %g", (double)error, (double)tol);
  PetscCall(VecDestroy(&u_exact));

  PetscCall(MatDestroy(&j_shell));
  PetscCall(VecDestroy(&ctx.u_curr));
  PetscCall(VecDestroy(&u_sol));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFree2(ctx.u_plus_iv, ctx.complex_fun_eval));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  test:
    suffix: explicit
    args: -ts_monitor_error -ts_dt 0.1 -ts_type rk -ts_adapt_type none -pc_type none
    requires: !complex
    filter: sed -e '/^2-norm of error /d'
    output_file: output/empty.out

  test:
    suffix: implicit
    args: -ts_monitor_error -ts_dt 0.01 -ts_type bdf -ts_bdf_order 2 -ts_adapt_type none
    requires: !complex
    filter: sed -e '/^2-norm of error /d'
    output_file: output/empty.out
TEST*/
