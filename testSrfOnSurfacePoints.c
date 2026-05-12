#include <petsc.h>
#include <petsc/private/dmpleximpl.h>
#include "constants.h"
#include "surface.h"

typedef enum {BEM_POINT, BEM_PANEL, BEM_POINT_MF, BEM_PANEL_MF} BEMType;
static const char *const BEMTypes[] = {"point", "panel", "point_mf", "panel_mf", "BEMType", "BEM_", 0};

typedef enum {PROBLEM_SPHERE, PROBLEM_SPHERE_CP, PROBLEM_MOLECULE, NUM_PORBLEMS} ProblemType;
static const char *const ProblemTypes[] = {"sphere", "sphere_cp", "molecule", "ProblemType", "PROBLEM_", 0};

/* Performance characterization */
PetscLogEvent CalcE_Event, CalcL_Event, CalcR_Event, CalcStoQ_Event, CalcStoS_Event, IntegratePanel_Event;

typedef struct {
  Vec q;   /* Charge values */
  Vec xyz; /* Charge coordinates, always 3D */
  Vec R;   /* Charge radii */
} PQRData;

typedef struct {
  Vec          w;       /* The vertex weights */
  Mat          Kp;      /* The surface-to-surface operator (1/2 epshat I - K') */
  PetscReal    epsHat;  /* (eps_2 - eps_1) / (eps_2 + eps_1) */
  PetscReal    alpha;   /* Size of charge-hydration asymmetry */
  PetscReal    beta;    /* Slope for field dependence of asymmetry (width of transition zone) */
  PetscReal    gamma;   /* Transition field strength */
  PetscReal    mu;      /* Assures t(0) = 0, so mu should be −alpha tanh(−gamma) */
  PetscReal    damping; /* Damping coefficient for Jacobian, 0.0 indicates the Picard iteration */
  PetscBool    picard;  /* Use the Picard iteration */
} SLICCtx;

typedef struct {
  /* Model parameters */
  ProblemType probType;   /* Indicates the type of test, e.g. sphere */
  PetscBool   useSLIC;    /* Flag to use the Solvation Layer Interface Condition */
  SLICCtx     slicCtx;    /* Context with SLIC parameters */
  /* Physical parameters */
  PetscReal   epsIn;      /* solute dielectric coefficient */
  PetscReal   epsOut;     /* solvent dielectric coefficient */
  char        pdbFile[PETSC_MAX_PATH_LEN]; /* Chemists are crazy and have never heard of normalized data */
  char        crgFile[PETSC_MAX_PATH_LEN];
  /* Surface file */
  PetscInt    srfNum;     /* Resolution of mesh file */
  char        basename[PETSC_MAX_PATH_LEN];
  char        srfFile[PETSC_MAX_PATH_LEN];
  char        pntFile[PETSC_MAX_PATH_LEN];
  /* Point BEM parameters */
  PetscReal   density;    /* Density of points on surface */
  /* Sphere setup */
  PetscReal   R;          /* Sphere radius */
  PetscReal   origin[3];  /* Sphere center */
  PetscInt    numCharges; /* Number of atomic charges in the solute */
  PetscReal   h;          /* Charge spacing */
  /* Analytical parameters */
  PetscInt    Nmax;       /* Order of the multipole expansion */
} SolvationContext;

/* . En - The normal electric field at this point */
static PetscReal SLIC_t(PetscReal En, void *user)
{
  SLICCtx        *ctx   = (SLICCtx *) user;
  const PetscReal alpha = ctx->alpha;
  const PetscReal beta  = ctx->beta;
  const PetscReal gamma = ctx->gamma;
  const PetscReal mu    = ctx->mu;

  return alpha * PetscTanhReal(beta*En - gamma) + mu;
}

/* . En - The normal electric field at this point */
static PetscReal SLIC_dtdx(PetscReal En, void *user)
{
  SLICCtx        *ctx   = (SLICCtx *) user;
  const PetscReal alpha = ctx->alpha;
  const PetscReal beta  = ctx->beta;
  const PetscReal gamma = ctx->gamma;

  return alpha * beta * PetscSqr(1.0/PetscCoshReal(beta*En - gamma));
}

PetscErrorCode ProcessOptions(MPI_Comm comm, SolvationContext *ctx)
{
  PetscFunctionBeginUser;
  ctx->probType   = PROBLEM_SPHERE;
  ctx->useSLIC    = PETSC_FALSE;
  ctx->epsIn      = 4;
  ctx->epsOut     = 80;
  ctx->srfNum     = 1;
  ctx->R          = 6.0;
  ctx->origin[0]  = 0.0;
  ctx->origin[1]  = 0.0;
  ctx->origin[2]  = 0.0;
  ctx->numCharges = 100;
  ctx->h          = 1.0;
  ctx->Nmax       = 100;
  ctx->density    = 1.0;

  /* The SLIC defaults come from "A Simple Electrostatic Model for the Hard-Sphere Solute Component of Nonpolar Solvation", Cooper and Bardhan, 2020.

    alpha =   0.898
    beta  = −30.476
    gamma =  −0.151
    mu    =  −0.449
    phi   =   0.095
  */
  ctx->slicCtx.alpha   =   0.898;
  ctx->slicCtx.beta    = -30.476;
  ctx->slicCtx.gamma   =  -0.151;
  ctx->slicCtx.mu      = -SLIC_t(0.0, &ctx->slicCtx);
  ctx->slicCtx.damping = 1.0;
  ctx->slicCtx.picard  = PETSC_FALSE;

  PetscCall(PetscStrcpy(ctx->basename, "../../jay-pointbem/geometry/sphere_R6_vdens"));
  PetscOptionsBegin(comm, "", "Solvation Problem Options", "BIBEE");
    PetscCall(PetscOptionsEnum("-prob_type", "Type of test case", "testSrfOnSurfacePoints", ProblemTypes, (PetscEnum) ctx->probType, (PetscEnum *) &ctx->probType, NULL));
    PetscCall(PetscOptionsBool("-slic", "Use the SLIC model", "testSrfOnSurfacePoints", ctx->useSLIC, &ctx->useSLIC, NULL));
    PetscCall(PetscOptionsReal("-slic_alpha", "The size of charge-hydration asymmetry", "testSrfOnSurfacePoints", ctx->slicCtx.alpha, &ctx->slicCtx.alpha, NULL));
    PetscCall(PetscOptionsReal("-slic_beta", "The width of the transition zone", "testSrfOnSurfacePoints", ctx->slicCtx.beta, &ctx->slicCtx.beta, NULL));
    PetscCall(PetscOptionsReal("-slic_gamma", "The transition field strength", "testSrfOnSurfacePoints", ctx->slicCtx.gamma, &ctx->slicCtx.gamma, NULL));
    PetscCall(PetscOptionsReal("-slic_mu", "The \\mu SLIC parameter", "testSrfOnSurfacePoints", ctx->slicCtx.mu, &ctx->slicCtx.mu, NULL));
    PetscCall(PetscOptionsReal("-slic_damping", "The damping coefficient for the SLIC Jacobian, 0.0 indicates the Picard iteration", "testSrfOnSurfacePoints", ctx->slicCtx.damping, &ctx->slicCtx.damping, NULL));
    PetscCall(PetscOptionsBool("-slic_picard", "Use the Picard iteration for SLIC solution", "testSrfOnSurfacePoints", ctx->slicCtx.picard, &ctx->slicCtx.picard, NULL));
    PetscCall(PetscOptionsReal("-epsilon_solute", "The dielectric coefficient of the solute", "testSrfOnSurfacePoints", ctx->epsIn, &ctx->epsIn, NULL));
    PetscCall(PetscOptionsReal("-epsilon_solvent", "The dielectric coefficient of the solvent", "testSrfOnSurfacePoints", ctx->epsOut, &ctx->epsOut, NULL));
    PetscCall(PetscOptionsString("-pdb_filename", "The filename for the .pdb file", "testSrfOnSurfacePoints", ctx->pdbFile, ctx->pdbFile, sizeof(ctx->pdbFile), NULL));
    PetscCall(PetscOptionsString("-crg_filename", "The filename for the .crg file", "testSrfOnSurfacePoints", ctx->crgFile, ctx->crgFile, sizeof(ctx->crgFile), NULL));
    PetscCall(PetscOptionsInt("-num_charges", "The number of atomic charges in the solute", "testSrfOnSurfacePoints", ctx->numCharges, &ctx->numCharges, NULL));
    PetscCall(PetscOptionsString("-srf_base", "The basename for the .srf file", "testSrfOnSurfacePoints", ctx->basename, ctx->basename, sizeof(ctx->basename), NULL));
    PetscCall(PetscOptionsInt("-srf_num", "The resolution number of the mesh", "testSrfOnSurfacePoints", ctx->srfNum, &ctx->srfNum, NULL));
    PetscCall(PetscOptionsInt("-nmax", "The order of the multipole expansion", "testSrfOnSurfacePoints", ctx->Nmax, &ctx->Nmax, NULL));
    PetscCall(PetscOptionsReal("-density", "The density of points for BEM", "testSrfOnSurfacePoints", ctx->density, &ctx->density, NULL));
  PetscOptionsEnd();

  PetscCall(PetscSNPrintf(ctx->srfFile, PETSC_MAX_PATH_LEN-1, "%s%d.srf", ctx->basename, (int) ctx->srfNum));
  PetscCall(PetscSNPrintf(ctx->pntFile, PETSC_MAX_PATH_LEN-1, "%s%d.pnt", ctx->basename, (int) ctx->srfNum));

  PetscCall(PetscLogEventRegister("IntegratePanel",   DM_CLASSID, &IntegratePanel_Event));
  PetscCall(PetscLogEventRegister("CalcSurfToSurf",   DM_CLASSID, &CalcStoS_Event));
  PetscCall(PetscLogEventRegister("CalcSurfToCharge", DM_CLASSID, &CalcStoQ_Event));
  PetscCall(PetscLogEventRegister("CalcLMatrix",      DM_CLASSID, &CalcL_Event));
  PetscCall(PetscLogEventRegister("CalcReactPot",     DM_CLASSID, &CalcR_Event));
  PetscCall(PetscLogEventRegister("CalcSolvEnergy",   DM_CLASSID, &CalcE_Event));
  PetscFunctionReturn(0);
}

PetscErrorCode PQRCreateFromPDB(MPI_Comm comm, const char pdbFile[], const char crgFile[], PQRData *pqr)
{
  PetscViewer    viewerPDB, viewerCRG;
  PetscScalar   *q, *x;
  PetscReal     *charges, *coords;
  PetscInt       n = 0, i, d;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscCall(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscViewerCreate(comm, &viewerPDB));
  PetscCall(PetscViewerSetType(viewerPDB, PETSCVIEWERASCII));
  PetscCall(PetscViewerFileSetMode(viewerPDB, FILE_MODE_READ));
  PetscCall(PetscViewerFileSetName(viewerPDB, pdbFile));
  if (!rank) {
    char     buf[128];
    PetscInt line = 0, maxSize = 1024, cnt = 1;

    PetscCall(PetscMalloc2(maxSize, &charges, maxSize*3, &coords));
    while (cnt) {
      PetscInt c = 0;

      /* Read line */
      do {PetscCall(PetscViewerRead(viewerPDB, &buf[c++], 1, &cnt, PETSC_CHAR));}
      while (buf[c-1] != '\n' && buf[c-1] != '\0' && cnt);
      /* Parse line */
      if (c > 6 &&
          ((buf[0] == 'A' && buf[1] == 'T' && buf[2] == 'O' && buf[3] == 'M') ||
           (buf[0] == 'H' && buf[1] == 'E' && buf[2] == 'T' && buf[3] == 'A' && buf[4] == 'T' && buf[5] == 'M'))) {
        double tmp;

        if (n >= maxSize) {
          /* Reallocate and copy */
        }
        buf[66] = '\0';
        ierr = sscanf(&buf[60], "%lg", &tmp); if (ierr != 1) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Could not read charge for line %d of PDB file %s", line, pdbFile);
        charges[n] = tmp;
        buf[54] = '\0';
        ierr = sscanf(&buf[46], "%lg", &tmp); if (ierr != 1) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Could not read z coordinate for line %d of PDB file %s", line, pdbFile);
        coords[n*3+2] = tmp;
        buf[46] = '\0';
        ierr = sscanf(&buf[38], "%lg", &tmp); if (ierr != 1) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Could not read y coordinate for line %d of PDB file %s", line, pdbFile);
        coords[n*3+1] = tmp;
        buf[38] = '\0';
        ierr = sscanf(&buf[31], "%lg", &tmp); if (ierr != 1) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Could not read x coordinate for line %d of PDB file %s", line, pdbFile);
        coords[n*3+0] = tmp;
        /* Residue id [23-27] */
        /* Segment id [21-22] */
        /* Residue name [17-20] */
        /* Atom name [12-15] */
        ++n;
      }
      ++line;
    }
  }
  PetscCall(PetscViewerDestroy(&viewerPDB));

  PetscCall(VecCreate(comm, &pqr->q));
  PetscCall(VecSetSizes(pqr->q, n, PETSC_DETERMINE));
  PetscCall(PetscObjectSetName((PetscObject) pqr->q, "Atomic Charges"));
  PetscCall(VecSetFromOptions(pqr->q));
  PetscCall(VecCreate(comm, &pqr->xyz));
  PetscCall(VecSetSizes(pqr->xyz, n*3, PETSC_DETERMINE));
  PetscCall(PetscObjectSetName((PetscObject) pqr->xyz, "Atomic XYZ"));
  PetscCall(VecSetBlockSize(pqr->xyz, 3));
  PetscCall(VecSetFromOptions(pqr->xyz));

  PetscCall(VecGetArray(pqr->q, &q));
  PetscCall(VecGetArray(pqr->xyz, &x));
  for (i = 0; i < n; ++i) {
    q[i] = charges[i];
    for (d = 0; d < 3; ++d) x[i*3+d] = coords[i*3+d];
  }
  PetscCall(VecRestoreArray(pqr->q, &q));
  PetscCall(VecRestoreArray(pqr->xyz, &x));
  PetscCall(PetscFree2(charges, coords));

  if (crgFile) {
    PetscCall(PetscViewerCreate(comm, &viewerCRG));
    PetscCall(PetscViewerSetType(viewerCRG, PETSCVIEWERASCII));
    PetscCall(PetscViewerFileSetMode(viewerCRG, FILE_MODE_READ));
    PetscCall(PetscViewerFileSetName(viewerCRG, crgFile));
    if (!rank) {
      char     buf[128];
      PetscInt cnt = 1;

      /* The CRG file is required to have the same nubmer of atoms in the same order as the PDB */
      PetscCall(VecGetArray(pqr->q, &q));
      for (i = -1; i < n; ++i) {
        double    tmp;
        PetscInt  c = 0;

        /* Read line */
        do {PetscCall(PetscViewerRead(viewerCRG, &buf[c++], 1, &cnt, PETSC_CHAR));}
        while (buf[c-1] != '\n' && buf[c-1] != '\0' && cnt);
        if (!cnt) break;
        if (i < 0) continue;
        buf[22] = '\0';
        ierr = sscanf(&buf[14], "%lg", &tmp); if (ierr != 1) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Could not read charge for line %" PetscInt_FMT " of CRG file %s", i+1, crgFile);
        q[i] = tmp;
        /* Segment id [13] */
        /* Residue number [9-12] */
        /* Residue name [6-8] */
        /* Atom name [0-5] */
      }
      PetscCall(VecRestoreArray(pqr->q, &q));
    }
    PetscCall(PetscViewerDestroy(&viewerCRG));
  }

  PetscCall(VecDuplicate(pqr->q, &pqr->R));
  PetscCall(PetscObjectSetName((PetscObject) pqr->R, "Atomic radii"));
  PetscCall(VecSet(pqr->R, 0.0));
  PetscFunctionReturn(0);
}

PetscErrorCode PQRViewFromOptions(PQRData *pqr)
{
  PetscFunctionBeginUser;
  PetscCall(VecViewFromOptions(pqr->xyz, NULL, "-pqr_vec_view"));
  PetscCall(VecViewFromOptions(pqr->q,   NULL, "-pqr_vec_view"));
  PetscCall(VecViewFromOptions(pqr->R,   NULL, "-pqr_vec_view"));
  PetscFunctionReturn(0);
}

PetscErrorCode PQRDestroy(PQRData *pqr)
{
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&pqr->xyz));
  PetscCall(VecDestroy(&pqr->q));
  PetscCall(VecDestroy(&pqr->R));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode factorial(PetscInt n, PetscReal *fact)
{
  PetscReal f = 1.0;
  PetscInt  i;

  PetscFunctionBeginUser;
  for (i = 2; i <= n; ++i) f *= i;
  *fact = f;
  PetscFunctionReturn(0);
}

static inline PetscErrorCode convertToSpherical(const PetscScalar xyz[], PetscScalar r[])
{
  const PetscReal x = PetscRealPart(xyz[0]);
  const PetscReal y = PetscRealPart(xyz[1]);
  const PetscReal z = PetscRealPart(xyz[2]);

  PetscFunctionBeginUser;
  r[0] = PetscSqrtReal(x*x + y*y + z*z);
  r[1] = atan2(y, x);
  r[2] = 0.0;
  if (PetscAbsReal(r[0]) > 0.0) r[2] = acos(z/r[0]);
  PetscFunctionReturn(0);
}

/*@
  legendre - Calculate the Associated Legendre polynomial P^l_m(x).

  Input Parameters:
+ l - The order
. m - The suborder
. x - The argument

  Output Parameter:
. leg - An array of the associated Legendre polynomials evaluated at the points x

  Note: Speed is obtained by direct calculation of polynomial coefficients rather than recursion.
  Polynomial coefficients can increase in magnitude very quickly with polynomial degree,
  leading to decreased accuracy (estimated by err). If you need higher degrees for the polynomials,
  use recursion-based algorithms.

  Level: intermediate

.seealso:
@*/
PetscErrorCode legendre(PetscInt l, PetscInt m, PetscScalar x, PetscScalar *leg, PetscScalar *err)
{
  /* The error estimate is based on worst case scenario and the significant digits, and thus
     based on the largest polynomial coefficient and machine error, "eps" */
  PetscInt  maxcf;    /* largest polynomial coefficient */
  PetscReal cfnm = 1; /* proportionality constant for m < 0 polynomials compared to m > 0 */
  PetscReal cl, x2, p, f1, f2, f3;
  PetscInt  px, j;

  PetscFunctionBeginUser;
  /* The polynomials are not defined for |x| > 1 */
  PetscCheck(PetscAbsScalar(x) <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid input |x %g| > 1", PetscRealPart(x));
  /* Could also define this to be 0 */
  PetscCheck(PetscAbsInt(m) <= PetscAbsInt(l), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid input m %d > l %d", m, l);
  if (l < 0) l = -(l+1);
  if (m < 0) {
    PetscReal num, den;

    m    = -m;
    PetscCall(factorial(l-m, &num));
    PetscCall(factorial(l+m, &den));
    cfnm = PetscPowInt(-1, m)*num/den;
  }
  /* Calculate coef of maximum degree in x from the explicit analytical formula */
  PetscCall(factorial(2*l, &f1));
  PetscCall(factorial(l,   &f2));
  PetscCall(factorial(l-m, &f3));
  cl    = PetscPowInt(-1, m) * cfnm * f1/((1 << l)*f2*f3);
  maxcf = PetscAbsInt(cl);
  px    = l-m;
  /* Power of x changes from one term to the next by 2. Also needed for sqrt(1-x^2). */
  x2    = x*x; /* TODO make pointwise square */
  /* Calculate efficiently P_l^m (x)/sqrt(1-x^2)^(m/2) - that is, only the polynomial part.
     At least one coefficient is guaranteed to exist - there is no null Legendre polynomial. */
  p     = cl; /* TODO make an array of cl */
  for (j = l-1; j >= 0; --j) {
    /* Check the exponent of x for current coefficient, px. If it is 0 or 1, just exit the loop */
    if (px < 2) break;
    /* If current exponent is >=2, there is a "next" coefficient; multiply p by x2 and add it. Calculate the current coefficient */
    cl = -(j+j+2-l-m)*(j+j+1-l-m)/(2*(j+j+1)*(l-j))*cl;

    if (maxcf < PetscAbsReal(cl)) maxcf = PetscAbsReal(cl);
    /* ...and add to the polynomial */
    p = p*x2 + cl; /* TODO make this pointwise multiply */
    /* Decrease the exponent of x - this is the exponent of x corresponding to the newly added coefficient */
    px -= 2;
  }
  /* Estimate the error */
  if (err) *err = maxcf*PETSC_MACHINE_EPSILON;

  /* Now we're done adding coefficients. However, if the exponent of x
     corresponding to the last added coefficient is 1 (polynomial is odd),
     multiply the polynomial by x */
  if (px == 1) p = p*x;

  /* All that's left is to multiply the whole thing with sqrt(1-x^2)^(m/2). No further calculations are needed if m = 0. */
  if (m == 0) {*leg = p; PetscFunctionReturn(0);}

  x2 = 1-x2;
  /* First, multiply by the integer part of m/2 */
  for (j = 1; j < PetscFloorReal(m/2.0); ++j) p = p*x2; /* TODO make this pointwise multiply */
  /* If m is odd, there is an additional factor sqrt(1-x^2) */
  if (m % 2) p = p*PetscSqrtReal(x2); /* TODO make this pointwise multiply */
  *leg = p;
  PetscFunctionReturn(0);
}

/*
  This is code to compute P^m_n(z) = (-1)^m (1 - z^2)^{m/2} \frac{d^m P_n(z)}{dz^m}

  leg is an arry of length nz*(l+1)

  Note: http://www.accefyn.org.co/revista/Vol_37/145/541-544.pdf
*/
PetscErrorCode legendre2(PetscInt l, PetscInt nz, PetscScalar z, PetscScalar leg[])
{
  PetscReal sqz2   = PetscSqrtReal(1.0 - PetscSqr(PetscRealPart(z)));
  PetscReal hsqz2  = 0.5*sqz2;
  PetscReal ihsqz2 = PetscRealPart(z)/hsqz2;
  PetscReal fac    = 1.0;
  PetscInt  pre    = l % 2 ? -1 : 1;
  PetscInt  m;

  PetscFunctionBeginUser;
  for (m = 2; m <= l; ++m) fac *= m;
  if (!l) {
    leg[0] = 1.0;
  } else if (l == 1) {
    leg[0] = -hsqz2;
    leg[1] = z;
    leg[2] = sqz2;
  } else {
    leg[0] = (1.0 - 2.0*PetscAbsReal(l - 2.0*PetscFloorReal(l/2.0)))*PetscPowReal(hsqz2, l)/fac;
    leg[1] = -leg[0]*l*ihsqz2;
    for (m = 1; m < 2*l; ++m) leg[m+1] = (m - l)*ihsqz2*leg[m] - (2*l - m + 1)*m*leg[m-1];
  }
  for (m = 0; m <= 2*l; ++m, pre = -pre) leg[m] *= pre;
  PetscFunctionReturn(0);
}

/*
  Select a set of point charges from a grid with spacing dx which are inside sphere of radius R, and delta away from the surface
*/
PetscErrorCode makeSphereChargeDistribution(PetscReal R, PetscInt numCharges, PetscReal dx, PetscReal delta, PQRData *data)
{
  PetscRandom     rand;
  const PetscReal maxChargeValue = 0.85;
  PetscInt        numPoints      = 0, *select, c;
  PetscReal       x, y, z;

  PetscFunctionBeginUser;
  {
    PetscReal vals[8];
    PetscInt  nmax = 8, i;
    PetscBool flg;

    PetscCall(PetscOptionsGetRealArray(NULL, NULL, "-test", vals, &nmax, &flg));
    if (flg) {
      numCharges = nmax/4;
      PetscCall(VecCreate(PETSC_COMM_WORLD, &data->q));
      PetscCall(VecSetSizes(data->q, numCharges, PETSC_DETERMINE));
      PetscCall(PetscObjectSetName((PetscObject) data->q, "Atomic Charges"));
      PetscCall(VecSetFromOptions(data->q));
      PetscCall(VecCreate(PETSC_COMM_WORLD, &data->xyz));
      PetscCall(VecSetSizes(data->xyz, numCharges*3, PETSC_DETERMINE));
      PetscCall(PetscObjectSetName((PetscObject) data->xyz, "Atomic XYZ"));
      PetscCall(VecSetBlockSize(data->xyz, 3));
      PetscCall(VecSetFromOptions(data->xyz));
      PetscCall(VecDuplicate(data->q, &data->R));
      PetscCall(PetscObjectSetName((PetscObject) data->R, "Atomic radii"));
      PetscCall(VecSet(data->R, 0.0));
      for (i = 0; i < numCharges; ++i) {
        PetscCall(VecSetValues(data->q, 1, &i, &vals[i*4], INSERT_VALUES));
        PetscCall(VecSetValuesBlocked(data->xyz, 1, &i, &vals[i*4+1], INSERT_VALUES));
      }
      PetscFunctionReturn(0);
    }
  }
  if (delta < 0.0) delta = dx;

  /* Form a grid of points [-R, R]^3 with spacing dx */
  for (z = -R; z < R; z += dx) {
    for (y = -R; y < R; y += dx) {
      for (x = -R; x < R; x += dx) {
        const PetscReal dist = sqrt(x*x + y*y + z*z);

        if (dist < R - delta) ++numPoints;
      }
    }
  }

  PetscCall(VecCreate(PETSC_COMM_WORLD, &data->q));
  PetscCall(VecSetSizes(data->q, numCharges, PETSC_DETERMINE));
  PetscCall(PetscObjectSetName((PetscObject) data->q, "Atomic Charges"));
  PetscCall(VecSetFromOptions(data->q));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &data->xyz));
  PetscCall(VecSetSizes(data->xyz, numCharges*3, PETSC_DETERMINE));
  PetscCall(PetscObjectSetName((PetscObject) data->xyz, "Atomic XYZ"));
  PetscCall(VecSetBlockSize(data->xyz, 3));
  PetscCall(VecSetFromOptions(data->xyz));

  PetscCall(PetscCalloc1(numPoints, &select));
  if ((numCharges >= 0) && (numCharges < numPoints)) {
    PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
    PetscCall(PetscRandomSetFromOptions(rand));
    PetscCall(PetscRandomSetInterval(rand, 0, numPoints));
    for (c = 0; c < numCharges; ++c) {
      PetscCall(PetscRandomGetValueReal(rand, &x));
      if (select[(PetscInt) PetscFloorReal(x)]) --c;
      select[(PetscInt) PetscFloorReal(x)] = 1;
    }
    PetscCall(PetscRandomDestroy(&rand));
  }

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(PetscRandomSetInterval(rand, -maxChargeValue, maxChargeValue));
  numPoints = 0; c = 0;
  for (z = -R; z < R; z += dx) {
    for (y = -R; y < R; y += dx) {
      for (x = -R; x < R; x += dx) {
        const PetscReal dist   = sqrt(x*x + y*y + z*z);
        PetscReal       pos[3] = {x, y, z}, q;

        if (dist < R - delta) {
          if (select[numPoints]) {
            PetscCall(PetscRandomGetValueReal(rand, &q));
            PetscCall(VecSetValues(data->q, 1, &c, &q, INSERT_VALUES));
            PetscCall(VecSetValuesBlocked(data->xyz, 1, &c, pos, INSERT_VALUES));
            ++c;
          }
          ++numPoints;
        }
      }
    }
  }
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFree(select));

  PetscCall(VecDuplicate(data->q, &data->R));
  PetscCall(PetscObjectSetName((PetscObject) data->R, "Atomic radii"));
  PetscCall(VecSet(data->R, 0.0));
  PetscFunctionReturn(0);
}

/*@
  computeEnm - Compute the multipole coefficients for the protein charges

  Input Parameters:
+ b - the sphere radius, in Angstroms
. epsIn - the dielectric constant inside the protein
. pqrData - the PQRData context
. qVec - The charge vector to use instead of the pqrData vector
- Nmax - the maximum multipole order to use

  Output Parameters:
. Enm - The vector of multipole coefficients (packed real part, imaginary part)

  Level: beginner

.seealso: computeBnm(), doAnalytical()
@*/
PetscErrorCode computeEnm(PetscReal b, PetscReal epsIn, PQRData *pqr, Vec qVec, PetscInt Nmax, Vec Enm)
{
  PetscScalar *xyz, *q;
  PetscReal   *P;
  PetscInt     Nq, n, m, k, idx;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc1(2*Nmax+1,&P));
  PetscCall(VecGetLocalSize(qVec, &Nq));
  PetscCall(VecSet(Enm, 0.0));
  PetscCall(VecGetArray(pqr->xyz, &xyz));
  PetscCall(VecGetArray(qVec, &q));
  for (k = 0; k < Nq; ++k) {
    PetscScalar r[3], val;

    PetscCall(convertToSpherical(&xyz[k*3], r));
    for (n = 0, idx = 0; n <= Nmax; ++n) {
      PetscCall(legendre2(n, 1, cos(r[2]), P));
      for (m = -n ; m <= n; ++m, ++idx) {
        const PetscReal Pnm  = P[PetscAbsInt(m)+n];
        PetscReal       ff; /* (n - |m|)! / (n + |m|)! */
        PetscReal       num, den;

        //PetscCall(PetscPrintf(PETSC_COMM_SELF, "Charge %d P(%d, |%d|) %g\n", k, n, m, Pnm));
        PetscCall(factorial(n - PetscAbsInt(m), &num));
        PetscCall(factorial(n + PetscAbsInt(m), &den));
        ff   = num/den;
		val  = ff * q[k] * PetscPowScalar(r[0], n) * Pnm;
        PetscCall(VecSetValue(Enm, idx*2+0,  val*cos(m*r[1]), ADD_VALUES));
        PetscCall(VecSetValue(Enm, idx*2+1, -val*sin(m*r[1]), ADD_VALUES));
      }
    }
  }
  PetscCall(VecRestoreArray(pqr->xyz, &xyz));
  PetscCall(VecRestoreArray(qVec, &q));
  PetscCall(PetscFree(P));
  PetscFunctionReturn(0);
}

/*@
  computeBnm - Compute the multipole coefficients for the sphere interior (reaction potential)

  Input Parameters:
+ b - the sphere radius, in Angstroms
. epsIn - the dielectric constant inside the protein
. epsOut - the dielectric constant outside the protein
. Nmax - the maximum multipole order to use
- Enm - The vector of multipole coefficients for the charge field

  Output Parameters:
. Bnm - The vector of multipole coefficients (packed real part, imaginary part)

  Level: beginner

.seealso: computeEnm(), doAnalytical()
@*/
PetscErrorCode computeBnm(PetscReal b, PetscReal epsIn, PetscReal epsOut, PetscInt Nmax, Vec Enm, Vec Bnm)
{
#if 0
  PetscReal    epsHat = 2.0*(epsIn - epsOut)/(epsIn + epsOut);
#endif
  PetscScalar *bnm;
  PetscInt     n, m, idx;

  PetscFunctionBeginUser;
  PetscCall(VecCopy(Enm, Bnm));
  PetscCall(VecGetArray(Bnm, &bnm));
  for (n = 0, idx = 0; n <= Nmax; ++n) {
    const PetscReal val = ((epsIn - epsOut)*(n+1))/(epsIn * (n*epsIn + (n+1)*epsOut)) * (1.0/PetscPowReal(b, (2*n+1)));

    for (m = -n; m <= n; ++m, ++idx) {
      bnm[idx*2+0] *= val;
      bnm[idx*2+1] *= val;
#if 0
     /* I forget what these coefficients are for */
	 Vlambda = b/(1+2*n);
	 Klambda = -1/(2*(1+2*n));
	 Snm(iIndex,jIndex) = Bnm(iIndex,jIndex) / Vlambda;
	 Snm2(iIndex,jIndex) = epsHat/(1 + epsHat*Klambda) * (n+1)/b^(2*n+2)* Enm(iIndex,jIndex);
#endif
    }
  }
  PetscCall(VecRestoreArray(Bnm, &bnm));
  PetscFunctionReturn(0);
}

/*@
  computePotential - Compute the reaction potential at the charge locations

  Input Parameters:
+ b - the sphere radius, in Angstroms
. pqrData - the PQRData context
. epsIn - the dielectric constant inside the protein
. epsOut - the dielectric constant outside the protein
. Nmax - the maximum multipole order to use
- Enm - The vector of multipole coefficients for the charge field

  Output Parameters:
. phi - The reaction potential values at the charge locations

  Level: beginner

.seealso: computeEnm(), computeBnm(), doAnalytical()
@*/
PetscErrorCode computePotential(PQRData *pqr, PetscInt Nmax, Vec Bnm, Vec phi)
{
  PetscScalar *xyz, *bnm, *p;
  PetscReal   *P;
  PetscInt     Nq, n, m, k, idx = 0;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(pqr->q, &Nq));
  PetscCall(PetscMalloc1(2*Nmax+1,&P));
  PetscCall(VecSet(phi, 0.0));
  PetscCall(VecGetArray(pqr->xyz, &xyz));
  PetscCall(VecGetArray(Bnm, &bnm));
  PetscCall(VecGetArray(phi, &p));
  for (k = 0; k < Nq; ++k) {
    PetscScalar r[3], val;

    PetscCall(convertToSpherical(&xyz[k*3], r));
    for (n = 0, idx = 0; n <= Nmax; ++n) {
      PetscCall(legendre2(n, 1, cos(r[2]), P));
      for (m = -n ; m <= n; ++m, ++idx) {
        const PetscReal Pnm  = P[PetscAbsInt(m)+n];

        p[k] += PetscPowReal(r[0], n) * Pnm * (bnm[idx*2+0] * cos(m*r[1]) - bnm[idx*2+1] * sin(m*r[1]));
        val = bnm[idx*2+1] * cos(m*r[1]) + bnm[idx*2+0] * sin(m*r[1]);
        PetscCheck(val < 1e-2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Imaginary part of potential is nonzero %g", (double) val);
      }
    }
  }
  PetscCall(VecRestoreArray(pqr->xyz, &xyz));
  PetscCall(VecRestoreArray(Bnm, &bnm));
  PetscCall(VecRestoreArray(phi, &p));
  PetscCall(PetscFree(P));
  PetscFunctionReturn(0);
}

/*@
  doAnalytical - Compute the analytical solvation matrix L

  Input Parameters:
+ b - the sphere radius, in Angstroms
. epsIn - the dielectric constant inside the protein
. epsOut - the dielectric constant outside the protein
. pqrData - the PQRData context
- Nmax - the maximum multipole order to use

  Output Parameters:
+ L    - the actual solvation matrix (Hessian)
- Lbib - the BIBEE/CFA solvation matrix (Hessian)

  Level: beginner

  Note: In order to get kcal/mol energies, you need to multiply by 332.112 outside this function

.seealso: computeEnm(), computeBnm(), computePotential()
@*/
PetscErrorCode doAnalytical(PetscReal b, PetscReal epsIn, PetscReal epsOut, PQRData *pqr, PetscInt Nmax, Mat *L)
{
  Vec          Enm, Bnm, tmpq;
  PetscScalar *a;
  PetscInt     Nq, q;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(pqr->q, &Nq));
  PetscCall(VecDuplicate(pqr->q, &tmpq));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, Nq, Nq, NULL, L));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) *L, "lref_"));
  PetscCall(MatDenseGetArray(*L, &a));

  /* \sum^{N_{max}}_{l=0} 2l + 1 = 2 * (Nmax)(Nmax+1)/2 + Nmax+1 = (Nmax+1)^2 */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, PetscSqr(Nmax+1) * 2, &Enm));
  PetscCall(PetscObjectSetName((PetscObject) Enm, "Enm Coefficients"));
  PetscCall(VecDuplicate(Enm, &Bnm));
  PetscCall(PetscObjectSetName((PetscObject) Bnm, "Bnm Coefficients"));
  for (q = 0; q < Nq; ++q) {
    Vec phi;

    PetscCall(VecSet(tmpq, 0.0));
    PetscCall(VecSetValue(tmpq, q, 1.0, INSERT_VALUES));

    /* The vector phi should be L(:,q) */
    PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, Nq, &a[Nq*q], &phi));
    PetscCall(PetscObjectSetName((PetscObject) phi, "Reaction Potential"));
    PetscCall(computeEnm(b, epsIn, pqr, tmpq, Nmax, Enm));
    PetscCall(computeBnm(b, epsIn, epsOut, Nmax, Enm, Bnm));
    PetscCall(computePotential(pqr, Nmax, Bnm, phi));
    PetscCall(VecDestroy(&phi));
  }
  PetscCall(MatDenseRestoreArray(*L, &a));
  PetscCall(MatAssemblyBegin(*L, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*L, MAT_FINAL_ASSEMBLY));
  PetscCall(VecDestroy(&Enm));
  PetscCall(VecDestroy(&Bnm));
  PetscCall(VecDestroy(&tmpq));
  PetscFunctionReturn(0);
}

/*@
  IntegratePanel - Returns potential at evaluation point due to unit monopole and unit dipole uniformly distributed on a panel.

  Input Parameters:
. panel - The vertex coordinates for this panel in the panel coordinate system
. point - The evaluation point in the panel coordinate system
. normal - [Optional] The evaluation direction in the panel coordinate system

  Output Parameters:
. fss - the potential at evalpnt due to a panel monopole
. fds - the potential at evalpnt due to a panel normal dipole distribution
. fess - the derivative of the monopole potential at evalpnt along direction
. feds - the derivative of the dipole potential at evalpnt along direction

  Note: This is called calcp() in Matlab and FFTSVD. All calculations take place in the panel coordinate system,
  in which the face lies in the x-y plane.
@*/
PetscErrorCode IntegratePanel(PetscInt numCorners, const PetscReal npanel[], const PetscReal point[], const PetscReal normal[], PetscScalar *fss, PetscScalar *fds, PetscScalar *fess, PetscScalar *feds)
{
  PetscScalar fs, fsx, fsy,      fes;
  PetscScalar fd, fdx, fdy, fdz, fed;
  PetscReal   zn  = point[2], znabs = PetscAbsReal(zn);
  PetscReal   elen[4]; /* the length of each edge in the panel */
  PetscReal   ct[4];   /* cos(th) where th is the angle that panel edge i makes with the x-axis */
  PetscReal   st[4];   /* sin(th) where th is the angle that panel edge i makes with the x-axis */
  PetscReal   fe[4];   /* x-z plane square distance from evalpnt to each vertex */
  PetscReal   r[4];    /* distance from evalpnt to each vertex */
  PetscReal   xmxv[4]; /* x distance from evalpnt to each vertex */
  PetscReal   ymyv[4]; /* y distance from evalpnt to each vertex */
  PetscReal   xri[4];  /* cos(th) where th is the angle to panel made by vector from each vertex to evalpnt */
  PetscReal   yri[4];  /* sin(th) where th is the angle to panel made by vector from each vertex to evalpnt */
  PetscBool   isNormal = PETSC_FALSE; /* The evaluation point lies along the line passing through a vertex oriented along the normal */
  PetscInt    c;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(IntegratePanel_Event, 0, 0, 0, 0));
  for (c = 0; c < numCorners; ++c) {
    /* Jay used a left-handed coordinate system, so iterate backwards */
    const PetscInt curr = (numCorners - c)%numCorners;
    const PetscInt next = (numCorners*2 - c - 1)%numCorners;
    PetscReal      dx[3];
    PetscInt       d;

    elen[c] = 0.0;
    for (d = 0; d < 3; ++d) elen[c] += PetscSqr(npanel[next*3+d] - npanel[curr*3+d]);
    elen[c] = PetscSqrtReal(elen[c]);
    /* My coordinate system seems rotated compared to Jay's */
    ct[c] = (npanel[next*3+0] - npanel[curr*3+0])/elen[c];
    st[c] = (npanel[next*3+1] - npanel[curr*3+1])/elen[c];

    for (d = 0; d < 3; ++d) dx[d] = point[d] - npanel[curr*3+d];
    xmxv[c] = dx[0];
    ymyv[c] = dx[1];
    fe[c]   = PetscSqr(dx[0]) + PetscSqr(dx[2]);
    r[c]    = PetscSqrtReal(PetscSqr(dx[1]) + fe[c]);
    if (r[c] < 1.005*znabs) isNormal = PETSC_TRUE;
    if (normal) {
      xri[c] = xmxv[c]/r[c];
      yri[c] = ymyv[c]/r[c];
    }
  }

  if (feds && *((PetscInt *) feds) == 0) {
    PetscInt d;
    for (d = 0; d < 3; ++d) {PetscCall(PetscPrintf(PETSC_COMM_SELF, "elen[%d] %g\n", d, elen[d]));}
    for (d = 0; d < 3; ++d) {PetscCall(PetscPrintf(PETSC_COMM_SELF, "ct[%d] %g\n", d, ct[d]));}
    for (d = 0; d < 3; ++d) {PetscCall(PetscPrintf(PETSC_COMM_SELF, "st[%d] %g\n", d, st[d]));}
    for (d = 0; d < 3; ++d) {PetscCall(PetscPrintf(PETSC_COMM_SELF, "xmxv[%d] %g\n", d, xmxv[d]));}
    for (d = 0; d < 3; ++d) {PetscCall(PetscPrintf(PETSC_COMM_SELF, "ymyv[%d] %g\n", d, ymyv[d]));}
    for (d = 0; d < 3; ++d) {PetscCall(PetscPrintf(PETSC_COMM_SELF, "fe[%d] %g\n", d, fe[d]));}
    for (d = 0; d < 3; ++d) {PetscCall(PetscPrintf(PETSC_COMM_SELF, "r[%d] %g\n", d, r[d]));}
  }

  /* The potential and dipole contributions are made by summing up a contribution from each edge */
  fs = 0;
  fd = 0;
  if (normal) {
    fsx = 0; fsy = 0;
    fdx = 0; fdy = 0; fdz = 0;
  }


  for (c = 0; c < numCorners; ++c) {
    const PetscInt next = (c+1)%numCorners;
    PetscReal      v, arg;

    /* v is the projection of the eval-i edge on the perpend to the side-i:
       Exploits the fact that corner points in panel coordinates. */
    v = xmxv[c]*st[c] - ymyv[c]*ct[c];

    /* arg == zero if eval on next-i edge, but then v = 0. */
    arg = (r[c]+r[next] - elen[c])/(r[c]+r[next] + elen[c]);
    if (arg > 0.0) {
      PetscReal fln;

      fln = -PetscLogReal(arg);
      fs  = fs + v * fln;
      if (normal) {
        PetscReal fac;

        fac = (r[c]+r[next]-elen[c]) * (r[c]+r[next]+elen[c]);
        fac = v*(elen[c]+ elen[c])/fac;
        fsx = fsx + (fln*st[c] - fac*(xri[c] + xri[next]));
        fsy = fsy - (fln*ct[c] + fac*(yri[c] + yri[next]));
        fdz = fdz - (fac*( 1.0/r[c] + 1.0/r[next]));
      }
    }
    PetscReal s1, c1, s2, c2, s12, c12, val;

    if (!isNormal) {
      /* eval not near a vertex normal, use Hess-Smith */
      s1 = v*r[c];
      c1 = znabs*(xmxv[c]*ct[c]+ymyv[c]*st[c]);
      s2 = v*r[next];
      c2 = znabs*(xmxv[next]*ct[c]+ymyv[next]*st[c]);
    } else {
      /* eval near a vertex normal, use Newman */
      s1 = (fe[c]*st[c])-(xmxv[c]*ymyv[c]*ct[c]);
      c1 = znabs*r[c]*ct[c];
      s2 = (fe[next]*st[c])-(xmxv[next]*ymyv[next]*ct[c]);
      c2 = znabs*r[next]*ct[c];
    }

    s12 = (s1*c2)-(s2*c1);
    c12 = (c1*c2)+(s1*s2);
    val = atan2(s12, c12);
    fd  = fd+val;
    if (normal) {
      PetscReal fac, u1, u2, rr, fh1, fh2;

      u1 = xmxv[c]*ct[c] + ymyv[c]*st[c];
      u2 = xmxv[next]*ct[c]+ymyv[next]*st[c];
      if (isNormal) {
        rr  = r[c]*r[c];
        fh1 = xmxv[c]*ymyv[c];
        fh2 = xmxv[next]*ymyv[next];
        fac = c1/((c1*c1+s1*s1)*rr );
        if (zn < 0.0) fac = -fac;
        fdx = fdx + ((rr*v+fh1*u1)*fac);
        fdy = fdy - (fe[c]*u1*fac);
        rr  = r[next]*r[next];
        fac = c2/((c2*c2+s2*s2)*rr);
        if (zn < 0.0) fac = -fac;
        fdx = fdx - ((rr*v+fh2*u2)*fac);
        fdy = fdy + fe[next]*u2*fac;
      } else {
        fac = zn/(c1*c1+s1*s1);
        fdx = fdx + (u1*v*xri[c]+r[c]*ymyv[c])*fac;
        fdy = fdy + (u1*v*yri[c]-r[c]*xmxv[c])*fac;
        fac = zn/(c2*c2+s2*s2);
        fdx = fdx - ((u2*v*xri[next]+r[next]*ymyv[next])*fac);
        fdy = fdy - ((u2*v*yri[next]-r[next]*xmxv[next])*fac);
      }
    }
  }

  /* I do not understand this line, and it is screwing up */
  // if (fd < 0.0) fd = fd + 2*PETSC_PI;
  if (fd < -1.0e-7) fd = fd + 2*PETSC_PI;
  if (zn < 0.0) fd = -fd;

  fs = fs - zn*fd;

  if (normal) {
    fsx = fsx - zn*fdx;
    fsy = fsy - zn*fdy;
    fes = normal[0]*fsx + normal[1]*fsy - normal[2]*fd;
    fed = normal[0]*fdx + normal[1]*fdy + normal[2]*fdz;
    PetscCall(PetscLogFlops((2 + 61) * numCorners + 14));
  }
  PetscCall(PetscLogFlops((24 + 29) * numCorners + 2));

  /* No area normalization */
  *fss = fs;
  *fds = fd;
  if (normal) *fess = fes;
  if (normal) *feds = fed;
  PetscCall(PetscLogEventEnd(IntegratePanel_Event, 0, 0, 0, 0));
  PetscFunctionReturn(0);
}

/*@
  makeSurfaceToSurfacePanelOperators_Laplace - Make A matrix mapping the surface to itself

  Input Parameters:
+ coordinates - The vertex coordinates
. w - The vertex weights
- n - The vertex normals

  Output Parameters:
+ V - The single layer operator
- K - The double layer operator

  Level: beginner

.seealso: doAnalytical()
@*/
PetscErrorCode makeSurfaceToSurfacePanelOperators_Laplace(DM dm, Vec w, Vec n, Mat *V, Mat *K)
{
  const PetscReal fac = 1.0/4.0/PETSC_PI;
  Vec             coordinates;
  PetscSection    coordSection;
  PetscInt        Np;
  PetscInt        i, j;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(CalcStoS_Event, 0, 0, 0, 0));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMPlexGetHeightStratum(dm, 0, NULL, &Np));
  if (V) {PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, Np, Np, NULL, V));}
  if (K) {PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, Np, Np, NULL, K));}
  for (i = 0; i < Np; ++i) {
    PetscScalar *coords = NULL;
    PetscReal    panel[12], R[9], v0[3];
    PetscInt     numCorners, coordSize, d, e;

    PetscCall(DMPlexGetConeSize(dm, i, &numCorners));
    PetscCall(DMPlexVecGetClosure(dm, coordSection, coordinates, i, &coordSize, &coords));
    for (d = 0; d < 3; ++d) v0[d] = coords[d];
    PetscCall(DMPlexComputeProjection3Dto2D(coordSize, coords, R)); /* 28 + 36 + 27 = 91 flops */
    for (d = 0; d < numCorners; ++d) {
      panel[d*3+0] = PetscRealPart(coords[d*2+0]);
      panel[d*3+1] = PetscRealPart(coords[d*2+1]);
      panel[d*3+2] = 0.0;
    }
    PetscCall(DMPlexVecRestoreClosure(dm, coordSection, coordinates, i, &coordSize, &coords));
    for (j = 0; j < Np; ++j) {
      PetscScalar *tcoords = NULL;
      PetscReal    centroid[3], cloc[3];
      PetscScalar  fss, fds;

      PetscCall(DMPlexVecGetClosure(dm, coordSection, coordinates, j, NULL, &tcoords));
      for (d = 0; d < 3; ++d) {
        centroid[d] = 0.0;
        for (e = 0; e < numCorners; ++e) centroid[d] += tcoords[e*3+d];
        centroid[d] /= numCorners;
      }
      /* Rotate centroid into panel coordinate system */
      for (d = 0; d < 3; ++d) {
        cloc[d] = 0.0;
        for (e = 0; e < 3; ++e) {
          cloc[d] += R[e*3+d] * (centroid[e] - v0[e]);
        }
      }
      PetscCall(DMPlexVecRestoreClosure(dm, coordSection, coordinates, j, NULL, &tcoords));
      /* 'panel' is the coordinates of the panel vertices in the panel coordinate system */
      /*  TODO pass normals if we want fess for Kp */
      PetscCall(IntegratePanel(numCorners, panel, cloc, NULL, &fss, &fds, NULL, NULL));

      if (V) {PetscCall(MatSetValue(*V, j, i, fss*fac, INSERT_VALUES));}
      if (K) {PetscCall(MatSetValue(*K, j, i, fds*fac, INSERT_VALUES));}
      /* if (Kp) {PetscCall(MatSetValue(*singleLayer, j, i, fess/4/PETSC_PI, INSERT_VALUES));} */
    }
  }
  PetscCall(PetscLogFlops(37 * Np*Np + 91 * Np + 2));
  if (V) {PetscCall(PetscLogFlops(Np*Np));}
  if (K) {PetscCall(PetscLogFlops(Np*Np));}
  if (V) {PetscCall(MatAssemblyBegin(*V, MAT_FINAL_ASSEMBLY));PetscCall(MatAssemblyEnd(*V, MAT_FINAL_ASSEMBLY));}
  if (K) {PetscCall(MatAssemblyBegin(*K, MAT_FINAL_ASSEMBLY));PetscCall(MatAssemblyEnd(*K, MAT_FINAL_ASSEMBLY));}
  PetscCall(PetscLogEventEnd(CalcStoS_Event, 0, 0, 0, 0));
  PetscFunctionReturn(0);
}

/*@
  makeSurfaceToChargePanelOperators - Make B and C matrices mapping point charges to the surface points

  Input Parameters:
+ coordinates - The vertex coordinates
. w - The vertex weights
. n - The vertex normals
- pqrData - the PQRData context

  Output Parameters:
+ potential -
. field -
. singleLayer -
- doubleLayer -

  Level: beginner

.seealso: doAnalytical()
@*/
PetscErrorCode makeSurfaceToChargePanelOperators(DM dm, Vec w, Vec n, PQRData *pqr, Mat *potential, Mat *field, Mat *singleLayer, Mat *doubleLayer)
{
  const PetscReal    fac = 1.0/4.0/PETSC_PI;
  Vec                coordinates;
  PetscSection       coordSection;
  const PetscScalar *xyz;
  PetscInt           Nq, Np;
  PetscInt           i, j;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(CalcStoQ_Event, 0, 0, 0, 0));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(VecGetLocalSize(pqr->q, &Nq));
  PetscCall(DMPlexGetHeightStratum(dm, 0, NULL, &Np));
  if (potential || field) SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "Do not currently make the potential or field operators");
  if (potential)   {PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, Np, Nq, NULL, potential));}
  if (field)       {PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, Np, Nq, NULL, field));}
  if (singleLayer) {PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, Nq, Np, NULL, singleLayer));}
  if (doubleLayer) {PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, Nq, Np, NULL, doubleLayer));}
  PetscCall(VecGetArrayRead(pqr->xyz, &xyz));
  for (i = 0; i < Np; ++i) {
    PetscScalar *coords = NULL;
    PetscReal    panel[12], R[9], v0[3];
    PetscInt     numCorners, coordSize, d, e;

    PetscCall(DMPlexGetConeSize(dm, i, &numCorners));
    PetscCall(DMPlexVecGetClosure(dm, coordSection, coordinates, i, &coordSize, &coords));
    for (d = 0; d < 3; ++d) v0[d] = coords[d];
    PetscCall(DMPlexComputeProjection3Dto2D(coordSize, coords, R)); /* 28 + 36 + 27 = 91 flops */
    for (d = 0; d < numCorners; ++d) {
      panel[d*3+0] = PetscRealPart(coords[d*2+0]);
      panel[d*3+1] = PetscRealPart(coords[d*2+1]);
      panel[d*3+2] = 0.0;
    }
    PetscCall(DMPlexVecRestoreClosure(dm, coordSection, coordinates, i, &coordSize, &coords));
    for (j = 0; j < Nq; ++j) {
      PetscReal   qloc[3];
      PetscScalar fss, fds;

      /* Rotate charge location into panel coordinate system */
      for (d = 0; d < 3; ++d) {
        qloc[d] = 0.0;
        for (e = 0; e < 3; ++e) {
          qloc[d] += R[e*3+d] * (xyz[j*3+e] - v0[e]);
        }
      }
      /* 'panel' is the coordinates of the panel vertices in the panel coordinate system */
      PetscCall(IntegratePanel(numCorners, panel, qloc, NULL, &fss, &fds, NULL, NULL));

#if 0
      if (!i) {
        for (d = 0; d < numCorners; ++d) {
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "v%d (%g, %g, %g)\n", d, panel[d*3+0], panel[d*3+1], panel[d*3+2]));
        }
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "q (%g, %g, %g)\n", d, qloc[0], qloc[1], qloc[2]));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "fss %g fds %g\n", fss, fds));
      }
#endif

      if (potential)   {PetscCall(MatSetValue(*potential,   i, j, 0.0,     INSERT_VALUES));}
      if (field)       {PetscCall(MatSetValue(*field,       i, j, 0.0,     INSERT_VALUES));}
      if (singleLayer) {PetscCall(MatSetValue(*singleLayer, j, i, fss*fac, INSERT_VALUES));}
      if (doubleLayer) {PetscCall(MatSetValue(*doubleLayer, j, i, fds*fac, INSERT_VALUES));}
    }
  }
  PetscCall(PetscLogFlops(27 * Np*Nq + 91 * Np + 2));
  if (singleLayer) {PetscCall(PetscLogFlops(Np*Nq));}
  if (doubleLayer) {PetscCall(PetscLogFlops(Np*Nq));}
  PetscCall(VecRestoreArrayRead(pqr->xyz, &xyz));
  if (potential)   {PetscCall(MatAssemblyBegin(*potential,   MAT_FINAL_ASSEMBLY));PetscCall(MatAssemblyEnd(*potential,   MAT_FINAL_ASSEMBLY));}
  if (field)       {PetscCall(MatAssemblyBegin(*field,       MAT_FINAL_ASSEMBLY));PetscCall(MatAssemblyEnd(*field,       MAT_FINAL_ASSEMBLY));}
  if (singleLayer) {PetscCall(MatAssemblyBegin(*singleLayer, MAT_FINAL_ASSEMBLY));PetscCall(MatAssemblyEnd(*singleLayer, MAT_FINAL_ASSEMBLY));}
  if (doubleLayer) {PetscCall(MatAssemblyBegin(*doubleLayer, MAT_FINAL_ASSEMBLY));PetscCall(MatAssemblyEnd(*doubleLayer, MAT_FINAL_ASSEMBLY));}
  PetscCall(PetscLogEventEnd(CalcStoQ_Event, 0, 0, 0, 0));
  PetscFunctionReturn(0);
}

typedef struct {
  PetscReal *surf_coords;
  PetscReal *surf_weights;
  PetscReal *surf_normals;
} PointSurfCtx;

static PetscErrorCode PointSurfCtxDestroy(void **ctx)
{
  PointSurfCtx *ectx = (PointSurfCtx *)*ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscFree(ectx->surf_coords));
  PetscCall(PetscFree(ectx->surf_weights));
  PetscCall(PetscFree(ectx->surf_normals));
  PetscCall(PetscFree(ectx));
  *ctx = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SingleLayerSSKernel(PetscInt sdim, PetscInt M, PetscInt N, const PetscInt *rows, const PetscInt *cols, PetscScalar *ptr, void *ctx)
{
  PointSurfCtx    *ectx = (PointSurfCtx *)ctx;
  const PetscReal  fac  = 1.0 / 4.0 / PETSC_PI;
  PetscInt         j, k, d;

  PetscFunctionBeginUser;
  for (j = 0; j < M; j++) {
    for (k = 0; k < N; k++) {
      PetscReal r = 0.0, rvec[3];

      for (d = 0; d < 3; d++) {
        rvec[d] = ectx->surf_coords[rows[j] * 3 + d] - ectx->surf_coords[cols[k] * 3 + d];
        r += rvec[d] * rvec[d];
      }
      r = PetscSqrtReal(r);
      if (r > 1e-6) {
        ptr[j + M * k] = ectx->surf_weights[cols[k]] * fac / r;
      } else {
        const PetscReal R0 = PetscSqrtReal(ectx->surf_weights[cols[k]] / PETSC_PI);
        ptr[j + M * k]    = (2.0 * PETSC_PI * R0) / (4.0 * PETSC_PI);
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DoubleLayerSSKernel(PetscInt sdim, PetscInt M, PetscInt N, const PetscInt *rows, const PetscInt *cols, PetscScalar *ptr, void *ctx)
{
  PointSurfCtx    *ectx = (PointSurfCtx *)ctx;
  const PetscReal  fac  = 1.0 / 4.0 / PETSC_PI;
  PetscInt         j, k, d;

  PetscFunctionBeginUser;
  for (j = 0; j < M; j++) {
    for (k = 0; k < N; k++) {
      PetscReal r = 0.0, dot = 0.0, rvec[3];

      for (d = 0; d < 3; d++) {
        rvec[d] = ectx->surf_coords[rows[j] * 3 + d] - ectx->surf_coords[cols[k] * 3 + d];
        dot += rvec[d] * ectx->surf_normals[cols[k] * 3 + d];
        r += rvec[d] * rvec[d];
      }
      r              = PetscSqrtReal(r);
      ptr[j + M * k] = (r > 1e-6) ? ectx->surf_weights[cols[k]] * dot * fac / PetscPowRealInt(r, 3) : 0.0;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  PetscReal *surf_coords;
  PetscReal *surf_weights;
  PetscReal *surf_normals;
  PetscReal *charge_coords;
} PointSurfChargeCtx;

static PetscErrorCode PointSurfChargeCtxDestroy(void **ctx)
{
  PointSurfChargeCtx *ectx = (PointSurfChargeCtx *)*ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscFree(ectx->surf_coords));
  PetscCall(PetscFree(ectx->surf_weights));
  PetscCall(PetscFree(ectx->surf_normals));
  PetscCall(PetscFree(ectx->charge_coords));
  PetscCall(PetscFree(ectx));
  *ctx = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* potential[i,j] = G(surf_i, charge_j): rows=surface (target), cols=charges (source) */
static PetscErrorCode PotentialSCKernel(PetscInt sdim, PetscInt M, PetscInt N, const PetscInt *rows, const PetscInt *cols, PetscScalar *ptr, void *ctx)
{
  PointSurfChargeCtx *ectx = (PointSurfChargeCtx *)ctx;
  const PetscReal     fac  = 1.0 / 4.0 / PETSC_PI;
  PetscInt            j, k, d;

  PetscFunctionBeginUser;
  for (j = 0; j < M; j++) {
    for (k = 0; k < N; k++) {
      PetscReal r = 0.0, rvec[3];

      for (d = 0; d < 3; d++) {
        rvec[d] = ectx->surf_coords[rows[j] * 3 + d] - ectx->charge_coords[cols[k] * 3 + d];
        r += rvec[d] * rvec[d];
      }
      r              = PetscSqrtReal(r);
      ptr[j + M * k] = (r >= 1e-10) ? fac / r : 0.0;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* field[i,j] = dG/dn(surf_i, charge_j): rows=surface (target), cols=charges (source) */
static PetscErrorCode FieldSCKernel(PetscInt sdim, PetscInt M, PetscInt N, const PetscInt *rows, const PetscInt *cols, PetscScalar *ptr, void *ctx)
{
  PointSurfChargeCtx *ectx = (PointSurfChargeCtx *)ctx;
  const PetscReal     fac  = 1.0 / 4.0 / PETSC_PI;
  PetscInt            j, k, d;

  PetscFunctionBeginUser;
  for (j = 0; j < M; j++) {
    for (k = 0; k < N; k++) {
      PetscReal r = 0.0, dot = 0.0, rvec[3];

      for (d = 0; d < 3; d++) {
        rvec[d] = ectx->surf_coords[rows[j] * 3 + d] - ectx->charge_coords[cols[k] * 3 + d];
        dot += rvec[d] * ectx->surf_normals[rows[j] * 3 + d];
        r += rvec[d] * rvec[d];
      }
      r              = PetscSqrtReal(r);
      ptr[j + M * k] = (r >= 1e-10) ? -dot * fac / PetscPowRealInt(r, 3) : 0.0;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* singleLayer[j,i] = G(charge_j, surf_i)*w_i: rows=charges (target), cols=surface (source) */
static PetscErrorCode SingleLayerSCKernel(PetscInt sdim, PetscInt M, PetscInt N, const PetscInt *rows, const PetscInt *cols, PetscScalar *ptr, void *ctx)
{
  PointSurfChargeCtx *ectx = (PointSurfChargeCtx *)ctx;
  const PetscReal     fac  = 1.0 / 4.0 / PETSC_PI;
  PetscInt            j, k, d;

  PetscFunctionBeginUser;
  for (j = 0; j < M; j++) {
    for (k = 0; k < N; k++) {
      PetscReal r = 0.0, rvec[3];

      for (d = 0; d < 3; d++) {
        rvec[d] = ectx->surf_coords[cols[k] * 3 + d] - ectx->charge_coords[rows[j] * 3 + d];
        r += rvec[d] * rvec[d];
      }
      r              = PetscSqrtReal(r);
      ptr[j + M * k] = (r >= 1e-10) ? fac / r * ectx->surf_weights[cols[k]] : 0.0;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* doubleLayer[j,i] = dG/dn(charge_j, surf_i)*w_i: rows=charges (target), cols=surface (source) */
static PetscErrorCode DoubleLayerSCKernel(PetscInt sdim, PetscInt M, PetscInt N, const PetscInt *rows, const PetscInt *cols, PetscScalar *ptr, void *ctx)
{
  PointSurfChargeCtx *ectx = (PointSurfChargeCtx *)ctx;
  const PetscReal     fac  = 1.0 / 4.0 / PETSC_PI;
  PetscInt            j, k, d;

  PetscFunctionBeginUser;
  for (j = 0; j < M; j++) {
    for (k = 0; k < N; k++) {
      PetscReal r = 0.0, dot = 0.0, rvec[3];

      for (d = 0; d < 3; d++) {
        rvec[d] = ectx->surf_coords[cols[k] * 3 + d] - ectx->charge_coords[rows[j] * 3 + d];
        dot += rvec[d] * ectx->surf_normals[cols[k] * 3 + d];
        r += rvec[d] * rvec[d];
      }
      r              = PetscSqrtReal(r);
      ptr[j + M * k] = (r >= 1e-10) ? -dot * fac / PetscPowRealInt(r, 3) * ectx->surf_weights[cols[k]] : 0.0;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  makeSurfaceToChargePointOperators - Make B and C matrices mapping point charges to the surface points

  Input Parameters:
+ coordinates - The vertex coordinates
. w - The vertex weights
. n - The vertex normals
- pqrData - the PQRData context

  Output Parameters:
+ potential -
. field -
. singleLayer -
- doubleLayer -

  Level: beginner

.seealso: doAnalytical()
@*/
PetscErrorCode makeSurfaceToChargePointOperators(Vec coordinates, Vec w, Vec n, PQRData *pqr, Mat *potential, Mat *field, Mat *singleLayer, Mat *doubleLayer)
{
  PointSurfChargeCtx *ectx;
  PetscContainer      ctxContainer;
  const PetscScalar  *coords, *xyz, *weights, *normals;
  PetscInt            Nq, Np, i;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(CalcStoQ_Event, 0, 0, 0, 0));
  PetscCall(VecGetLocalSize(pqr->q, &Nq));
  PetscCall(VecGetLocalSize(w, &Np));

  PetscCall(PetscNew(&ectx));
  PetscCall(PetscMalloc1(Np * 3, &ectx->surf_coords));
  PetscCall(PetscMalloc1(Np, &ectx->surf_weights));
  PetscCall(PetscMalloc1(Np * 3, &ectx->surf_normals));
  PetscCall(PetscMalloc1(Nq * 3, &ectx->charge_coords));
  PetscCall(VecGetArrayRead(coordinates, &coords));
  PetscCall(VecGetArrayRead(pqr->xyz, &xyz));
  PetscCall(VecGetArrayRead(w, &weights));
  PetscCall(VecGetArrayRead(n, &normals));
  for (i = 0; i < Np * 3; i++) ectx->surf_coords[i]  = PetscRealPart(coords[i]);
  for (i = 0; i < Np; i++) ectx->surf_weights[i]      = PetscRealPart(weights[i]);
  for (i = 0; i < Np * 3; i++) ectx->surf_normals[i]  = PetscRealPart(normals[i]);
  for (i = 0; i < Nq * 3; i++) ectx->charge_coords[i] = PetscRealPart(xyz[i]);
  PetscCall(VecRestoreArrayRead(coordinates, &coords));
  PetscCall(VecRestoreArrayRead(pqr->xyz, &xyz));
  PetscCall(VecRestoreArrayRead(w, &weights));
  PetscCall(VecRestoreArrayRead(n, &normals));

  PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &ctxContainer));
  PetscCall(PetscContainerSetPointer(ctxContainer, ectx));
  PetscCall(PetscContainerSetCtxDestroy(ctxContainer, PointSurfChargeCtxDestroy));

  if (potential) {
    PetscCall(MatCreateHtoolFromKernel(PETSC_COMM_SELF, Np, Nq, Np, Nq, 3, ectx->surf_coords, ectx->charge_coords, PotentialSCKernel, ectx, potential));
    PetscCall(MatSetFromOptions(*potential));
    PetscCall(MatAssemblyBegin(*potential, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*potential, MAT_FINAL_ASSEMBLY));
    PetscCall(PetscObjectCompose((PetscObject)*potential, "kernelctx", (PetscObject)ctxContainer));
  }
  if (field) {
    PetscCall(MatCreateHtoolFromKernel(PETSC_COMM_SELF, Np, Nq, Np, Nq, 3, ectx->surf_coords, ectx->charge_coords, FieldSCKernel, ectx, field));
    PetscCall(MatSetFromOptions(*field));
    PetscCall(MatAssemblyBegin(*field, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*field, MAT_FINAL_ASSEMBLY));
    PetscCall(PetscObjectCompose((PetscObject)*field, "kernelctx", (PetscObject)ctxContainer));
  }
  if (singleLayer) {
    PetscCall(MatCreateHtoolFromKernel(PETSC_COMM_SELF, Nq, Np, Nq, Np, 3, ectx->charge_coords, ectx->surf_coords, SingleLayerSCKernel, ectx, singleLayer));
    PetscCall(MatSetFromOptions(*singleLayer));
    PetscCall(MatAssemblyBegin(*singleLayer, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*singleLayer, MAT_FINAL_ASSEMBLY));
    PetscCall(PetscObjectCompose((PetscObject)*singleLayer, "kernelctx", (PetscObject)ctxContainer));
  }
  if (doubleLayer) {
    PetscCall(MatCreateHtoolFromKernel(PETSC_COMM_SELF, Nq, Np, Nq, Np, 3, ectx->charge_coords, ectx->surf_coords, DoubleLayerSCKernel, ectx, doubleLayer));
    PetscCall(MatSetFromOptions(*doubleLayer));
    PetscCall(MatAssemblyBegin(*doubleLayer, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*doubleLayer, MAT_FINAL_ASSEMBLY));
    PetscCall(PetscObjectCompose((PetscObject)*doubleLayer, "kernelctx", (PetscObject)ctxContainer));
  }
  PetscCall(PetscContainerDestroy(&ctxContainer));
  PetscCall(PetscLogEventEnd(CalcStoQ_Event, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  makeSurfaceToSurfacePointOperators_Laplace - Make V and K matrices mapping the surface to itself

  Input Parameters:
+ epsIn - the dielectric constant inside the protein
. epsOut - the dielectric constant outside the protein
. pqrData - the PQRData context
. coordinates - The vertex coordinates
. w - The vertex weights
- n - The vertex normals

  Output Parameters:
+ V - The single layer surface operator
- K - The double layer surface operator

  Level: beginner

.seealso: doAnalytical()
@*/
PetscErrorCode makeSurfaceToSurfacePointOperators_Laplace(Vec coordinates, Vec w, Vec n, Mat *V, Mat *K)
{
  PointSurfCtx      *ectx;
  PetscContainer     ctxContainer;
  const PetscScalar *coords, *weights, *normals;
  PetscInt           Np, i;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(CalcStoS_Event, 0, 0, 0, 0));
  PetscCall(VecGetLocalSize(w, &Np));

  PetscCall(PetscNew(&ectx));
  PetscCall(PetscMalloc1(Np * 3, &ectx->surf_coords));
  PetscCall(PetscMalloc1(Np, &ectx->surf_weights));
  PetscCall(PetscMalloc1(Np * 3, &ectx->surf_normals));
  PetscCall(VecGetArrayRead(coordinates, &coords));
  PetscCall(VecGetArrayRead(w, &weights));
  PetscCall(VecGetArrayRead(n, &normals));
  for (i = 0; i < Np * 3; i++) ectx->surf_coords[i]  = PetscRealPart(coords[i]);
  for (i = 0; i < Np; i++) ectx->surf_weights[i]      = PetscRealPart(weights[i]);
  for (i = 0; i < Np * 3; i++) ectx->surf_normals[i]  = PetscRealPart(normals[i]);
  PetscCall(VecRestoreArrayRead(coordinates, &coords));
  PetscCall(VecRestoreArrayRead(w, &weights));
  PetscCall(VecRestoreArrayRead(n, &normals));

  PetscCall(PetscContainerCreate(PETSC_COMM_SELF, &ctxContainer));
  PetscCall(PetscContainerSetPointer(ctxContainer, ectx));
  PetscCall(PetscContainerSetCtxDestroy(ctxContainer, PointSurfCtxDestroy));

  if (V) {
    PetscCall(MatCreateHtoolFromKernel(PETSC_COMM_SELF, Np, Np, Np, Np, 3, ectx->surf_coords, ectx->surf_coords, SingleLayerSSKernel, ectx, V));
    PetscCall(MatSetFromOptions(*V));
    PetscCall(MatAssemblyBegin(*V, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*V, MAT_FINAL_ASSEMBLY));
    PetscCall(PetscObjectCompose((PetscObject)*V, "kernelctx", (PetscObject)ctxContainer));
  }
  if (K) {
    PetscCall(MatCreateHtoolFromKernel(PETSC_COMM_SELF, Np, Np, Np, Np, 3, ectx->surf_coords, ectx->surf_coords, DoubleLayerSSKernel, ectx, K));
    PetscCall(MatSetFromOptions(*K));
    PetscCall(MatAssemblyBegin(*K, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*K, MAT_FINAL_ASSEMBLY));
    PetscCall(PetscObjectCompose((PetscObject)*K, "kernelctx", (PetscObject)ctxContainer));
  }
  PetscCall(PetscContainerDestroy(&ctxContainer));
  PetscCall(PetscLogEventEnd(CalcStoS_Event, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  makeBEMPcmQualMatrices - Make solvation matrix, L = C A^{-1} B in the Polarizable Continuum Model

  Input Parameters:
+ epsIn - the dielectric constant inside the protein
. epsOut - the dielectric constant outside the protein
. pqrData - the PQRData context
. coordinates - The vertex coordinates
. w - The vertex weights
- n - The vertex normals

  Output Parameters:
. L - The solvation matrix

  Level: beginner

.seealso: doAnalytical()
@*/
PetscErrorCode makeBEMPcmQualMatrices(DM dm, BEMType bem, PetscReal epsIn, PetscReal epsOut, PQRData *pqr, Vec coordinates, Vec w, Vec n, Mat *L)
{
  const PetscReal epsHat = (epsIn - epsOut)/(epsIn + epsOut);
  KSP             ksp;
  PC              pc;
  Mat             A, Bp, B, C, S;
  Vec             d;

  PetscFunctionBeginUser;
  switch (bem) {
  case BEM_POINT:
    PetscCall(makeSurfaceToSurfacePointOperators_Laplace(coordinates, w, n, NULL, &A));
    PetscCall(makeSurfaceToChargePointOperators(coordinates, w, n, pqr, NULL, &B, &C, NULL));
    PetscCall(PetscLogEventBegin(CalcL_Event, 0, 0, 0, 0));
    /* Convert B (MatHtool) to dense before in-place modification */
    {
      Mat Bdense;
      PetscCall(MatConvert(B, MATDENSE, MAT_INITIAL_MATRIX, &Bdense));
      PetscCall(MatDestroy(&B));
      B = Bdense;
    }
    /* B = chargesurfop.dphidnCoul */
    PetscCall(MatDiagonalScale(B, w, NULL));
    PetscCall(MatScale(B, -1/epsIn));
    break;
  case BEM_PANEL:
    PetscCall(makeSurfaceToSurfacePanelOperators_Laplace(dm, w, NULL /*n*/, NULL, &A));
    PetscCall(makeSurfaceToChargePanelOperators(dm, w, NULL /*n*/, pqr, NULL, NULL, &C, &Bp));
    PetscCall(PetscLogEventBegin(CalcL_Event, 0, 0, 0, 0));
    /* Bp = chargesurfop.dlpToCharges */
    PetscCall(MatTranspose(Bp, MAT_INITIAL_MATRIX, &B));
    PetscCall(MatDestroy(&Bp));
    PetscCall(MatScale(B, -1/epsIn));
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid BEM type: %d", bem);
  }
  /* C = chargesurfop.slpToCharges */
  PetscCall(MatScale(C, 4.0*PETSC_PI));
  /* A = surfsurfop.K */
  {
    Mat At;
    PetscCall(MatTranspose(A, MAT_INITIAL_MATRIX, &At));
    PetscCall(MatDestroy(&A));
    A = At;
  }
  PetscCall(MatDiagonalScale(A, NULL, w));
  PetscCall(VecDuplicate(w, &d));
  PetscCall(VecCopy(w, d));
  PetscCall(VecScale(d, 1.0/(2.0*epsHat)));
  PetscCall(MatDiagonalSet(A, d, ADD_VALUES));
  PetscCall(VecDestroy(&d));

  PetscCall(KSPCreate(PetscObjectComm((PetscObject) A), &ksp));
  PetscCall(KSPSetType(ksp, KSPPREONLY));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCLU));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetUp(ksp));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, A->rmap->n, B->cmap->n, NULL, &S));
  PetscCall(KSPMatSolve(ksp, B, S));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatMatMult(C, S, MAT_INITIAL_MATRIX, PETSC_DEFAULT, L));
  PetscCall(MatDestroy(&S));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscLogEventEnd(CalcL_Event, 0, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeBEMResidual(SNES snes, Vec x, Vec r, void *ctx)
{
  Mat *A = (Mat *) ctx;

  PetscFunctionBegin;
  PetscCall(MatMult(*A, x, r));
  PetscFunctionReturn(0);
}

static PetscErrorCode ComputeBEMJacobian(SNES snes, Vec x, Mat J, Mat P, void *ctx)
{
  Mat *A = (Mat *) ctx;

  PetscFunctionBegin;
  PetscCall(MatCopy(*A, P, SAME_NONZERO_PATTERN));
  PetscFunctionReturn(0);
}

/*
  Our nonlinear equation is F(u) = b, which for our case is

    \frac{1}{2\hat\epsilon} \sigma − K' \sigma + \hat T(K' \sigma) \sigma = \hat g

  where

      \hat g = B q
    \hat T u = T(\hat g - u)

  and we understand the 3rd term to be a pointwise product, which gives

    \frac{1}{2\hat\epsilon} \sigma − K' \sigma + T (\hat g - K' \sigma) \sigma = \hat g

  The normal electric field E_n, Eq. (3.35) in Tom's thesis, is given by \hat g - K' \sigma. \hat g is the RHS of our nonlinear equations.
*/
static PetscErrorCode ComputeSLICResidual(SNES snes, Vec X, Vec R, void *user)
{
  SLICCtx         *ctx = (SLICCtx *) user;
  PetscScalar      a   = 1.0/(2.0*ctx->epsHat);
  Vec              gHat;
  const PetscReal *w, *s, *g;
  PetscScalar     *r;
  PetscInt         n, i;

  PetscFunctionBegin;
  PetscCall(SNESGetRhs(snes, &gHat));
  PetscCall(MatMult(ctx->Kp, X, R)); /* R now has -K' \sigma */
  PetscCall(VecGetLocalSize(R, &n));
  PetscCall(VecGetArrayRead(X, &s));
  PetscCall(VecGetArrayRead(ctx->w, &w));
  PetscCall(VecGetArrayRead(gHat, &g));
  PetscCall(VecGetArray(R, &r));
  for (i = 0; i < n; ++i) r[i] += w[i]*s[i]*(a + SLIC_t((g[i] + r[i])/w[i], ctx)); /* Divide by w since Kp and g are already scaled */
  PetscCall(VecRestoreArrayRead(X, &s));
  PetscCall(VecRestoreArrayRead(ctx->w, &w));
  PetscCall(VecRestoreArrayRead(gHat, &g));
  PetscCall(VecRestoreArray(R, &r));
  PetscFunctionReturn(0);
}

/* \hat T K' (s + eps v) (s + eps v)
 = \hat T (K' s + eps K' v) (s + eps v)
 = vec( t(ks_i + eps kv_i) (s_i + eps v_i) )
 ~ vec( t(ks_i) s_i + eps kv_i t'(ks_i) s_i + eps v_i t(ks_i) )

  1/eps (T K' (s + eps v) (s + eps v) - T K' s (s) - eps J v)
= 1/eps (vec( t(ks_i + eps kv_i) (s + eps v) ) - vec( t(ks_i) s_i) - eps J v)
= 1/eps (vec( t(ks_i) s_i + eps kv_i t'(ks_i) s_i + eps v_i t(ks_i)) - vec( t(ks_i) s_i ) - eps J v)
= 1/eps (eps vec( t'(ks_i) kv_i s_i + eps v_i t(ks_i)) - eps J v)
= vec( t'(ks_i) s_i kv_i + t(ks_i) v_i) - J v = 0

so that can can define J as

  J = diag ( t'(ks_i) s_i) K' + diag(t(ks_i))
*/
static PetscErrorCode ComputeSLICJacobian(SNES snes, Vec X, Mat J, Mat P, void *user)
{
  SLICCtx           *ctx = (SLICCtx *) user;
  PetscScalar        a   = 1.0/(2.0*ctx->epsHat);
  Vec                gHat, En, d;
  PetscScalar       *r;
  const PetscScalar *w, *x, *en;
  PetscInt           n, i;

  PetscFunctionBegin;
  PetscCall(SNESGetRhs(snes, &gHat));
  PetscCall(VecDuplicate(ctx->w, &En));
  PetscCall(VecDuplicate(ctx->w, &d));
  PetscCall(MatCopy(ctx->Kp, P, SAME_NONZERO_PATTERN));
  PetscCall(MatMult(ctx->Kp, X, En));
  PetscCall(VecAXPY(En, 1.0, gHat));
  PetscCall(VecPointwiseDivide(En, En, ctx->w));
  /* First add the diag ( t'(ks_i) s_i) K' contribution */
  PetscCall(VecGetLocalSize(d, &n));
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArrayRead(En, &en));
  PetscCall(VecGetArray(d, &r));
  for (i = 0; i < n; ++i) r[i] = 1.0 + ctx->damping*x[i]*SLIC_dtdx(en[i], ctx); /* Divide by w since Kp and g are already scaled */
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArray(d, &r));
  PetscCall(MatDiagonalScale(P, d, NULL));
  /* Add 1/(2 epsHat) I + diag(T(gHat - K' \sigma)_ */
  PetscCall(VecGetArrayRead(ctx->w, &w));
  PetscCall(VecGetArrayRead(En, &en));
  PetscCall(VecGetArray(d, &r));
  for (i = 0; i < n; ++i) r[i] = a*w[i] + ctx->damping*w[i]*SLIC_t(en[i], ctx); /* Divide by w since Kp and g are already scaled */
  PetscCall(VecRestoreArrayRead(ctx->w, &w));
  PetscCall(VecRestoreArrayRead(En, &en));
  PetscCall(VecRestoreArray(d, &r));
  PetscCall(MatDiagonalSet(P, d, ADD_VALUES));
  PetscCall(VecDestroy(&d));
  PetscFunctionReturn(0);
}

/* Here we use the operators

     P = 1/(2 \hat\epsilon) I - K' + diag(t(ks_i))

and solver

  P(x_n) x_{n+1} = b

since this satisfies the fixed point equation

  P(x) x = b
  (1/2e I - K' + diag(t(b - K' x))) x = b
  1/2e x - K' x + vec(t(E_n) x_i) = b
  F(x) = b
*/
static PetscErrorCode ComputeSLICJacobianPicard(SNES snes, Vec X, Mat J, Mat P, void *user)
{
  SLICCtx           *ctx = (SLICCtx *) user;
  PetscScalar        a   = 1.0/(2.0*ctx->epsHat);
  Vec                gHat, d;
  PetscScalar       *r;
  const PetscScalar *w, *g;
  PetscInt           n, i;

  PetscFunctionBegin;
  PetscCall(SNESGetRhs(snes, &gHat));
  PetscCall(VecDuplicate(ctx->w, &d));
  PetscCall(VecGetLocalSize(d, &n));
  PetscCall(MatCopy(ctx->Kp, P, SAME_NONZERO_PATTERN));
  /* Now P contains K' + diag ( t(ks_i) ) */
  PetscCall(MatMult(ctx->Kp, X, d));
  PetscCall(VecGetArrayRead(ctx->w, &w));
  PetscCall(VecGetArrayRead(gHat, &g));
  PetscCall(VecGetArray(d, &r));
  for (i = 0; i < n; ++i) r[i] = w[i]*(a + ctx->damping*SLIC_t((g[i] + r[i])/w[i], ctx)); /* Divide by w since Kp and g are already scaled */
  PetscCall(VecRestoreArrayRead(ctx->w, &w));
  PetscCall(VecRestoreArrayRead(gHat, &g));
  PetscCall(VecRestoreArray(d, &r));
  PetscCall(MatDiagonalSet(P, d, ADD_VALUES));
  PetscFunctionReturn(0);
}

#include <petsc/private/snesimpl.h>

/* TODO: Put in the nonlinear preconditioner code
   TODO: Get line search working

   Picard for differential equations:

   y' = F(y) => y = G(y) which is a fixed point, then uses
   y_{n+1} = G(y_n)

   Jay is using

     (A + diag(h(x_n))) x_{n+1} = b

   Now if x_{n+1} = x_n = x, we have

     A x + diag(h(x)) x - b = 0
*/
static PetscErrorCode PicardSolve(SNES snes, Vec b, Vec u)
{
  SLICCtx       *ctx;
  SNESConvergedReason reason = SNES_CONVERGED_ITERATING;
  KSP            ksp;
  Mat            J, Jp;
  Vec            X, F;
  PetscReal      atol, rtol, stol, fnorm, xnorm = 0.0, ynorm = 0.0;
  PetscInt       maxit, i;

  PetscFunctionBegin;
  PetscCall(SNESGetKSP(snes, &ksp));
#if 1
  PetscCall(SNESSetSolution(snes, u));
#endif
  PetscCall(SNESGetSolution(snes, &X));
  PetscCall(SNESGetFunction(snes, &F, NULL, NULL));
  PetscCall(SNESGetJacobian(snes, &J, &Jp, NULL, (void **) &ctx));
#if 1
  snes->vec_rhs = b;
  PetscCall(PetscObjectReference((PetscObject) b));
#else
  PetscCall(SNESGetRhs(snes, &b));
#endif
  if (!b) SETERRQ(PetscObjectComm((PetscObject) snes), PETSC_ERR_ARG_WRONG, "Picard iteration requires a constant rhs vector");
  PetscCall(SNESGetTolerances(snes, &atol, &rtol, &stol, &maxit, NULL));
  PetscCall(SNESComputeFunction(snes, X, F));

  PetscCall(SNESSetIterationNumber(snes, 0));
  PetscCall(VecNorm(F, NORM_2, &fnorm));
  SNESCheckFunctionDomainError(snes, fnorm);
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject) snes));
  PetscCall(SNESSetFunctionNorm(snes, fnorm));
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject) snes));
  PetscCall(SNESLogConvergenceHistory(snes, fnorm, 0));
  PetscCall(SNESMonitor(snes, 0, fnorm));

  PetscCall((*snes->ops->converged)(snes, 0, 0.0, 0.0, fnorm, &reason, snes->cnvP));
  PetscCall(SNESSetConvergedReason(snes, reason));
  if (reason) PetscFunctionReturn(0);

  for (i = 0; i < maxit; ++i) {
    PetscInt lits;

    if (snes->ops->update) {PetscCall((*snes->ops->update)(snes, i));}

#if 1
    /* Solve J X = b, where J is Jacobian matrix */
    PetscCall(SNESComputeJacobian(snes, X, J, Jp));
    SNESCheckJacobianDomainError(snes);
    PetscCall(KSPSetOperators(ksp, J, Jp));
    PetscCall(KSPSolve(ksp, b, X));
    SNESCheckKSPSolve(snes);
    PetscCall(KSPGetIterationNumber(ksp, &lits));
    PetscCall(PetscInfo(snes, "iter=%D, linear solve iterations=%D\n", i, lits));
#else
    /* Solve J Y = F, where J is Jacobian matrix */
    PetscCall(SNESComputeJacobian(snes, X, J, Jp));
    SNESCheckJacobianDomainerror(snes);
    PetscCall(KSPSetOperators(ksp, J, Jp));
    PetscCall(KSPSolve(ksp, F, Y));
    SNESCheckKSPSolve(snes);
    PetscCall(KSPGetIterationNumber(ksp, &lits));
    PetscCall(PetscInfo2(snes, "iter=%D, linear solve iterations=%D\n", i, lits));
#endif
#if 0
    /* Compute a (scaled) negative update in the line search routine:
         X <- X - lambda*Y
       and evaluate F = function(X) (depends on the line search).
    */
    SNESLineSearchReason lssucceed;
    PetscReal            gnorm;

    gnorm = fnorm;
    ierr  = SNESLineSearchApply(linesearch, X, F, &fnorm, Y));
    ierr  = SNESLineSearchGetReason(linesearch, &lssucceed));
    ierr  = SNESLineSearchGetNorms(linesearch, &xnorm, &fnorm, &ynorm));
    ierr  = PetscInfo4(snes, "fnorm=%18.16e, gnorm=%18.16e, ynorm=%18.16e, lssucceed=%d\n", (double) gnorm, (double) fnorm, (double) ynorm, (int) lssucceed));
    if (reason) break;
    SNESCheckFunctionNorm(snes, fnorm);
    if (lssucceed) {
      if (snes->stol*xnorm > ynorm) {
        PetscCall(SNESSetConvergedReason(snes, SNES_CONVERGED_SNORM_RELATIVE));
        PetscFunctionReturn(0);
      }
      if (++snes->numFailures >= snes->maxFailures) {
        PetscBool ismin;
        PetscCall(SNESSetConvergedReason(snes, SNES_DIVERGED_LINE_SEARCH));
        PetscCall(SNESNEWTONLSCheckLocalMin_Private(snes, J, F, fnorm, &ismin));
        if (ismin) {PetscCall(SNESSetConvergedReason(snes, SNES_DIVERGED_LOCAL_MIN));}
        break;
      }
    }
#else
    PetscCall(SNESComputeFunction(snes, X, F));
    PetscCall(VecNorm(F, NORM_2, &fnorm));
    PetscCall(VecNorm(X, NORM_2, &xnorm));
    SNESCheckFunctionDomainError(snes, fnorm);
    ynorm = xnorm;
#endif
    /* Monitor convergence */
    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)snes));
    PetscCall(SNESSetIterationNumber(snes, i+1));
    PetscCall(SNESSetFunctionNorm(snes, fnorm));
    snes->ynorm = ynorm;
    snes->xnorm = xnorm;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)snes));
    PetscCall(SNESLogConvergenceHistory(snes, fnorm, lits));
    PetscCall(SNESMonitor(snes, i+1, fnorm));
    /* Check convergence */
    PetscCall((*snes->ops->converged)(snes, i+1, xnorm, ynorm, fnorm, &reason, snes->cnvP));
    PetscCall(SNESSetConvergedReason(snes, reason));
    if (reason) break;
  }
  if (i == maxit) {
    PetscCall(PetscInfo(snes, "Maximum number of iterations has been reached: %D\n", maxit));
    if (!reason) {PetscCall(SNESSetConvergedReason(snes, SNES_DIVERGED_MAX_IT));}
  }
  /* From SNESSolve() */
  PetscCall(SNESConvergedReasonViewFromOptions(snes));
  PetscCall(SNESViewFromOptions(snes, NULL, "-snes_view"));
  PetscCall(VecViewFromOptions(X, (PetscObject) snes, "-snes_view_solution"));
  PetscCall(DMMonitor(snes->dm));
  PetscFunctionReturn(0);
}

/*@
  makeBEMPcmQualReactionPotential - Make the reaction potential, phi_react = Lq = C A^{-1} Bq in the Polarizable Continuum Model

  Input Parameters:
+ epsIn - the dielectric constant inside the protein
. epsOut - the dielectric constant outside the protein
. pqrData - the PQRData context
. coordinates - The vertex coordinates
. w - The vertex weights
. n - The vertex normals
- ctx - The solvation context

  Output Parameters:
. react - The reaction potential

  Level: beginner

.seealso: doAnalytical()
@*/
PetscErrorCode makeBEMPcmQualReactionPotential(DM dm, BEMType bem, PetscReal epsIn, PetscReal epsOut, PQRData *pqr, Vec coordinates, Vec w, Vec n, Vec react, SolvationContext *ctx)
{
  const PetscReal epsHat = (epsIn - epsOut)/(epsIn + epsOut);
  SNES            snes;
  Mat             J, A, Bp, B, C;
  Vec             d, t0, t1, t2;

  PetscFunctionBeginUser;
  switch (bem) {
  case BEM_POINT_MF:
    PetscCall(makeSurfaceToSurfacePointOperators_Laplace(coordinates, w, n, NULL, &A));
    PetscCall(makeSurfaceToChargePointOperators(coordinates, w, n, pqr, NULL, &B, &C, NULL));
    PetscCall(PetscLogEventBegin(CalcR_Event, 0, 0, 0, 0));
    /* B = chargesurfop.dphidnCoul */
    PetscCall(MatDiagonalScale(B, w, NULL));
    PetscCall(MatScale(B, -1/epsIn));
    break;
  case BEM_PANEL_MF:
    PetscCall(makeSurfaceToSurfacePanelOperators_Laplace(dm, w, NULL /*n*/, NULL, &A));
    PetscCall(makeSurfaceToChargePanelOperators(dm, w, NULL /*n*/, pqr, NULL, NULL, &C, &Bp));
    PetscCall(PetscLogEventBegin(CalcR_Event, 0, 0, 0, 0));
    /* Bp = chargesurfop.dlpToCharges */
    PetscCall(MatTranspose(Bp, MAT_INITIAL_MATRIX, &B));
    PetscCall(MatDestroy(&Bp));
    PetscCall(MatScale(B, -1/epsIn));
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid BEM type: %d", bem);
  }
  /* C = chargesurfop.slpToCharges */
  PetscCall(MatScale(C, 4.0*PETSC_PI));
  /* A = surfsurfop.K */
  {
    Mat At;
    PetscCall(MatTranspose(A, MAT_INITIAL_MATRIX, &At));
    PetscCall(MatDestroy(&A));
    A = At;
  }
  PetscCall(MatDiagonalScale(A, NULL, w));
  if (ctx->useSLIC) {
    ctx->slicCtx.w      = w;
    ctx->slicCtx.Kp     = A;
    ctx->slicCtx.epsHat = epsHat;
  } else {
    PetscCall(VecDuplicate(w, &d));
    PetscCall(VecCopy(w, d));
    PetscCall(VecScale(d, 1.0/(2.0*epsHat)));
    PetscCall(MatDiagonalSet(A, d, ADD_VALUES));
    PetscCall(VecDestroy(&d));
  }

  PetscCall(VecDuplicate(w, &t0));
  PetscCall(VecDuplicate(t0, &t1));
  PetscCall(PetscObjectSetName((PetscObject) t0, "Coulomb_Surface_Potential"));
  PetscCall(PetscObjectSetName((PetscObject) t1, "Reaction_Surface_Charge"));
  PetscCall(MatMult(B, pqr->q, t0));

  /* Can do Picard by using the Jacobian that gets made, the rhs that is passed in, and NEWTONLS
       F(x) = A x - b,   J(x) = A,   J dx = F(0)  ==>  A dx = -b,   x = 0 - dx

       A(0) dx_1 = F(0) - b = A(0) 0 - b = -b
         x_1 = 0 - dx_1 = 0 - (-p_1) = p_1
       A(x_1) dx_2 = F(x_1) - b = A(x_1) x_1 - b  ==>  A(x_1) (dx_2 - x_1) = -b
         dx_2 - x_1 = -p_2
         x_2 = x_1 - dx_2 = x_1 - (-p_2 + x_1) = p_2
  */

  if (bem == BEM_POINT_MF) {
    PetscCall(makeSurfaceToSurfacePointOperators_Laplace(coordinates, w, n, NULL, &J));
    {
      Mat Jt;
      PetscCall(MatTranspose(J, MAT_INITIAL_MATRIX, &Jt));
      PetscCall(MatDestroy(&J));
      J = Jt;
    }
    PetscCall(MatDiagonalScale(J, NULL, w));
    if (!ctx->useSLIC) {
      PetscCall(VecDuplicate(w, &d));
      PetscCall(VecCopy(w, d));
      PetscCall(VecScale(d, 1.0/(2.0*epsHat)));
      PetscCall(MatDiagonalSet(J, d, ADD_VALUES));
      PetscCall(VecDestroy(&d));
    }
  } else PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &J));
  PetscCall(VecDuplicate(t0, &t2));
  PetscCall(SNESCreate(PetscObjectComm((PetscObject) dm), &snes));
  if (ctx->useSLIC) {
    PetscCall(SNESSetFunction(snes, t2, ComputeSLICResidual, &ctx->slicCtx));
    PetscCall(SNESSetJacobian(snes, J, J, ComputeSLICJacobian, &ctx->slicCtx));
  } else {
    PetscCall(SNESSetFunction(snes, t2, ComputeBEMResidual, &A));
    PetscCall(SNESSetJacobian(snes, J, J, ComputeBEMJacobian, &A));
  }
  PetscCall(SNESSetFromOptions(snes));
  if (ctx->slicCtx.picard) {
    PetscCall(SNESSetJacobian(snes, J, J, ComputeSLICJacobianPicard, &ctx->slicCtx));
    PetscCall(PicardSolve(snes, t0, t1));
  } else {
    PetscCall(SNESSolve(snes, t0, t1));
  }
  PetscCall(SNESDestroy(&snes));
  PetscCall(VecViewFromOptions(t1, NULL, "-charge_view"));

  PetscCall(MatMult(C, t1, react));
  PetscCall(VecDestroy(&t0));
  PetscCall(VecDestroy(&t1));
  PetscCall(VecDestroy(&t2));
  PetscCall(MatDestroy(&J));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscLogEventEnd(CalcR_Event, 0, 0, 0, 0));
  PetscFunctionReturn(0);
}

/*@
  CalculateAnalyticSolvationEnergy - Calculate the solvation energy 1/2 q^T L q

  Input Parameters:
+ epsIn - the dielectric constant inside the protein
. epsOut - the dielectric constant outside the protein
. pqrData - the PQRData context
. R - The sphere radius
- Nmax - The multipole order

  Output Parameters:
+ react - The reaction potential, L q
- E - The solvation energy

  Level: beginner

.seealso: doAnalytical()
@*/
PetscErrorCode CalculateAnalyticSolvationEnergy(PetscReal epsIn, PetscReal epsOut, PQRData *pqr, PetscReal R, PetscInt Nmax, Vec react, PetscReal *E)
{
  const PetscReal q     = ELECTRON_CHARGE;
  const PetscReal Na    = AVOGADRO_NUMBER;
  const PetscReal JperC = 4.184; /* Jouled/Calorie */
  const PetscReal cf    = Na * (q*q/EPSILON_0)/JperC * (1e10/1000) * 1/4/PETSC_PI; /* kcal ang/mol */
  Mat             L;

  PetscFunctionBeginUser;
  PetscAssertPointer(pqr, 3);
  PetscAssertPointer(E, 6);
  PetscCall(doAnalytical(R, epsIn, epsOut, pqr, Nmax, &L));
  PetscCall(MatMult(L, pqr->q, react));
  PetscCall(MatDestroy(&L));
  PetscCall(VecDot(pqr->q, react, E));
  *E  *= cf * 0.5;
  PetscFunctionReturn(0);
}

/*@
  CalculateBEMSolvationEnergy - Calculate the solvation energy 1/2 q^T L q

  Input Parameters:
+ dm - The DM
. prefix - A prefix to use for the objects created
. bem - The type BEM method
. epsIn - the dielectric constant inside the protein
. epsOut - the dielectric constant outside the protein
. pqrData - the PQRData context
. w - The weights
. n - The normals
- ctx - The solvation context

  Output Parameters:
+ react - The reaction potential, L q
- E - The solvation energy

  Level: beginner

.seealso: doAnalytical()
@*/
PetscErrorCode CalculateBEMSolvationEnergy(DM dm, const char prefix[], BEMType bem, PetscReal epsIn, PetscReal epsOut, PQRData *pqr, Vec w, Vec n, Vec react, PetscReal *E, SolvationContext *ctx)
{
  const PetscReal q     = ELECTRON_CHARGE;
  const PetscReal Na    = AVOGADRO_NUMBER;
  const PetscReal JperC = 4.184; /* Jouled/Calorie */
  const PetscReal cf    = Na * (q*q/EPSILON_0)/JperC * (1e10/1000) * 1/4/PETSC_PI; /* kcal ang/mol */
  Mat             L     = NULL;
  Vec             coords;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscAssertPointer(pqr, 6);
  PetscAssertPointer(E, 10);
  switch (bem) {
  case BEM_POINT:
  case BEM_PANEL:
    PetscCall(DMGetCoordinatesLocal(dm, &coords));
    PetscCall(makeBEMPcmQualMatrices(dm, bem, epsIn, epsOut, pqr, coords, w, n, &L));
    PetscCall(PetscObjectSetName((PetscObject) L, "L"));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject) L, prefix));
    PetscCall(MatViewFromOptions(L, NULL, "-mat_view"));

    PetscCall(PetscLogEventBegin(CalcE_Event, L, react, pqr->q, 0));
    PetscCall(MatMult(L, pqr->q, react));
    PetscCall(MatDestroy(&L));
    break;
  case BEM_POINT_MF:
  case BEM_PANEL_MF:
    PetscCall(DMGetCoordinatesLocal(dm, &coords));
    PetscCall(makeBEMPcmQualReactionPotential(dm, bem, epsIn, epsOut, pqr, coords, w, n, react, ctx));
    PetscCall(PetscLogEventBegin(CalcE_Event, L, react, pqr->q, 0));
    break;
  }
  PetscCall(VecDot(pqr->q, react, E));
  *E  *= cf * 0.5;
  PetscCall(PetscLogEventEnd(CalcE_Event, L, react, pqr->q, 0));
  PetscFunctionReturn(0);
}

typedef struct {
  PetscSurface s;        /* The surface we are using */
  Mat          V, K;     /* The single and double layer operators */
  Mat          Vtmp;     /* Temporary amtrix for forming Jacobian */
  Vec          J;        /* Storage for the cathodic+anodic current boundary condition */
  PetscReal    curAnode; /* The prescribed current at the anode */
} CPCtx;

static PetscErrorCode CPComputeCurrent(Vec Phi, Vec J, CPCtx *ctx)
{
  const PetscScalar *phi;
  PetscScalar       *j;
  //PetscReal          jA = ctx->curAnode;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(Phi, &phi));
  PetscCall(VecGetArray(J, &j));
  /* TODO Loop over cathode points */
  j[0] = PetscExpReal((phi[0] + 693.91)/24.0) - (1.0/86.06 + PetscExpReal((phi[0] + 521.6)/23.17)) - PetscExpReal((phi[0] + 707.57)/55.0);
  /* TODO Loop over anode points */
  PetscCall(VecRestoreArrayRead(Phi, &phi));
  PetscCall(VecRestoreArray(J, &j));
  PetscFunctionReturn(0);
}

static PetscErrorCode CPComputeCurrentDerivative(Vec Phi, Vec dJdphi, CPCtx *ctx)
{
  const PetscScalar *phi;
  PetscScalar       *djdphi;
  //PetscReal          jAdphi = 0.0;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(Phi, &phi));
  PetscCall(VecGetArray(dJdphi, &djdphi));
  /* TODO Loop over cathode points */
  djdphi[0] = PetscExpReal((phi[0] + 693.91)/24.0)/24.0 - PetscExpReal((phi[0] + 521.6)/23.17)/23.17 - PetscExpReal((phi[0] + 707.57)/55.0)/55.0;
  /* TODO Loop over anode points */
  PetscCall(VecRestoreArrayRead(Phi, &phi));
  PetscCall(VecRestoreArray(dJdphi, &djdphi));
  PetscFunctionReturn(0);
}

static PetscErrorCode CPComputeResidual(SNES snes, Vec X, Vec F, void *user)
{
  CPCtx *ctx = (CPCtx *) user;

  PetscFunctionBeginUser;
  PetscCall(MatMult(ctx->K, X, F));
  PetscCall(VecAXPY(F, 0.5, X));
  PetscCall(CPComputeCurrent(X, ctx->J, ctx));
  PetscCall(MatMultAdd(ctx->V, ctx->J, F, F));
  PetscFunctionReturn(0);
}

static PetscErrorCode CPComputeJacobian(SNES snes, Vec X, Mat J, Mat P, void *user)
{
  CPCtx *ctx = (CPCtx *) user;

  PetscFunctionBeginUser;
  /* 1/2 I + K + V f' */
  PetscCall(MatCopy(ctx->K, P, SAME_NONZERO_PATTERN));
  PetscCall(VecCopy(ctx->s->vertexAreas, ctx->J));
  PetscCall(VecScale(ctx->J, 0.5));
  PetscCall(MatDiagonalSet(P, ctx->J, ADD_VALUES));
  PetscCall(CPComputeCurrentDerivative(X, ctx->J, ctx));
  PetscCall(MatCopy(ctx->V, ctx->Vtmp, SAME_NONZERO_PATTERN));
  PetscCall(MatDiagonalScale(ctx->Vtmp, NULL, ctx->J));
  PetscCall(MatAXPY(P, 1.0, ctx->Vtmp, SAME_NONZERO_PATTERN));
  PetscFunctionReturn(0);
}

/*
  The purpose of cathodic protection is to protect a piece of infrastructure, such as a pipe, from corrosion when it is immersed in an electrolytic fluid.
The pipe is a cathode, meaning it is reduced, absorbing electrons, as it corrodes. To prevent corrosion, we introduce a "sacrifical" anode, meaning it is
oxidized, giving up electrons, such that the potential on the cathode is kept below some maximum value. We can model this with the following boundary
integral equation:
$$
  \left( \frac{1}{2} I + K_\Gamma \right) \phi + V_{\Gamma_C} f(φ) - V_{\Gamma_A} g = 0
$$
where $\Gamma_C$ is the cathode boundary, $\Gamma_A$ is the anode boundary, and $\Gamma = \Gamma_C \cup \Gamma_A$ is the combined boundary. The function
$g$ is a prescribed potential on the anode, and $f$ describes the current on the cathode arising from iron oxidation, oxygen reduction, and hydrogen evolution.
This is often described by the Butler-Volmer condition
$$
  f(\phi) = C_1 e^{C_2 \phi} - e^{-C_3 \phi}
$$
but we will also use the experimental relation for steel in seawater
$$
  f(\phi) = e^{(\phi + 693.91)/24} - \left( \frac{1}{86.06} + e^{(\phi + 521.6)/23.17} right) - e^{(\phi + 707.57)/55}
$$
We assume uniform conductivity (permittivity) of the electrolytic solution surroudning our anode and cathode.
This should more properly be modeled as a Poisson-Boltzmann equation, which we will do in the future.
*/
PetscErrorCode CalculateBEMCPPotential(DM dm, const char prefix[], BEMType bem, Vec w, Vec n, Vec phi)
{
  MPI_Comm comm;
  CPCtx    ctx;
  SNES     snes;
  Mat      J;
  Vec      coordinatesLocal, F;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscObjectGetComm((PetscObject) dm, &comm));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinatesLocal));
  switch (bem) {
  case BEM_POINT:
    PetscCall(makeSurfaceToSurfacePointOperators_Laplace(coordinatesLocal, w, n, &ctx.V, &ctx.K));
    PetscCall(MatDiagonalScale(ctx.K, NULL, w));
    PetscCall(PetscObjectSetName((PetscObject) ctx.K, "Double Layer"));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject) ctx.K, prefix));
    PetscCall(MatViewFromOptions(ctx.K, NULL, "-dl_view"));
    PetscCall(MatDiagonalScale(ctx.V, NULL, w));
    PetscCall(PetscObjectSetName((PetscObject) ctx.V, "Single Layer"));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject) ctx.V, prefix));
    PetscCall(MatViewFromOptions(ctx.V, NULL, "-sl_view"));
    break;
  default: SETERRQ(comm, PETSC_ERR_SUP, "Only support point BEM, not %s", BEMTypes[bem]);
  }
  PetscCall(MatCreateVecs(ctx.K, &ctx.J, NULL));
  /* Solve system */
  PetscCall(SNESCreate(comm, &snes));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(makeSurfaceToSurfacePointOperators_Laplace(coordinatesLocal, w, n, NULL, &J));
  PetscCall(MatDiagonalScale(J, NULL, w));
  PetscCall(VecDuplicate(ctx.J, &F));
  PetscCall(SNESSetFunction(snes, F, CPComputeResidual, &ctx));
  PetscCall(SNESSetJacobian(snes, J, J, CPComputeJacobian, &ctx));
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&F));
  PetscCall(SNESSolve(snes, NULL, phi));
  PetscCall(SNESDestroy(&snes));
  PetscCall(MatDestroy(&ctx.V));
  PetscCall(MatDestroy(&ctx.K));
  PetscFunctionReturn(0);
}

/* Overload time to be the sphere radius */
static void snapToSphere(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                         const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                         const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                         PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscReal norm2 = 0.0, fac;
  PetscInt  n = uOff[1] - uOff[0], d;

  for (d = 0; d < n; ++d) norm2 += u[d]*u[d];
  fac = t/PetscSqrtReal(norm2);
  for (d = 0; d < n; ++d) f0[d] = u[d]*fac;
}

PetscErrorCode ProblemPrintReport(DM dm, SolvationContext *ctx)
{
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) dm, &comm));
  if (ctx->useSLIC) {
    SLICCtx *slic = &ctx->slicCtx;

    PetscCall(PetscPrintf(comm, "SLIC Paramters:\n  alpha: %g\n  beta: %g\n  gamma: %g\n  mu: %g\n",
                       slic->alpha, slic->beta, slic->gamma, slic->mu));
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  /* Constants */
#if 0
  const PetscReal  q     = ELECTRON_CHARGE;
  const PetscReal  Na    = AVOGADRO_NUMBER;
  const PetscReal  JperC = 4.184; /* Jouled/Calorie */
  const PetscReal  kB    = Na * BOLTZMANN_K/4.184/1000.0; /* Now in kcal/K/mol */
  const PetscReal  cf    = Na * (q*q/EPSILON_0)/JperC * (1e10/1000) * 1/4/PETSC_PI; /* kcal ang/mol */
#endif
  /* Problem data */
  DM               dm;
  PQRData          pqr;
  PetscSurface     msp = NULL;
  Vec              panelAreas, vertWeights, vertNormals, react;
  PetscReal        totalArea;
  SolvationContext ctx;
  /* Solvation Energies */
  PetscScalar      Eref = 0.0, ESimple = 0.0, ESurf = 0.0, ESurfMF = 0.0, EPanel = 0.0, EMSP = 0.0, Edm = 0.0;
  PetscLogStage    stageSimple, stageSurf, stageSurfMF, stagePanel, stageMSP;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  PetscCall(PetscLogDefaultBegin());
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &ctx));
  /* Make PQR */
  switch (ctx.probType) {
  case PROBLEM_SPHERE:
    PetscCall(makeSphereChargeDistribution(ctx.R, ctx.numCharges, ctx.h, PETSC_DETERMINE, &pqr));
    PetscCall(PQRViewFromOptions(&pqr));
    break;
  case PROBLEM_MOLECULE:
    PetscCall(PQRCreateFromPDB(PETSC_COMM_WORLD, ctx.pdbFile, ctx.crgFile, &pqr));
    break;
  default: break;
  }
  if (ctx.probType != PROBLEM_SPHERE_CP) {
    PetscCall(VecDuplicate(pqr.q, &react));
    PetscCall(PetscObjectSetName((PetscObject) react, "Reaction Potential"));
    /* Make surface */
    PetscCall(loadSrfIntoSurfacePoints(PETSC_COMM_WORLD, ctx.srfFile, &vertNormals, &vertWeights, &panelAreas, &totalArea, &dm));
    {
      PetscInt cStart, cEnd, vStart, vEnd;

      PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
      PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "SRF %D vertices %D cells\n", vEnd-vStart, cEnd-cStart));
    }
    PetscCall(PetscSurfaceCreateMSP(PETSC_COMM_WORLD, ctx.pntFile, &msp));
    if (msp) {
      PetscInt Nv;

      PetscCall(VecGetSize(msp->vertexAreas, &Nv));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MSP %D vertices\n", Nv));
    }
    /* Calculate solvation energy */
    PetscCall(PetscLogStageRegister("Point Surface", &stageSurf));
    PetscCall(PetscLogStageRegister("Point Surface MF", &stageSurfMF));
    PetscCall(PetscLogStageRegister("Panel Surface", &stagePanel));
    PetscCall(PetscLogStagePush(stageSurf));
    PetscCall(CalculateBEMSolvationEnergy(dm, "lsrf_", BEM_POINT, ctx.epsIn, ctx.epsOut, &pqr, vertWeights, vertNormals, react, &ESurf, &ctx));
    PetscCall(PetscLogStagePop());
    PetscCall(PetscLogStagePush(stageSurfMF));
    PetscCall(CalculateBEMSolvationEnergy(dm, "lsrf_mf_", BEM_POINT_MF, ctx.epsIn, ctx.epsOut, &pqr, vertWeights, vertNormals, react, &ESurfMF, &ctx));
    PetscCall(PetscLogStagePop());
    PetscCall(PetscLogStagePush(stagePanel));
    PetscCall(CalculateBEMSolvationEnergy(dm, "lpanel_", BEM_PANEL, ctx.epsIn, ctx.epsOut, &pqr, panelAreas, vertNormals, react, &EPanel, &ctx));
    PetscCall(PetscLogStagePop());
  }
  if (msp) {
    PetscCall(PetscLogStageRegister("MSP Surface", &stageMSP));
    PetscCall(PetscLogStagePush(stageMSP));
    PetscCall(CalculateBEMSolvationEnergy(msp->dmV, "lmsp_", BEM_POINT, ctx.epsIn, ctx.epsOut, &pqr, msp->vertexAreas, msp->vertexNormals, react, &EMSP, &ctx));
    PetscCall(PetscLogStagePop());
  }
  PetscCall(ProblemPrintReport(dm, &ctx));
  /* Verification */
  if (ctx.probType == PROBLEM_SPHERE) {
    const PetscInt Np = PetscCeilReal(4.0 * PETSC_PI * PetscSqr(ctx.R))*ctx.density;
    DM             dmSimple;
    Vec            vertWeightsSimple, vertNormalsSimple;

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Total area: %g Sphere area: %g\n", totalArea, 4*PETSC_PI*PetscPowRealInt(ctx.R, 2)));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Simple %D vertices\n", Np));
    PetscCall(makeSphereSurface(PETSC_COMM_WORLD, ctx.origin, ctx.R, Np, &vertWeightsSimple, &vertNormalsSimple, NULL, &dmSimple));

    PetscCall(PetscLogStageRegister("Simple Surface", &stageSimple));
    PetscCall(PetscLogStagePush(stageSimple));
    PetscCall(CalculateBEMSolvationEnergy(dmSimple, "lsimple_", BEM_POINT, ctx.epsIn, ctx.epsOut, &pqr, vertWeightsSimple, vertNormalsSimple, react, &ESimple, &ctx));
    PetscCall(PetscLogStagePop());
    PetscCall(CalculateAnalyticSolvationEnergy(ctx.epsIn, ctx.epsOut, &pqr, ctx.R, ctx.Nmax, react, &Eref));
    PetscCall(VecDestroy(&vertWeightsSimple));
    PetscCall(VecDestroy(&vertNormalsSimple));
    PetscCall(DMDestroy(&dmSimple));
    {
      PetscSurface s;
      DM           sdm;
      PetscReal    totArea;
      PetscInt     Nr = Np <= 12 ? 0 : PetscCeilReal(0.5*PetscLog2Real(Np / 12.0)), r, vStart, vEnd;

      PetscCall(DMPlexCreateSphereMesh(PETSC_COMM_WORLD, 2, PETSC_TRUE, 1.0, &sdm));
      {
        DM       cdm;
        PetscFE  fe;
        PetscInt dim, dE;

        PetscCall(DMGetCoordinateDM(sdm, &cdm));
        PetscCall(DMGetDimension(sdm, &dim));
        PetscCall(DMGetCoordinateDim(sdm, &dE));
        PetscCall(PetscFECreateLagrange(PETSC_COMM_SELF, dim, dE, PETSC_TRUE, 1, -1, &fe));
        PetscCall(DMSetField(cdm, 0, NULL, (PetscObject) fe));
        PetscCall(PetscFEDestroy(&fe));
        PetscCall(DMCreateDS(cdm));
      }
      for (r = 0; r < Nr; ++r) {
        DM rdm, cdm, rcdm;
        PetscCall(DMRefine(sdm, PETSC_COMM_WORLD, &rdm));
        PetscCall(DMGetCoordinateDM(sdm, &cdm));
        PetscCall(DMGetCoordinateDM(rdm, &rcdm));
        PetscCall(DMCopyDisc(cdm, rcdm));
        PetscCall(DMPlexRemapGeometry(rdm, ctx.R, snapToSphere));
        PetscCall(DMDestroy(&sdm));
        sdm  = rdm;
      }
      PetscCall(PetscSurfaceCreate(sdm, &s));
      PetscCall(DMPlexGetDepthStratum(sdm, 0, &vStart, &vEnd));
      PetscCall(VecNorm(s->vertexAreas, NORM_1, &totArea));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Total cell area: %g Tot vertex area: %g Sphere area: %g\n", s->totalArea, totArea, 4*PETSC_PI*PetscPowRealInt(ctx.R, 2)));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "DM %D vertices\n", vEnd-vStart));
      PetscCall(DMViewFromOptions(sdm, NULL, "-sphere_dm_view"));
      PetscCall(VecViewFromOptions(s->cellAreas,     NULL, "-careas_view"));
      PetscCall(VecViewFromOptions(s->cellNormals,   NULL, "-cnormals_view"));
      PetscCall(VecViewFromOptions(s->vertexAreas,   NULL, "-vareas_view"));
      PetscCall(VecViewFromOptions(s->vertexNormals, NULL, "-vnormals_view"));
      PetscCall(CalculateBEMSolvationEnergy(sdm, "ldm_", BEM_POINT, ctx.epsIn, ctx.epsOut, &pqr, s->vertexAreas, s->vertexNormals, react, &Edm, &ctx));
      PetscCall(PetscSurfaceDestroy(&s));
      PetscCall(DMDestroy(&sdm));
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Eref = %.6f ESurf   = %.6f Error = %.6f Rel. error = %.4f\n", Eref, ESurf,   Eref-ESurf,   (Eref-ESurf)/Eref));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Eref = %.6f ESurfMF = %.6f Error = %.6f Rel. error = %.4f\n", Eref, ESurfMF, Eref-ESurfMF, (Eref-ESurfMF)/Eref));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Eref = %.6f ESimple = %.6f Error = %.6f Rel. error = %.4f\n", Eref, ESimple, Eref-ESimple, (Eref-ESimple)/Eref));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Eref = %.6f Edm     = %.6f Error = %.6f Rel. error = %.4f\n", Eref, Edm,     Eref-Edm,     (Eref-Edm)/Eref));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Eref = %.6f EPanel  = %.6f Error = %.6f Rel. error = %.4f\n", Eref, EPanel,  Eref-EPanel,  (Eref-EPanel)/Eref));
  } else if (ctx.probType == PROBLEM_SPHERE_CP) {
    DM             dm;
    Vec            vertWeightsCP, vertNormalsCP;
    const PetscInt Np = PetscCeilReal(4.0 * PETSC_PI * PetscSqr(ctx.R))*ctx.density;
    PetscLogStage  stageCP;

    PetscCall(makeSphereSurface(PETSC_COMM_WORLD, ctx.origin, ctx.R, Np, &vertWeightsCP, &vertNormalsCP, NULL, &dm));
    PetscCall(PetscLogStageRegister("Cathodic Protection", &stageCP));
    PetscCall(PetscLogStagePush(stageCP));
    PetscCall(CalculateBEMCPPotential(dm, "cp_", BEM_POINT, vertWeightsCP, vertNormalsCP, react));
    PetscCall(PetscLogStagePop());
  } else {
    Eref = 1.0; /* TODO This should be higher resolution BEM */
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Eref = %.6f ESurf   = %.6f Error = %.6f Rel. error = %.4f\n", Eref, ESurf,  Eref-ESurf,  (Eref-ESurf)/Eref));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Eref = %.6f ESurfMF = %.6f Error = %.6f Rel. error = %.4f\n", Eref, ESurfMF, Eref-ESurfMF, (Eref-ESurfMF)/Eref));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Eref = %.6f EPanel  = %.6f Error = %.6f Rel. error = %.4f\n", Eref, EPanel, Eref-EPanel, (Eref-EPanel)/Eref));
    if (msp) {PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Eref = %.6f EMSP   = %.6f Error = %.6f Rel. error = %.4f\n", Eref, EMSP, Eref-EMSP, (Eref-EMSP)/Eref));}
  }
  /* Output flops */
  if (ctx.probType != PROBLEM_SPHERE_CP) {
    PetscEventPerfInfo stagePerfInfo, eventPerfInfo;

    PetscCall(PetscLogStageGetPerfInfo(stageSurf, &stagePerfInfo));
    PetscCall(PetscLogEventGetPerfInfo(stageSurf, CalcStoS_Event, &eventPerfInfo));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Flops_Surf  = %.4e Flops_S2S_Surf  = %.4e\n", stagePerfInfo.flops, eventPerfInfo.flops));
    PetscCall(PetscLogStageGetPerfInfo(stageSurfMF, &stagePerfInfo));
    PetscCall(PetscLogEventGetPerfInfo(stageSurfMF, CalcStoS_Event, &eventPerfInfo));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Flops_SurfMF= %.4e Flops_S2S_Surf  = %.4e\n", stagePerfInfo.flops, eventPerfInfo.flops));
    PetscCall(PetscLogStageGetPerfInfo(stagePanel, &stagePerfInfo));
    PetscCall(PetscLogEventGetPerfInfo(stagePanel, CalcStoS_Event, &eventPerfInfo));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Flops_Panel = %.4e Flops_S2S_Panel = %.4e\n", stagePerfInfo.flops, eventPerfInfo.flops));
    if (msp) {
      PetscCall(PetscLogStageGetPerfInfo(stageMSP, &stagePerfInfo));
      PetscCall(PetscLogEventGetPerfInfo(stageMSP, CalcStoS_Event, &eventPerfInfo));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Flops_MSP   = %.4e Flops_S2S_MSP   = %.4e\n", stagePerfInfo.flops, eventPerfInfo.flops));
    }
    if (ctx.probType == PROBLEM_SPHERE) {
      PetscCall(PetscLogStageGetPerfInfo(stageSimple, &stagePerfInfo));
      PetscCall(PetscLogEventGetPerfInfo(stageSimple, CalcStoS_Event, &eventPerfInfo));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Flops_Simple = %.4e Flops_S2S_Simple = %.4e\n", stagePerfInfo.flops, eventPerfInfo.flops));
    }
  }
  /* Cleanup */
  PetscCall(VecDestroy(&vertWeights));
  PetscCall(VecDestroy(&vertNormals));
  PetscCall(VecDestroy(&panelAreas));
  PetscCall(VecDestroy(&react));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscSurfaceDestroy(&msp));
  PetscCall(PQRDestroy(&pqr));
  PetscCall(PetscFinalize());
  return 0;
}
