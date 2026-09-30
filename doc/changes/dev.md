# Changes: Development

% STYLE GUIDELINES:
% * Capitalize sentences
% * Use imperative, e.g., Add, Improve, Change, etc.
% * Don't use a period (.) at the end of entries
% * If multiple sentences are needed, use a period or semicolon to divide sentences, but not at the end of the final sentence

```{rubric} General:
```

```{rubric} Configure/Build:
```

```{rubric} Sys:
```

```{rubric} Event Logging:
```

```{rubric} PetscViewer:
```

```{rubric} PetscDraw:
```

```{rubric} AO:
```

```{rubric} IS:
```

```{rubric} VecScatter / PetscSF:
```

```{rubric} PF:
```

```{rubric} Vec:
```

```{rubric} PetscSection:
```

```{rubric} PetscPartitioner:
```

```{rubric} Mat:
```

```{rubric} MatCoarsen:
```

```{rubric} PC:
```

- Add `PCHPDDMSetHarmonicOverlap()`, `PCHPDDMSetEPSThreshold()`, `PCHPDDMSetEPSDimensions()`, and `PCHPDDMSetSVDDimensions()` to configure `PCHPDDM` coarsening, and `PCHPDDMGetSubKSP()` to access its per-level solvers
- Reject a positive `-pc_hpddm_levels_1_eps_nev` combined with `-pc_hpddm_levels_1_svd_threshold_relative`, or a positive `-pc_hpddm_levels_1_svd_nsv` combined with `-pc_hpddm_levels_1_eps_threshold_relative`, instead of silently ignoring the incompatible threshold
- Retain previously configured `PCHPDDM` coarsening settings when `PCSetFromOptions()` is called without the corresponding options; removing these options no longer restores defaults
- Preserve previously enabled nested options processing when `PCHPDDMSetAuxiliaryMat()` replaces the overlap index set, so recreated per-level solvers reapply their options

```{rubric} KSP:
```

```{rubric} SNES:
```

```{rubric} SNESLineSearch:
```

```{rubric} TS:
```

```{rubric} TAO:
```

```{rubric} TaoTerm:
```

```{rubric} PetscRegressor:
```

```{rubric} PetscDA:
```

```{rubric} DM:
```

```{rubric} DMSwarm:
```

```{rubric} DMPlex:
```

```{rubric} FE/FV:
```

```{rubric} DMNetwork:
```

```{rubric} DMStag:
```

```{rubric} DT:
```

```{rubric} Fortran:
```
