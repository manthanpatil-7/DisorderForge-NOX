"""Input-view routing system.

Reference: Model Family Contract §12

Each expert instance receives a designated feature subset (view).
The factory function constructs experts with correct input dimensions
and validates that the wired view matches available features.

Pre-approved views (§12.2):
  V-ESM:       ESM-2 (1280) + lightweight (41)         = 1321
  V-T5:        ProtT5 (1024) + lightweight (41)         = 1065
  V-ESM+pLDDT: ESM-2 (1280) + lightweight (41) + pLDDT (2) = 1323
  V-FULL:      All features concatenated                = 2347
  V-LITE:      Lightweight only (diagnostic, non-ensemble) = 41
"""

from dataclasses import dataclass

ESM2_DIM = 1280
PROT_T5_DIM = 1024
LIGHTWEIGHT_DIM = 41  # from src.features.sequence_features
PLDDT_OUTPUT_DIM = 2  # from src.features.plddt_extract

# ─── Part 5 Phase 1 dims ───────────────────────────────────────────
SAPROT_DIM = 1280              # westlake-repl/SaProt_650M_AF2
ESM3_DIM = 1536                # EvolutionaryScale/esm3-sm-open-v1
ALPHAFOLD_FEATURES_DIM = 7     # RSA, SS_H, SS_E, SS_C, CN, pLDDT, Flex
SELF_REFINEMENT_DIM = 1        # Part 4 ensemble probability per residue


@dataclass(frozen=True)
class InputView:
    """Named input-view specification."""
    view_id: str
    description: str
    dim: int
    requires_esm2: bool = False
    requires_prot_t5: bool = False
    requires_plddt: bool = False
    requires_lightweight: bool = True
    ensemble_eligible: bool = True  # V-LITE is diagnostic only
    # ── Part 5 Phase 1 channel toggles ────────────────────────────
    requires_saprot: bool = False
    requires_esm3: bool = False
    requires_alphafold_features: bool = False
    requires_self_refinement: bool = False
    # When True, the self-refinement channel must be loaded with
    # requires_grad=False (RT-31 enforces). The training pipeline tags
    # the slice indices [self_refinement_start, self_refinement_end) so
    # the trainer can detach gradients on that channel.
    self_refinement_no_grad: bool = False


# ─── Pre-approved views ─────────────────────────────────────────────

VIEWS = {
    "V-ESM": InputView(
        view_id="V-ESM",
        description="ESM-2 + lightweight sequence features",
        dim=ESM2_DIM + LIGHTWEIGHT_DIM,
        requires_esm2=True,
    ),
    "V-T5": InputView(
        view_id="V-T5",
        description="ProtT5-XL + lightweight sequence features",
        dim=PROT_T5_DIM + LIGHTWEIGHT_DIM,
        requires_prot_t5=True,
    ),
    "V-ESM+pLDDT": InputView(
        view_id="V-ESM+pLDDT",
        description="ESM-2 + lightweight + pLDDT",
        dim=ESM2_DIM + LIGHTWEIGHT_DIM + PLDDT_OUTPUT_DIM,
        requires_esm2=True,
        requires_plddt=True,
    ),
    "V-ESM+T5": InputView(
        view_id="V-ESM+T5",
        description="ESM-2 + ProtT5 + lightweight (no pLDDT)",
        dim=ESM2_DIM + PROT_T5_DIM + LIGHTWEIGHT_DIM,
        requires_esm2=True,
        requires_prot_t5=True,
    ),
    "V-FULL": InputView(
        view_id="V-FULL",
        description="All available features concatenated",
        dim=ESM2_DIM + PROT_T5_DIM + LIGHTWEIGHT_DIM + PLDDT_OUTPUT_DIM,
        requires_esm2=True,
        requires_prot_t5=True,
        requires_plddt=True,
    ),
    "V-LITE": InputView(
        view_id="V-LITE",
        description="Lightweight features only (diagnostic baseline)",
        dim=LIGHTWEIGHT_DIM,
        ensemble_eligible=False,
    ),

    # ─── Part 5 Phase 1 input modes (M27 candidates) ────────────────
    # Mode A is the Part 4 baseline ("V-ESM" above). Modes B–F below.

    # Mode B: ESM-2 + lightweight + Part 4 self-refinement prediction
    "P5-Mode-B-SelfRefine": InputView(
        view_id="P5-Mode-B-SelfRefine",
        description=("Part 5 Mode B — ESM-2 + lightweight + Part 4 prediction "
                     "(self-refinement, FROZEN per CLAUDE.md rule 27)"),
        dim=ESM2_DIM + LIGHTWEIGHT_DIM + SELF_REFINEMENT_DIM,
        requires_esm2=True,
        requires_self_refinement=True,
        self_refinement_no_grad=True,
    ),

    # Mode C: SaProt (replaces ESM-2) + lightweight
    "P5-Mode-C-SaProt": InputView(
        view_id="P5-Mode-C-SaProt",
        description="Part 5 Mode C — SaProt + lightweight (replaces ESM-2)",
        dim=SAPROT_DIM + LIGHTWEIGHT_DIM,
        requires_esm2=False,
        requires_saprot=True,
    ),

    # Mode D: ESM-3 (replaces ESM-2) + lightweight
    "P5-Mode-D-ESM3": InputView(
        view_id="P5-Mode-D-ESM3",
        description="Part 5 Mode D — ESM-3 + lightweight (replaces ESM-2)",
        dim=ESM3_DIM + LIGHTWEIGHT_DIM,
        requires_esm2=False,
        requires_esm3=True,
    ),

    # Mode E: ESM-2 + lightweight + AlphaFold 7-dim structural features
    "P5-Mode-E-AlphaFold": InputView(
        view_id="P5-Mode-E-AlphaFold",
        description=("Part 5 Mode E — ESM-2 + lightweight + AlphaFold 7-dim "
                     "structural features (RSA, SS_H, SS_E, SS_C, CN, pLDDT, Flex)"),
        dim=ESM2_DIM + LIGHTWEIGHT_DIM + ALPHAFOLD_FEATURES_DIM,
        requires_esm2=True,
        requires_alphafold_features=True,
    ),

    # Mode F1 (EXP-S05): SaProt + lightweight + Part 4 self-refinement
    "P5-Mode-F1-SaProt-SR": InputView(
        view_id="P5-Mode-F1-SaProt-SR",
        description=("Part 5 Mode F1 — SaProt + lightweight + Part 4 prediction "
                     "(SR channel FROZEN per rule 27). The SaProt backbone "
                     "already encodes structure; SR adds calibrated disorder prior."),
        dim=SAPROT_DIM + LIGHTWEIGHT_DIM + SELF_REFINEMENT_DIM,
        requires_saprot=True,
        requires_self_refinement=True,
        self_refinement_no_grad=True,
    ),

    # Mode F2 (EXP-S06): SaProt + lightweight + Part 4 SR + AlphaFold features
    "P5-Mode-F2-SaProt-SR-AF": InputView(
        view_id="P5-Mode-F2-SaProt-SR-AF",
        description=("Part 5 Mode F2 — SaProt + lightweight + Part 4 prediction "
                     "+ AlphaFold 7-dim features. Maximum input combo."),
        dim=SAPROT_DIM + LIGHTWEIGHT_DIM + SELF_REFINEMENT_DIM + ALPHAFOLD_FEATURES_DIM,
        requires_saprot=True,
        requires_self_refinement=True,
        requires_alphafold_features=True,
        self_refinement_no_grad=True,
    ),
}


# ─── Recommended expert instances (§12.3) ───────────────────────────

EXPERT_CONFIGS = {
    "A-esm": {"family": "A", "view": "V-ESM"},
    "A-t5": {"family": "A", "view": "V-T5"},
    "B-esm": {"family": "B", "view": "V-ESM"},
    "C-esm+plddt": {"family": "C", "view": "V-ESM+pLDDT"},
}


def get_available_views(
    has_esm2: bool = True,
    has_prot_t5: bool = False,
    has_plddt: bool = False,
) -> dict[str, InputView]:
    """Return views available given current feature availability."""
    available = {}
    for vid, view in VIEWS.items():
        if view.requires_esm2 and not has_esm2:
            continue
        if view.requires_prot_t5 and not has_prot_t5:
            continue
        if view.requires_plddt and not has_plddt:
            continue
        available[vid] = view
    return available


def create_expert_instance(
    family: str,
    view_id: str,
    config: dict | None = None,
    has_esm2: bool = True,
    has_prot_t5: bool = False,
    has_plddt: bool = False,
):
    """Create an expert model instance with the correct input view wiring.

    Args:
        family: "A" (CNN), "B" (BiLSTM), or "C" (MLP).
        view_id: One of the pre-approved view IDs.
        config: Optional dict of architecture hyperparameters.
        has_esm2/has_prot_t5/has_plddt: Feature availability flags.

    Returns:
        Instantiated model with correct input_dim for the view.

    Raises:
        ValueError: If view requires unavailable features or family unknown.
    """
    if view_id not in VIEWS:
        raise ValueError(f"Unknown view: {view_id}. Available: {list(VIEWS.keys())}")

    view = VIEWS[view_id]

    # Validate feature availability
    if view.requires_esm2 and not has_esm2:
        raise ValueError(f"View {view_id} requires ESM-2 but it's not available")
    if view.requires_prot_t5 and not has_prot_t5:
        raise ValueError(f"View {view_id} requires ProtT5 but it's not available")
    if view.requires_plddt and not has_plddt:
        raise ValueError(f"View {view_id} requires pLDDT but it's not available")

    config = config or {}
    input_dim = view.dim

    if family == "A":
        from src.models.family_a_cnn import DilatedResidualCNN
        return DilatedResidualCNN(input_dim=input_dim, **config)
    elif family == "B":
        from src.models.family_b_bilstm import BiLSTMHead
        return BiLSTMHead(input_dim=input_dim, **config)
    elif family == "C":
        from src.models.family_c_mlp import ResidueLocalMLP
        return ResidueLocalMLP(input_dim=input_dim, **config)
    else:
        raise ValueError(f"Unknown family: {family}. Must be 'A', 'B', or 'C'")
