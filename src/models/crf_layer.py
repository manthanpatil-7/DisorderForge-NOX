"""Linear-chain CRF output layer (P3-Ph2-S02).

Reference: P3_00 Amendment M-2; P3_04 §S02.

Two-state linear-chain CRF (state 0 = ORDERED, state 1 = DISORDERED) that
attaches to the per-residue logits produced by any expert head (CNN, BiLSTM,
or Transformer). Implementation is manual log-space so we don't add the
`pytorch-crf` dependency; it supports masking semantics that match BCE
exactly:

  - Contributing residues: labels in {ORDERED (0), DISORDERED (1)}
  - Excluded residues:     labels in {AMBIGUOUS (-2), MASKED (-1)}

Loss for a single sequence:

    NLL = log_partition(emissions, mask) - log_score_constrained(emissions, gold, mask)

where `log_partition` runs the standard forward algorithm marginalizing over
all label sequences, and `log_score_constrained` runs the SAME forward
algorithm but with non-masked positions clamped to their gold label. Masked
positions are marginalized in both numerator and denominator, so they
contribute zero to the NLL — same guarantee as BCE.

Inference exposes:
  - `marginals`: per-residue P(D) via forward-backward (use this for rpAP/AUC)
  - `viterbi`:   most-likely label sequence (use this for MaxF1)
"""

from __future__ import annotations

import torch
import torch.nn as nn

ORDERED = 0
DISORDERED = 1
NUM_STATES = 2

# Label semantics for masking (must match src/training/loss.py)
LABEL_AMB = -2
LABEL_MASKED = -1
LABEL_ORDERED = 0
LABEL_DISORDERED = 1


def _build_mask(labels: torch.Tensor) -> torch.Tensor:
    """Float mask: 1 for contributing positions, 0 for excluded."""
    return ((labels == LABEL_ORDERED) | (labels == LABEL_DISORDERED)).float()


def _logsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.logsumexp(x, dim=dim)


class LinearChainCRF(nn.Module):
    """2-state linear-chain CRF with masked NLL.

    Args:
        init_transition_scale: std for the init of the transition matrix.
            Plan §S02 advises near-zero (weak prior toward independence) to
            avoid Viterbi collapsing to all-O or all-D.
    """

    def __init__(self, init_transition_scale: float = 0.01):
        super().__init__()
        self.num_states = NUM_STATES
        self.transitions = nn.Parameter(torch.randn(NUM_STATES, NUM_STATES) * init_transition_scale)
        self.start = nn.Parameter(torch.zeros(NUM_STATES))
        self.end = nn.Parameter(torch.zeros(NUM_STATES))

    # ── Helpers ────────────────────────────────────────────────────

    def _emissions_from_logit(self, logit: torch.Tensor) -> torch.Tensor:
        """Convert per-residue scalar logits into 2-state emission scores [..., 2].

        Convention: emission[O] = 0, emission[D] = logit. Equivalently, the
        binary-classification logit is the difference of state scores. This
        keeps the CRF strictly compatible with sigmoid(logit) = P(D) when
        transition scores are zero.
        """
        zeros = torch.zeros_like(logit)
        return torch.stack((zeros, logit), dim=-1)

    # ── Forward algorithm ──────────────────────────────────────────

    def _forward_alg(
        self,
        emissions: torch.Tensor,         # [B, L, 2]
        seq_mask: torch.Tensor,          # [B, L] bool: True = real position (not pad)
        constrain: torch.Tensor | None,  # [B, L] long in {0, 1, -1} or None.
                                         # -1 → marginalize. 0/1 → clamp to that state.
    ) -> torch.Tensor:
        """Return [B] log-sum over allowed paths."""
        B, L, S = emissions.shape

        def _state_mask(t: int) -> torch.Tensor | None:
            """For position t, return [B, S] bool of allowed states (None = all)."""
            if constrain is None:
                return None
            c = constrain[:, t]                      # [B] in {-1, 0, 1}
            allowed = torch.zeros((B, S), dtype=torch.bool, device=emissions.device)
            free = (c == -1)
            allowed[free] = True
            for s in range(S):
                allowed[c == s, s] = True
            return allowed

        NEG_INF = torch.tensor(-1e30, device=emissions.device, dtype=emissions.dtype)

        # alpha_t[b, s] = log( sum over paths ending in state s at position t )
        # Initialize with start + emission_0 (constrained where applicable)
        alpha = self.start.unsqueeze(0) + emissions[:, 0, :]   # [B, S]
        sm0 = _state_mask(0)
        if sm0 is not None:
            alpha = torch.where(sm0, alpha, NEG_INF.expand_as(alpha))

        # Mask the t=0 position: if it's a padding position, keep alpha but it
        # only really matters when paired with the per-position seq_mask check
        # below (we treat seq_mask[:,0] as always True for non-empty proteins).

        for t in range(1, L):
            # transitions: [S_prev, S_curr]; alpha[..., None]: [B, S_prev, 1]
            score = alpha.unsqueeze(2) + self.transitions.unsqueeze(0)  # [B, S, S]
            score = _logsumexp(score, dim=1)  # [B, S]
            new_alpha = score + emissions[:, t, :]
            sm_t = _state_mask(t)
            if sm_t is not None:
                new_alpha = torch.where(sm_t, new_alpha, NEG_INF.expand_as(new_alpha))
            # If position t is padding (seq_mask False), keep alpha unchanged
            valid = seq_mask[:, t : t + 1]  # [B, 1]
            alpha = torch.where(valid, new_alpha, alpha)

        # Add end transitions only at the last real position per protein.
        # Easiest: compute final = alpha + end at the last valid index.
        seq_lengths = seq_mask.sum(dim=1).long()  # [B]
        idx = (seq_lengths - 1).clamp(min=0)
        # We've propagated alpha all the way to L-1, but for proteins shorter
        # than L the alpha at L-1 still equals alpha at the last-valid t
        # (because we kept it unchanged on pads). So:
        return _logsumexp(alpha + self.end.unsqueeze(0), dim=1)  # [B]

    # ── Public API ─────────────────────────────────────────────────

    def nll(
        self,
        logits: torch.Tensor,    # [B, L] or [B, L, 1] — per-residue scalar logits
        labels: torch.Tensor,    # [B, L] in {-2, -1, 0, 1}
        seq_mask: torch.Tensor | None = None,  # [B, L] bool, True = real position
    ) -> torch.Tensor:
        """Masked NLL averaged over contributing positions."""
        if logits.dim() == 3 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        B, L = logits.shape
        if seq_mask is None:
            seq_mask = torch.ones((B, L), dtype=torch.bool, device=logits.device)
        seq_mask = seq_mask.bool()

        emissions = self._emissions_from_logit(logits)  # [B, L, 2]

        # Constraint vector for the gold path: 0 or 1 where labeled, -1 where
        # masked/ambiguous (marginalize), -1 also for padding (won't matter
        # because we keep alpha unchanged on padding positions in _forward_alg).
        constrain = torch.full_like(labels, -1, dtype=torch.long)
        constrain = torch.where(labels == LABEL_ORDERED, torch.tensor(0, device=labels.device), constrain)
        constrain = torch.where(labels == LABEL_DISORDERED, torch.tensor(1, device=labels.device), constrain)

        log_Z = self._forward_alg(emissions, seq_mask, constrain=None)
        log_gold = self._forward_alg(emissions, seq_mask, constrain=constrain)
        nll_per_protein = log_Z - log_gold  # [B] >= 0

        # Average over contributing residues (matches BCE per-residue avg)
        contributing = _build_mask(labels) * seq_mask.float()  # [B, L]
        n_contrib = contributing.sum()
        if n_contrib < 1.0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        return nll_per_protein.sum() / n_contrib

    def marginals(
        self,
        logits: torch.Tensor,
        seq_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward-backward marginals: per-residue P(D). Returns [B, L]."""
        if logits.dim() == 3 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        B, L = logits.shape
        if seq_mask is None:
            seq_mask = torch.ones((B, L), dtype=torch.bool, device=logits.device)
        seq_mask = seq_mask.bool()

        emissions = self._emissions_from_logit(logits)  # [B, L, 2]

        # Forward
        alpha = torch.zeros((B, L, NUM_STATES), device=logits.device, dtype=logits.dtype)
        alpha[:, 0] = self.start.unsqueeze(0) + emissions[:, 0, :]
        for t in range(1, L):
            score = alpha[:, t - 1].unsqueeze(2) + self.transitions.unsqueeze(0)  # [B, S, S]
            score = _logsumexp(score, dim=1)
            new_alpha = score + emissions[:, t, :]
            valid = seq_mask[:, t : t + 1]
            alpha[:, t] = torch.where(valid, new_alpha, alpha[:, t - 1])

        # Backward
        beta = torch.zeros((B, L, NUM_STATES), device=logits.device, dtype=logits.dtype)
        # last valid position: end transition
        beta[:, L - 1] = self.end.unsqueeze(0).expand(B, NUM_STATES).clone()
        for t in range(L - 2, -1, -1):
            # score[b, s_curr] = logsumexp_{s_next} ( transitions[s_curr, s_next]
            #                                          + emissions[b, t+1, s_next] + beta[b, t+1, s_next] )
            score = (self.transitions.unsqueeze(0)
                     + emissions[:, t + 1, :].unsqueeze(1)
                     + beta[:, t + 1, :].unsqueeze(1))  # [B, S, S]
            new_beta = _logsumexp(score, dim=2)
            valid = seq_mask[:, t + 1 : t + 2]
            beta[:, t] = torch.where(valid, new_beta, beta[:, t + 1])

        # log marginal at position t for state s = alpha + beta
        log_marg = alpha + beta              # [B, L, 2]
        # Normalize per position
        log_marg = log_marg - _logsumexp(log_marg, dim=-1, keepdim=True) if False else log_marg - log_marg.logsumexp(dim=-1, keepdim=True)
        # P(D) = exp(log_marg[..., 1])
        return log_marg[..., 1].exp()

    def viterbi(
        self,
        logits: torch.Tensor,
        seq_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Viterbi-decoded label sequence [B, L] (long; states 0 or 1).

        Padding positions are filled with state 0 — caller should ignore via seq_mask.
        """
        if logits.dim() == 3 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        B, L = logits.shape
        if seq_mask is None:
            seq_mask = torch.ones((B, L), dtype=torch.bool, device=logits.device)
        seq_mask = seq_mask.bool()

        emissions = self._emissions_from_logit(logits)  # [B, L, 2]

        # Init
        score = self.start.unsqueeze(0) + emissions[:, 0, :]  # [B, S]
        backptr = torch.zeros((B, L, NUM_STATES), dtype=torch.long, device=logits.device)
        for t in range(1, L):
            # cand[b, s_prev, s_curr] = score[b, s_prev] + transitions[s_prev, s_curr] + emissions[b, t, s_curr]
            cand = score.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, t, :].unsqueeze(1)
            best_score, best_prev = cand.max(dim=1)
            new_score = best_score
            new_backptr = best_prev
            valid = seq_mask[:, t : t + 1]
            score = torch.where(valid, new_score, score)
            # store backptr only for valid positions (others won't be traced)
            backptr[:, t] = new_backptr

        # Termination with end transitions
        final = score + self.end.unsqueeze(0)
        last_states = final.argmax(dim=1)  # [B]

        # Backtrace
        seq_lengths = seq_mask.sum(dim=1).long()  # [B]
        path = torch.zeros((B, L), dtype=torch.long, device=logits.device)
        for b in range(B):
            Lb = int(seq_lengths[b].item())
            if Lb == 0:
                continue
            path[b, Lb - 1] = last_states[b]
            for t in range(Lb - 2, -1, -1):
                path[b, t] = backptr[b, t + 1, path[b, t + 1]]
        return path


def count_isolated_disorder(labels: torch.Tensor, seq_mask: torch.Tensor | None = None) -> int:
    """Count O→D→O transitions in a binary label tensor (diagnostic for §S07).

    Considers only adjacent positions where seq_mask is True.
    """
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)
    if seq_mask is None:
        seq_mask = torch.ones_like(labels, dtype=torch.bool)
    n = 0
    B, L = labels.shape
    for b in range(B):
        for t in range(1, L - 1):
            if not (seq_mask[b, t - 1] and seq_mask[b, t] and seq_mask[b, t + 1]):
                continue
            if int(labels[b, t]) == 1 and int(labels[b, t - 1]) == 0 and int(labels[b, t + 1]) == 0:
                n += 1
    return n
