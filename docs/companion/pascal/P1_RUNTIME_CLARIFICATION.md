# P1 runtime clarification (dated 2026-10-03)

**Status:** post-hoc provenance clarification — **not** a protocol rewrite or preregistration.  
**Immutable:** original `P1_PROTOCOL.md`, raw JSONL under `companion/pascal/artifacts/p1_runs/`, and job **117225** logs remain as executed.  
**Usability:** P1 remains a valid **within-runtime** tile-budget comparison on the observed Pascal allocation unless a concrete validity issue is found (none found here). No full P1 rerun solely for wording.

## A. BF16 provenance

### What P1 observed

Job 117225 loaded InternVL3-1B in **bfloat16** on a reported `Tesla V100-SXM3-32GB` because `torch.cuda.is_bf16_supported()` returned `True` (via shared `load_internvl` → `prefer_bf16`).

### What PyTorch 2.5.1 actually checks

Installed signature:

```text
torch.cuda.is_bf16_supported(including_emulation: bool = True)
```

Default path (`including_emulation=True`):

1. ROCm → True  
2. No CUDA → False  
3. CUDA ≥ 11 **and** device compute capability **major ≥ 8** → True (**native Ampere+ path**)  
4. Else if `including_emulation=False` → **False**  
5. Else try creating a bfloat16 tensor (`_check_bf16_tensor_supported`) → True if that succeeds (**emulation / usability path**)

V100 devices are compute capability **7.x**, so they do **not** pass step 3. A default `True` on V100 therefore indicates **BF16 tensor/runtime usability via the emulation branch**, not native CC≥8 hardware support.

### Distinctions (required)

| Concept | Meaning | P1 evidence |
|---------|---------|-------------|
| 1. BF16 tensor/runtime usability | Model can run with `dtype=bfloat16` tensors | Observed (load + 208×3 inference succeeded) |
| 2. Native hardware BF16 support | CC≥8 / non-emulation path | **Not** established by the default boolean; V100 is CC 7.x |
| 3. Specific kernel/op acceleration path | Which kernels ran in BF16 vs cast | **Unverified** in P1 |

Do **not** infer native BF16 acceleration from the default boolean or from parameter dtype alone.

### Protocol vs execution

- **Protocol wording (pre-eval):** if bf16 unsupported, use documented float16 fallback.  
- **Actual execution:** default `is_bf16_supported()` was True (emulation-inclusive), so float16 fallback was **not** applied; bfloat16 was used.  
- This is a **documented deviation**, not silent precision change after the fact. Companion docs that implied “Neumann-like native bf16 on V100” in present tense are corrected to the distinctions above.

### P2 follow-up measurements

During the P2 GPU allocation, record compute capability, `is_bf16_supported()` default, `is_bf16_supported(including_emulation=False)` if available, actual parameter/input dtypes, and inspectable attention implementations. See `P2_VERIFICATION.md`.

## B. Memory measurement scope

### Wrapper behavior (`companion/pascal/p1/infer_wrap.py`)

P1 temporarily replaced `torch.cuda.reset_peak_memory_stats` globally during each request so the shared backend’s internal pre-`generate()` reset would not clear peaks. Purpose: one **request-scoped** peak window (image open → decode).

### Field labeling

| Field | Intended meaning | Actual meaning under P1 wrapper |
|-------|------------------|----------------------------------|
| `request_gpu_memory` / summary `peak_*_request_*` | Request-scoped peaks | Correct for published tables |
| `gpu_memory` returned by `run_one_internvl` and stored as `gpu_memory_generation` | Generation-only peaks in the shared backend | **Not independently generation-only** while the reset monkey-patch is active — unsuitable for generation-only memory claims |

Published P1 compact tables use the **request-scoped** peaks. Raw JSONL retain both fields for provenance; derived analysis must not treat `gpu_memory_generation` as generation-only under this wrapper.

### P2 policy

Do **not** carry the global reset monkey-patch into P2. Demo omits live memory metrics or, if shown later, uses only the shared backend’s clearly labeled native scope.
