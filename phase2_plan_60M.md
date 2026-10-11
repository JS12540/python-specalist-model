# Phase 2 Research Plan — CodeT5-small (60M)

**Goal:** Find architecture + training changes that make a small (60M) model generate,
repair, and refactor Python better than plain fine-tuning — before spending compute
scaling any winning recipe to the 220M / 770M / Qwen / Maincoder models.

**Core principle for this phase:** Code is not natural language. It has a formal grammar
and a tree structure. A model does not need to decode token-by-token left-to-right to
produce valid code — it can decode a **structural skeleton** (the AST shape) in parallel,
then fill in the leaf content, then have a **grammar-constrained** converter turn that
structure into valid Python text. Every approach below is organized around exploiting
that structure, instead of treating code as a flat string of tokens.

---

## 0. Why 60M first

| Model | Time per epoch | Time to test one idea |
|---|---|---|
| CodeT5-small (60M) | ~3 min | ~15 min |
| CodeT5+ 220M | ~10 min | ~50 min |
| CodeT5+ 770M | ~25 min | ~2+ hours |

Every idea below is validated on 60M first. Only ideas that show a real gain get
carried forward to 220M, then to the rest of the Phase 1 model list
(CodeT5+ 770M, CodeGen-350M, Qwen 0.5B, Qwen 0.5B-Instruct, Maincoder-1B).

---

## 1. Tree-Based Encoding and Decoding

### 1.1 Tree-position ENCODING (input side)

Standard transformers give every input token a flat sequential position
(0, 1, 2, 3, ...). For code INPUT (used in repair and refactor, where the input
itself is a piece of code), this throws away information the AST already has.

**What changes:**
```
Standard position:   token_i gets position = i (its index in the sequence)
Tree position:       token_i gets position = (depth, sibling_index, node_type)
                      computed by parsing the input code into an AST and walking it
```

**Implementation:**
1. Parse the input code with `ast` (or `tree-sitter` for partial/broken code in repair tasks).
2. For every token, find which AST node it belongs to, and record:
   - `depth` — how deep in the tree (0 = module level)
   - `sibling_index` — which child of its parent it is
   - `node_type` — categorical (FunctionDef, If, Call, Name, ...), embedded as an extra
     learned embedding table added to the standard position embedding
3. NL instruction tokens (no code structure) keep flat sequential positions.
4. Only applies to **repair** and **refactor** tasks (their inputs contain code).
   The **generate** task has no code input, so this doesn't apply there.

**Where this plugs in:** add a `TreePositionEmbedding` module that produces a vector
added to the standard token + position embedding, before the first encoder layer.
No change to attention, no change to the rest of the architecture.

### 1.2 Tree-position DECODING (output side) — the more important half

This is the one that actually matters for all three tasks (generate/repair/refactor),
because every output is code.

**What changes:**
```
Standard decoding: at step t, the decoder only knows it generated t-1 previous tokens
                    (flat position = t)
Tree decoding:      at step t, we maintain a PARTIAL AST built from tokens generated
                    so far. The next token's position embedding reflects WHERE IT WILL
                    LAND in that partial tree (depth, which block it's inside, etc.)
```

**Implementation:**
1. At every decoding step, maintain a partial-parse state (this is nearly free — the
   same incremental parser used for grammar-constrained decoding, see Section 3.4,
   already tracks this).
2. Before predicting the next token, look up the current "insertion point" in the
   partial AST: what depth, what parent node type, what position among siblings.
3. Feed that as an additional embedding into the decoder's input for that step
   (added to the standard decoder position embedding).
4. This tells the model "you are currently 2 levels deep inside a `try` block, as the
   3rd statement" — explicit structural awareness instead of inferring it implicitly
   from the token history.

**Cost:** ~2-5ms per token for the incremental parse update — the same cost already
paid for grammar-constrained decoding (Section 3.4), so this is close to free once
that is implemented.

### 1.3 Experiment matrix for Section 1

| Exp | Encoder positions | Decoder positions | Tasks tested |
|---|---|---|---|
| 1A | Sequential (baseline) | Sequential (baseline) | generate, repair, refactor |
| 1B | Tree (code) + Sequential (NL) | Sequential | repair, refactor only |
| 1C | Sequential | Tree (partial AST) | generate, repair, refactor |
| 1D | Tree (code) + Sequential (NL) | Tree (partial AST) | generate, repair, refactor |

**Gate:** 1D only matters if it beats both 1B and 1C individually. If 1C alone captures
most of the gain, that's the one to carry forward (cheaper — doesn't need the encoder-side
AST parse for every repair/refactor input).

---

## 2. Mixture of Experts (MoE)

### 2.1 What we already know (Phase 1 finding)

MoE was tried on CodeT5+ 220M in Phase 1 with 4 experts, top-2 routing, ~400 training
examples. Result: **0.0% HumanEval — complete failure.** Two causes identified:
1. Slicing a pretrained dense FFN into N expert chunks destroys the pretrained
   neuron relationships between weights that used to work together.
2. The router is randomly initialized and had far too little data (400 examples) to
   learn sensible routing.

### 2.2 What changes this time

MoE is **not re-attempted until Stage 1's dataset reaches 5,000+ base examples**
(see Section 6, Phase 2 dataset plan). Below 10K examples, do not re-run this —
it is a known failure mode at this scale, confirmed empirically.

When re-attempted (only after the dataset is large enough):

| Change from Phase 1 attempt | Why |
|---|---|
| Initialize experts by cloning the FULL dense FFN (not slicing it into 1/N chunks) + small noise per expert | Preserves pretrained neuron relationships entirely; experts start identical and diverge during training instead of starting broken |
| Train router with a higher learning rate than the rest of the model for the first N steps (router warmup) | Router starts random; needs to learn routing fast before the experts start specializing around bad routing decisions |
| Add a load-balancing auxiliary loss term (standard in MoE literature — penalizes the router for always picking the same expert) | Without this, routing collapses to 1-2 experts always firing, wasting the extra capacity |
| Only convert the DECODER FFN to MoE, keep encoder dense | Decoder is what the loss directly supervises every token of; encoder changes are higher risk for a first retry |
| Require ≥10K training examples before running this experiment at all | Matches literature guidance (MoE needs large data to learn useful routing) and Phase 1's empirical failure point |

### 2.3 Experiment matrix for Section 2 (gated behind Section 6 dataset milestone)

| Exp | Architecture | Expert init | Data size |
|---|---|---|---|
| 2A | Dense (baseline) | — | 5K / 10K / 20K (whichever Stage 1 produces) |
| 2B | MoE decoder only, 4 experts top-2, full-FFN-clone init | Clone + noise | same as 2A |
| 2C | MoE decoder only, 8 experts top-2, full-FFN-clone init | Clone + noise | same as 2A |

**Gate:** Only proceed past 2A→2B if the dataset has reached 10K+ examples. If 2B still
fails at 10K, do not try 2C — the problem is architectural, not data volume, and MoE
should be dropped as a direction for small models entirely.

---

## 3. Other Approaches

### 3.1 Skeleton-First Parallel Generation (highest priority — directly uses the
AST-output insight)

This is the practical expression of "code is structural, not token-by-token":

```
Pass 1 (parallel): predict the SKELETON — control-flow shape with slots
    def fetch(url):
        try:
            _SLOT_1_
        except _SLOT_2_:
            _SLOT_3_
        return _SLOT_4_
    All structure tokens predicted in ONE forward pass (short sequence — just
    keywords and slot markers, not full content).

Pass 2 (parallel): fill ALL slots simultaneously, each slot independent
    _SLOT_1_ -> "r = await client.get(url)"
    _SLOT_2_ -> "Exception as e"
    _SLOT_3_ -> "raise RuntimeError(str(e))"
    _SLOT_4_ -> "r.json()"
    One forward pass fills every slot (they don't depend on each other once the
    skeleton is fixed) — or a small number of passes for longer slot content.

Total: ~2 forward passes instead of ~30-100 sequential autoregressive tokens.
```

**Dataset requirement:** deterministic skeleton extraction via AST (no LLM needed —
same philosophy as your existing repair/refactor mutators). See Section 6.

**Two new task types added to training:**
- `skeleton`: instruction → skeleton (structure only, with `_SLOT_n_` placeholders)
- `fill`: skeleton + instruction → all slot contents (single target string
  enumerating every slot's fill)

### 3.2 AST-T5 style span corruption (data-side technique, not architecture)

From AST-T5 (ICML 2024): instead of corrupting random character/token spans during
training (as plain T5 span-corruption does), corrupt spans that align to **AST subtree
boundaries** — mask a whole `if` block, a whole function call, a whole expression,
and have the model reconstruct it.

**Why this matters for you specifically:** AST-T5 beat CodeT5 by 2-3 points on exact
match specifically on code-to-code tasks (bug fixing, transpilation) — directly your
repair task. This is a change to how you generate repair-training examples, not a
model architecture change:

```
Current repair mutator: break code via random small edits (remove colon, remove
                         import, wrong indent, etc.)
AST-T5-style addition:  ALSO generate repair examples by masking whole AST subtrees
                         and asking the model to reconstruct them — a different,
                         complementary corruption strategy
```

### 3.3 Antecedent Prioritized (AP) Loss — see Section 5 (Training Loss).

### 3.4 Grammar-Constrained Decoding (already proven in Phase 1 — always on)

Confirmed in your IR hypothesis test: grammar-constrained beam search (generate
multiple beam candidates, keep the first that parses + compiles) matches IR's 100%
compile rate without any architecture change. This stays on for every experiment
in this phase — it is not itself an experiment, it is baseline infrastructure.

This is also the mechanism that makes the "decode AST structure, then convert to
text with grammar constraints" idea in your question concrete: once Section 3.1's
skeleton is predicted, a deterministic AST-to-source converter (Python's own
`ast.unparse()`, or a custom grammar-driven renderer) turns the structural
prediction into guaranteed-valid Python text — no token-by-token generation of
punctuation, indentation, or boilerplate syntax needed at all for the skeleton part.

### 3.5 Sinkhorn structural loss — see Section 5 (Training Loss), second candidate
if AP Loss shows promise.

### 3.6 Explicitly deprioritized (tested reasoning, not being pursued now)

| Approach | Why deprioritized |
|---|---|
| Mamba / State-Space Models | 2026 consensus: attention still wins for code at ~512-token scale, where exact variable-scope tracking and bracket-matching matter more than Mamba's O(n) efficiency advantage, which only pays off at much longer sequences than your target. |
| Diffusion code generation (CodeFusion-style) | No pretrained weights available anywhere (paper withdrawn, no public checkpoint). Own paper's results show only 5.1% on even one-line Python; full-function generation was never demonstrated to work — token interdependency (variable names, bracket matching) breaks down when all positions denoise simultaneously instead of sequentially. |
| High-rank LoRA (20-30% trainable) as a shortcut to "more capacity" | Empirically tested on Maincoder-1B in Phase 1: destroyed the model (78.1% -> 3.1% HumanEval) at lr=2e-4. High-rank LoRA behaves like full fine-tuning and needs full-fine-tune-level learning rate caution, not classic-LoRA learning rates. Not a separate technique worth re-exploring — if revisited, must use lr ~2e-5, not 2e-4. |
| Mixing encoder from one pretrained model with decoder from a different one | Requires a randomly-initialized cross-attention bridge (same failure mode as under-trained MoE routers) plus a hidden-dimension projection layer plus a tokenizer bridge. Needs 10K+ examples minimum, same gate as MoE. Not attempted in this phase. |

---

## 4. Metrics to Capture

Fourteen metrics total, in four groups. All are deterministic / automated —
no manual test writing, no LLM-as-judge.

### 4.1 Task-specific correctness (the core new metrics — answers "is it actually right")

| Metric | Task | Method | Tool |
|---|---|---|---|
| Generate pass@1 | generate | Golden I/O match: auto-generate 5-10 inputs from the target function's type hints via `Hypothesis.from_type()`, capture `target_code`'s outputs as golden values once at dataset-build time, then at eval time run the model's output on the same inputs and compare | `hypothesis` |
| Repair pass@1 | repair | (a) original error no longer raised on the original failing input, AND (b) golden I/O match on the fixed logic (confirms the fix is correct, not just that the crash is suppressed) | `exec` + `hypothesis` |
| Refactor correctness | refactor | Golden I/O match — refactored code's behavior MUST be identical to `target_code`'s captured outputs; this is a non-negotiable gate before the next two metrics are even considered | `hypothesis` |
| Refactor complexity delta | refactor | Cyclomatic complexity of model output <= cyclomatic complexity of the ugly input | `radon.complexity.cc_visit` |
| Refactor maintainability delta | refactor | Maintainability Index of model output >= Maintainability Index of the ugly input | `radon.metrics.mi_visit` |

### 4.2 Structural metrics (isolates whether tree/skeleton techniques are working)

| Metric | Task | Method |
|---|---|---|
| Skeleton accuracy | all | Does the predicted control-flow skeleton match the target skeleton (same `_SLOT_n_` structure, same keywords, same nesting)? |
| Slot-fill accuracy | all | Given the correct skeleton, what % of slots are filled with exactly correct content? |
| AST edit distance | all | Tree-edit distance between generated and target AST (catches "compiles and looks plausible but structurally wrong" cases that exact-match and CodeBLEU both miss) — via `zss` or `apted` library |
| Tree-depth accuracy | repair, refactor | Does the generated code's maximum nesting depth match the target's? |

### 4.3 Syntax / surface metrics (existing — keep as baseline sanity checks)

| Metric | Method |
|---|---|
| Compile % | `ast.parse()` + `compile()` succeed |
| Parse % | `ast.parse()` succeeds (weaker check than compile) |
| CodeBLEU | Token + structure similarity to `target_code` |

### 4.4 Benchmark metrics (external validity — comparable across all 7 models)

| Metric | Method |
|---|---|
| HumanEval pass@1 | 64 stratified problems, single-pass greedy decoding (confirmed as the correct standard — see Phase 1 methodology notes), test harness must actually call `check(entry_point)` |
| ClassEval pass@1 | 100-problem OOP/class-level benchmark — tests SOLID/design-pattern generation specifically, which is your stated differentiator vs generic code models |

---

## 5. Training Loss Modifications

Two candidates, tried in order — cheapest/simplest first.

### 5.1 Antecedent Prioritized (AP) Loss — try this first

**Idea:** weight the existing token-level cross-entropy loss by AST depth. Tokens
near the root of the tree (`def`, `if`, `try`, `for` — the structural "antecedents")
get a higher loss weight than tokens deep in the tree (leaf expressions, variable
names), because getting the parent structure wrong invalidates everything nested
inside it, while a wrong variable name is a smaller, localized error.

```
Standard loss:   L = (1/N) * sum_i  CE(token_i)
AP Loss:         L = (1/N) * sum_i  weight(token_i) * CE(token_i)
                 weight(token_i) = 1 / (1 + depth(token_i) * decay_factor)
```

**Implementation:**
1. At dataset-build time, for every `target_code`, parse the AST and compute each
   token's depth, then precompute a per-token weight array alongside `input_ids`
   and `labels`. One-time cost, stored in the dataset, zero runtime cost during
   training.
2. Subclass `Trainer`, override `compute_loss` to do per-token (not pre-averaged)
   cross-entropy, multiply by the precomputed weights, then average.
3. Nothing else changes — same optimizer, same learning rate, same schedule, same
   early stopping. This is a drop-in loss swap, not a new training recipe.

**Cost to test:** near zero — a few lines of code, one afternoon on the 60M model,
same data, same hyperparameters as the existing baseline, just swap the loss.

### 5.2 Sinkhorn Structural Loss — second candidate if AP Loss shows promise

**Idea:** add an auxiliary loss term that pulls the model's attention maps toward
matching the AST's shortest-path distances between tokens — if token A and token B
are 2 hops apart in the AST, the model's attention weight between them should
reflect that proximity.

```
Total loss = CE_loss + lambda * Sinkhorn(attention_maps, AST_shortest_path_matrix)
```

**Implementation cost:** higher than AP Loss — requires computing an AST shortest-path
distance matrix per training example and extracting/comparing attention maps during
training. Only attempt this if Section 5.1 (AP Loss) shows a measurable gain first;
if AP Loss doesn't help, this more expensive technique likely won't either, since
they're both targeting the same underlying signal (structural awareness via depth/
distance in the AST).

### 5.3 Experiment matrix for Section 5

| Exp | Loss | Encoder/decoder positions (from Section 1) | Data |
|---|---|---|---|
| 5A | Standard CE (baseline) | Sequential (baseline, 1A) | Stage 0 (existing ~500) |
| 5B | AP Loss | Sequential (baseline, 1A) | Stage 0 (existing ~500) |
| 5C | AP Loss | Best combination from Section 1's experiment matrix | Stage 1 (5K+) |
| 5D | AP Loss + Sinkhorn | Best combination from Section 1 | Stage 1 (5K+) — only if 5C improves over 5B |

---

## 6. Dataset Requirement (gates everything above)

| Stage | Size | What it unlocks |
|---|---|---|
| Stage 0 | Existing ~500 examples | Cheap validation of Section 1 (tree encoding/decoding) and Section 5.1 (AP Loss) on 60M — ~2 hours total |
| Stage 1 | 5,000+ base examples, 2 new task types added (skeleton, fill) | Real training for Section 3.1 (skeleton-first), confirms Stage 0 winners at larger scale |
| Stage 1.5 | Golden I/O capture added (Hypothesis-based) for every example | Unlocks Section 4.1's task-specific correctness metrics — must exist before any of Section 4.1 can be measured |
| 10,000+ | Required before re-attempting Section 2 (MoE) at all | MoE gate — confirmed failure point below this from Phase 1 |

**Immediate next action:** build the Stage 0 validation (skeleton extraction from the
existing 500 examples + AP Loss implementation) on CodeT5-small, since both are cheap,
fast, and tell you which Section 1 / Section 5 combination is worth carrying into the
Stage 1 dataset scale-up.

---

## 7. Overall Execution Order

```
1. Stage 0: validate tree-decoding (1C) + AP Loss (5B) cheaply on existing 500 examples, CodeT5-small
2. Build Stage 1 dataset: scale to 5,000+ examples, add skeleton/fill task types,
   add Hypothesis golden-I/O capture, add AST-T5-style subtree corruption variants
3. Confirm Stage 0 winners still win at Stage 1 scale (Section 1 + Section 5 combined)
4. Run skeleton-first parallel generation (Section 3.1) properly at Stage 1 scale
5. Only once dataset reaches 10,000+: retry MoE (Section 2) with corrected init/loss-balancing
6. Combine every winning technique into one recipe, confirm on CodeT5+ 220M
7. Only then: roll the single winning recipe out to CodeT5+ 770M, CodeGen-350M,
   Qwen 0.5B, Qwen 0.5B-Instruct, Maincoder-1B
```
