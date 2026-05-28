# SkillOpt Deep Dive

## What is SkillOpt

SkillOpt is a framework for optimizing natural-language **skill documents** — Markdown files that act as system prompts for AI agents — using an iterative training loop that mirrors deep learning training, but never touches model weights.

The core idea: instead of fine-tuning a model, you evolve the prompt. A skill document starts as a blank file (or a hand-written seed) and is iteratively refined by analyzing where the agent fails, generating structured edit patches, and applying the best ones. Over multiple epochs the skill document accumulates task-specific strategies, heuristics, and edge-case handling.

To use SkillOpt you need three things:

- **A skill document** — Markdown, can start empty
- **A dataset** — a JSON array of `{input, ground_truth}` items (at least a few hundred; default split is 2:1:7 train/val/test)
- **An environment adapter** — benchmark-specific code that runs a task item against the model and returns a score

---

## The Training Loop

Each training step executes six stages in sequence. Steps are grouped into epochs; epoch boundaries trigger two additional mechanisms (slow update and meta skill) described below.

### 1. Rollout (Forward Pass)
*~40 target model calls per step (parallel)*

The **target model** executes a batch of tasks using the current skill document as its system prompt. All tasks in the batch run in parallel via a `ThreadPoolExecutor` (default 64 workers). Items are sampled from the training split using a deterministic per-step seed; the batch is resume-aware — already-completed items are skipped on restart.

Each task execution produces three things:

**(a) Prediction**

The model is given a system prompt (the skill document embedded in a template) and a user prompt (the task input). For SearchQA this is a question + retrieved context passages. The model responds with reasoning and a final answer inside `<answer>...</answer>` tags. If no tags are found, the last non-empty line of the response is used as the answer.

Multi-turn is supported: if `max_turns > 1` and the first response doesn't contain answer tags, a refinement prompt is sent ("your previous answer was X, review and correct if needed") and the model gets another attempt.

**(b) Score — hard and soft**

Every result carries two scores:

- **`hard`** (int 0/1) — Exact Match after SQuAD-style normalization: lowercase, strip punctuation, remove articles (a/an/the), collapse whitespace. `hard=1` only if the normalized prediction exactly equals at least one normalized gold answer. This is the binary signal used to split failures from successes.
- **`soft`** (float 0.0–1.0) — Token-level F1, computed as `2 * precision * recall / (precision + recall)` over token bags, maximized across all gold answers. A partial-credit signal; used for step-level score aggregation and slow update comparisons.
- **`sub_em`** (float) — Substring match: 1.0 if any normalized gold is a substring of the prediction or vice versa. Auxiliary metric, not used for gating decisions.

If `hard=0`, a human-readable `fail_reason` is generated: `"EM=0: predicted 'X' but expected ['Y', 'Z']"`. This string appears in the trajectory and is visible to the analyst in the reflect stage.

**Limitations of EM as the hard signal**

EM is order-sensitive even after normalization — "Mary and John" vs "John and Mary" is a hard fail despite identical content. Since `hard` is the binary signal that gates skill updates and splits failures from successes, the analyst sees an ordering mismatch as a full failure and may push the skill to enforce a specific surface form rather than learning the underlying reasoning.

`soft` (F1) is bag-of-words and completely order-insensitive, so it handles this correctly — but F1 is not used for gating, only for aggregation stats and slow update comparisons.

A deeper issue: F1 measures token overlap against a fixed gold string, and the expected overlap naturally decreases as the answer space opens up. A factoid answer ("1945") should hit F1≈1.0 or it's wrong. A reasoning-heavy or multi-part answer can be substantively correct at F1=0.4 because valid phrasings diverge from the gold. Using a fixed threshold for `hard` is therefore miscalibrated in both directions: too strict for open-domain tasks (penalizes correct but differently-phrased answers), too loose for closed factoid tasks (passes partially-wrong answers).

SearchQA mitigates this by providing multiple gold answers per question, raising the ceiling F1 achievable for any valid phrasing. But it doesn't solve the problem for truly open domains. The right approach is **domain-calibrated thresholds**: set the `hard` boundary based on the empirical F1 distribution of known-correct answers in that benchmark. If human responses to open-ended questions cluster around F1=0.5, then F1≥0.4 is a reasonable `hard=1` threshold. For short factoid answers, F1≥0.9 makes more sense. At the extreme, fully open-domain tasks may need the optimizer model itself to score responses semantically rather than using token overlap at all.

`sub_em` has the opposite failure mode to F1: it is too permissive rather than too strict. It passes if the gold is a substring of the prediction or vice versa, which means "the answer is 1944 or 1945" scores `sub_em=1.0` for gold "1945" — the right token is present but surrounded by wrong content. Worse, for tasks where order carries meaning (ranked lists, step sequences, structured outputs), substring containment says nothing about whether the ordering is correct.

All three metrics therefore have distinct failure modes:

| Metric | Too strict when | Too loose when |
|---|---|---|
| `hard` (EM) | valid answer has different surface form or word order | — |
| `soft` (F1) | — | order matters; extra tokens inflate recall |
| `sub_em` | gold and prediction share tokens but in different order | prediction contains gold plus additional wrong content |

None is universally correct. The right choice depends on task structure: factoid QA with short unambiguous answers → EM works well; open generation or multi-part answers → domain-calibrated F1 threshold; ordered sequences or structured outputs → neither token metric is adequate and an LLM-as-judge scorer is more appropriate. The framework delegates this entirely to each benchmark's `evaluate()` but provides no guardrails against using the wrong metric as the primary training signal.

**The three metrics are not combined — `hard` controls everything**

Despite computing all three scores, the training loop is effectively 1-dimensional. `hard` (mean EM on the val set) is the only value that drives every decision:

- **Gate**: accepts a candidate skill only if `cand_hard > current_score`; `best_score` and `current_score` are both mean hard
- **Reflect split**: `failures = [r for r in results if not r.get("hard")]` — purely binary on `hard`
- **Slow update categorization**: `improved`/`regressed`/`persistent_fail`/`stable_success` are determined by hard scores alone; soft scores appear in the comparison text the optimizer reads but as context, not as the decision variable

`soft` and `sub_em` are computed, logged, and saved to `step_record.json` but have zero influence on training dynamics. They exist for human inspection only. This means all the failure modes described above — EM's order-sensitivity, F1's permissiveness toward extra tokens, sub_em's permissiveness toward extra content — flow directly into training decisions with no mitigation from the other metrics.

**(c) Trajectory**

The full conversation is saved to `predictions/<task_id>/conversation.json`. The final entry is always a synthetic `{"role": "system", "content": "[EVALUATION RESULT]..."}` message appended after scoring — this means the analyst sees not just what the model did but also the exact score and why it failed, all in the same conversation record.

The system prompt and user prompt are also saved separately as `.txt` files so the analyst can see exactly what the agent saw.

**Batch-level aggregation**

After all tasks complete, `hard` and `soft` are averaged across the batch:
- `rollout_hard` = mean hard score (accuracy)
- `rollout_soft` = mean soft score (average F1)

These aggregates are logged and stored in `step_record.json`. The live print during rollout shows running accuracy: `[rollout] 23/40 (acc=0.612) id=abc hard=1`.

### 2. Reflect (Backward Pass)
*~4–6 optimizer calls per step (parallel) — `ceil(n_failures / minibatch_size)` + `ceil(n_successes / minibatch_size)`*

The reflect stage takes the rollout results and produces **edit patches** — structured JSON proposals for modifying the skill document. The core design is **minibatch analysis**: instead of analyzing each failed trajectory independently, trajectories are grouped into minibatches of M and analyzed together in one optimizer call. This is directly analogous to minibatch SGD — the optimizer sees cross-trajectory patterns rather than individual noise.

#### Splitting failures and successes

Results are first split by `hard`:
- `failures`: all results where `hard=0`
- `successes`: all results where `hard=1` (only if `failure_only=False`)

Each group is shuffled deterministically (using `batch_seed` for failures, `batch_seed+1` for successes) then sliced into minibatches of size `gradient.minibatch_size` (default 8).

With a batch of 40 and minibatch size 8: up to 5 failure minibatches + up to 5 success minibatches = up to 10 parallel optimizer calls.

#### Resume support

Before making any API calls, the reflect stage checks `patches/minibatch_fail_000.json`, `minibatch_fail_001.json`, etc. Any already-saved patch files are loaded directly, skipping those optimizer calls. Only the remaining minibatches are sent to the API. This makes the reflect stage fully resumable.

#### Failure analyst (error minibatch)

Each failure minibatch is sent to `run_error_analyst_minibatch`. The optimizer receives:

**System prompt** (`analyst_error.md`):
> You are an expert failure-analysis agent for AI agent tasks. You will be given MULTIPLE failed agent trajectories from a single minibatch and the current skill document. Your job is to identify the most important COMMON failure patterns across the batch and propose a concise set of skill edits.
>
> Analysis process:
> 1. Read ALL trajectories in the minibatch
> 2. Identify the most prevalent, systematic failure patterns
> 3. For each pattern, classify its failure type
> 4. Propose edits that address COMMON patterns — not individual edge cases
> 5. Edits must be generalizable; do not hardcode task-specific values
> 6. Only patch gaps in the skill — do not duplicate existing content

**User message** contains, in order:
1. The current skill document (`## Current Skill`)
2. The edit budget: `"Produce at most L=4 edits"` (`## Edit Budget`)
3. Step buffer context: a rolling summary of failure patterns and rejected edits from all previous steps in this epoch (`## Previous Steps in This Epoch`) — so the analyst knows what has already been tried
4. Meta skill context: optimizer-side memory from the previous epoch (`## Optimizer Meta Skill`) — advisory, not enforced
5. All formatted trajectories from this minibatch (`## Failed Trajectories (N total)`)

Each trajectory in the formatted batch includes:
- A header: task description, task type, failure reason, number of turns
- The hidden reference answer (if available)
- The target system prompt the agent saw (the skill document at time of rollout)
- The target user prompt (the question + context)
- The full conversation with `[action]`/`[obs]` or `[step N think]`/`[step N action]`/`[step N obs]` formatting
- The `[EVALUATION RESULT]` block showing predicted vs gold answer and exact scores

Trajectories are truncated to 12,000 chars each (head + tail, middle dropped). The combined minibatch text can be large; `max_completion_tokens=4096` for patch mode.

**Output** — the optimizer responds with JSON only:
```json
{
  "batch_size": 8,
  "failure_summary": [
    {"failure_type": "wrong entity extraction", "count": 5, "description": "..."},
    {"failure_type": "ignores second date", "count": 3, "description": "..."}
  ],
  "patch": {
    "reasoning": "...",
    "edits": [
      {"op": "append", "content": "## Date Handling\nWhen multiple dates appear..."},
      {"op": "replace", "target": "extract the answer directly", "content": "identify which entity the question asks about before extracting"},
      {"op": "insert_after", "target": "## General Strategy", "content": "..."},
      {"op": "delete", "target": "Always use the first occurrence of the answer"}
    ]
  }
}
```

Four edit operations are supported: `append` (add to end), `insert_after` (insert after a specific heading/text), `replace` (swap exact text), `delete` (remove exact text). The analyst is explicitly told not to propose edits targeting the `<!-- SLOW_UPDATE_START/END -->` protected region.

#### Success analyst (success minibatch)

Runs in parallel with failure analysts when `failure_only=False`. Uses `analyst_success.md`:
> You are an expert success-pattern analyst. Your job is to identify generalizable behavior patterns that are COMMON across the batch and worth encoding in the skill. Only propose patches for patterns NOT already covered in the skill.

Same user message structure, but the trajectories section reads `## Successful Trajectories`. Output has `success_patterns` instead of `failure_summary`, otherwise identical patch format. Tagged `source_type="success"` to distinguish from failure patches downstream.

#### Parallelism

All minibatch calls (failures + successes combined) are submitted to a `ThreadPoolExecutor` with `gradient.analyst_workers` (default 16) workers. As each future completes, the patch is saved to disk and the count logged: `[analyst] 3/8 minibatch_fail_002 (8 trajs) → 3 edits`.

#### Step buffer — cross-step analyst memory

Rather than re-running the analyzer within a step, SkillOpt passes context about previous steps to every future reflect call. After each step completes (accepted or rejected), a digest is appended to `step_buffer`:

- failure patterns extracted from the rollout
- if rejected: the specific edits that were tried and the resulting score drop

At the next step, `_format_step_buffer(step_buffer)` is injected into every analyst prompt as `## Previous Steps in This Epoch`. So by step 5, the analyst already knows what was tried and failed in steps 1–4 and won't propose the same edits again.

`gradient.max_analyst_rounds` (default 3) is configured and logged but is currently unused — it appears nowhere in the reflect, aggregate, or select code as an actual loop bound. There is no mechanism that re-runs the analyzer within a single step.

### 3. Aggregate
*~3 optimizer calls per step — failure merge + success merge + final combined merge*

All patches from the reflect stage are merged into a single coherent patch via **hierarchical LLM calls**, not semantic deduplication. The process runs in three phases:

1. **Failure merge** — all failure patches are merged independently using `merge_failure.md` as the system prompt, processed in parallel batches of `merge_batch_size` (default 8). If there are more patches than `merge_batch_size`, multiple levels of merging occur until one patch remains.
2. **Success merge** — same process for success patches using `merge_success.md`.
3. **Final merge** — the two merged groups are combined in one final LLM call using `merge_final.md`, with failure patches explicitly marked as higher priority than success patches.

Each merge call receives the current skill document and the patches to merge, and outputs a single consolidated patch. If the LLM call fails, a fallback concatenates all edits directly. Meta skill context is injected into every merge call.

### 4. Select (Gradient Clipping)
*0–2 optimizer calls per step — 0 if merged edits already ≤ edit_budget; 1 ranking call if over budget; +1 if `lr_control_mode=autonomous`*

All aggregated patches are ranked by relevance score. The top K are selected where K = `optimizer.learning_rate` (default 4 — max edits per step). This is directly analogous to gradient clipping: it prevents the skill document from changing too drastically in one step.

The `optimizer.lr_scheduler` decays K across training:

- `cosine` — high early (exploration), smooth taper toward end (recommended)
- `linear` — linear decay
- `constant` — fixed K throughout
- `autonomous` — the optimizer decides K itself each step

### 5. Update (Parameter Update)
*0 calls in default patch mode (deterministic string ops); 1 optimizer call if `skill_update_mode=rewrite_from_suggestions`*

Selected patches are applied to the skill document, producing a **candidate skill**. Step-level edits cannot touch the protected slow-update region (see below).

### 6. Gate (Validation)
*N_val target model calls — full rollout over the val split (cached by skill hash, so identical candidates skip re-evaluation)*

The candidate skill is evaluated on the **selection split** (val set). The update is accepted only if the candidate outperforms the current skill. Rejected candidates are discarded; training continues from the last accepted skill.

This acts as continuous early stopping at the step level.

---

## Epoch Boundary Mechanisms

After all steps in an epoch complete (starting from epoch 2), two additional stages run before the next epoch begins. Both use **longitudinal comparison** — running the same 20 training items through both the previous epoch's final skill and the current epoch's final skill to measure what changed.

The 20 items are sampled from the training split using seed `= train.seed + epoch * 2000`, making the sample deterministic and epoch-specific but different each epoch.

---

## Slow Update
*~41 calls per epoch boundary — 20 target calls (prev skill rollout) + 20 target calls (curr skill rollout) + 1 optimizer call*

**Analogy:** Momentum. Prevents the agent from forgetting improvements made in earlier epochs.

**What it does:** Generates high-level task-facing guidance and force-injects it into a protected region of the skill document at each epoch boundary.

### Step-by-step

**1. Parallel rollout**

Both the previous epoch's final skill and the current epoch's final skill are run against the same 20 sampled training items. Each item gets two scores.

**2. Longitudinal categorization**

Each item is categorized by how its outcome changed:

| Category | Previous | Current | Priority |
|---|---|---|---|
| `regressed` | correct | wrong | Highest |
| `improved` | wrong | correct | — |
| `persistent_fail` | wrong | wrong | — |
| `stable_success` | correct | correct | Lowest |

Items are formatted into a structured comparison report. The report opens with a summary block, then four sections in priority order — regressions first, then persistent failures, improvements, and stable successes. Stable successes get no trajectories; all other categories include both the previous and current epoch trajectory (each truncated to 3000 chars, head+tail).

A concrete example of what the optimizer sees:

````
## Longitudinal Comparison Summary
Total samples: 20
- Improved (wrong→right): 4
- Regressed (right→wrong): 3
- Persistent failures (wrong→wrong): 8
- Stable successes (right→right): 5

### Regressions (right→wrong) — HIGHEST PRIORITY

#### Task abc123: Who signed the Treaty of Versailles?
- Prev epoch: PASS (soft=1.00) — answer: Woodrow Wilson
- Curr epoch: FAIL (soft=0.00) — answer: Georges Clemenceau
- Curr fail reason: EM=0: predicted 'Georges Clemenceau' but expected ['Woodrow Wilson']

**Previous epoch trajectory:**
```
[user] Who signed the Treaty of Versailles?
[assistant] Based on the context... <answer>Woodrow Wilson</answer>
[verification] [EVALUATION RESULT] Predicted: 'Woodrow Wilson' EM: 1.0 F1: 1.0000
```

**Current epoch trajectory:**
```
[user] Who signed the Treaty of Versailles?
[assistant] The treaty was signed by multiple parties... <answer>Georges Clemenceau</answer>
[verification] [EVALUATION RESULT] Predicted: 'Georges Clemenceau' EM: 0.0 F1: 0.0000
```

### Persistent Failures (wrong→wrong)
...

### Improvements (wrong→right)
...

### Stable Successes (right→right)
(no trajectories shown)
````

**3. Optimizer LLM call**

The optimizer receives:

- Previous epoch's final skill (truncated to 6000 chars)
- Current epoch's final skill (truncated to 6000 chars)
- The **previous slow update guidance** that was active during this epoch — so the optimizer can reflect on whether its past guidance actually helped
- The full categorized comparison report

The optimizer is prompted as a "strategic skill advisor" and outputs:

```json
{
  "reasoning": "...",
  "slow_update_content": "..."
}
```

`slow_update_content` is actionable task-facing text: concrete rules or strategies the agent should follow (e.g. "When the passage contains multiple dates, always identify which one the question is asking about before answering").

**4. Force-injection**

The `slow_update_content` is written into a protected HTML-comment-delimited region in the skill document:

```
<!-- SLOW_UPDATE_START -->
...guidance here...
<!-- SLOW_UPDATE_END -->
```

This injection is **not gated** — it always happens regardless of whether it improves val score. It is applied to both `current_skill` and `best_skill`.

**5. Protection from step-level edits**

Step-level apply_edit skips any edit targeting content inside the protected region (`status: "skipped_protected_slow_update_region"`). Append operations automatically insert content before the region to avoid collision.

### Key property

Because it is force-injected and ungated, slow update can override the step-level optimizer. It is designed for strategic, epoch-level corrections that the step-by-step patch system cannot see — particularly regressions where a previously learned skill was overwritten.

### Design concerns

**Too much power for an ungated mechanism.** Slow update writes unconditionally to both `current_skill` and `best_skill` — including your best checkpoint — with no validation gate. The protected region is immune to step-level correction for the entire next epoch. If the guidance is wrong, nothing can fix it until the next epoch boundary. The only safeguard is the optimizer's self-reflection ("did my previous guidance help?"), which is itself based on just 20 samples and is not independently validated.

This inverts the cost/risk relationship of the rest of the system: step-level edits are cheap and gated; slow update is expensive and ungated. Higher-cost mechanisms should have more validation, not less.

**Too heavy.** The 40 fresh target calls (20 for prev skill, 20 for curr skill) are paid every epoch boundary on top of normal training — roughly an extra half-batch rollout — solely to generate comparison data for one ungated optimizer call. Unlike the gate rollout, which validates a candidate before accepting it, these calls produce data that feeds a decision with no acceptance condition.

A more defensible design would reuse rollout data already computed during normal training steps (eliminating the 40 extra calls) and gate the injection against a small held-out sample before writing to `best_skill`.

**`best_skill` integrity is silently broken.** `best_skill` is the highest-scoring checkpoint validated by the gate — its `best_score` was earned by a specific skill that ran on the val set and beat all prior candidates. After slow update injects new guidance into it, that validated skill no longer exists on disk. The `best_score` in history now refers to a skill that has been overwritten. The actual `best_skill` file has never been evaluated — it could score higher or lower than `best_score`, and nothing in the system tracks this. At the end of training, the logged `best_score` describes a skill that is gone. The correct fix would be to re-evaluate `best_skill` after injection and update `best_score` accordingly, or at minimum flag it as stale.

---

## Meta Skill
*1 optimizer call per epoch boundary — reuses slow update's rollout data, no extra target calls*

**Analogy:** Meta-learning. Cross-epoch memory that improves how the optimizer generates and ranks edits, not what the agent does.

**What it does:** After each epoch, generates compact optimizer-facing memory and injects it into every reflect call throughout the next epoch.

### Step-by-step

**1. Same longitudinal comparison**

Reuses the same 20-item comparison data from slow update (or builds fresh if slow update was disabled). Also loads the previous epoch's meta skill content for self-reflection.

**2. Optimizer LLM call**

The optimizer receives:

- Previous epoch's final skill
- Current epoch's final skill
- The **previous meta skill** — so the optimizer can reflect on whether its past editing advice was useful
- The longitudinal comparison report (same format as slow update)

The optimizer is prompted as an "optimizer-coach" writing memory **for the future optimizer**, not for the agent. The focus is on editing strategy: which edit patterns caused regressions, what abstraction levels work, which failure-repair approaches to prioritize. Outputs:

```json
{
  "reasoning": "...",
  "meta_skill_content": "..."
}
```

**3. Injection into every reflect call**

At the start of each epoch, the meta skill from the previous epoch is loaded once and held fixed for the entire epoch. It is passed unchanged to every `reflect()`, aggregate, select, and LR-decision call across all steps — so if an epoch has 10 steps, all 10 use the exact same meta skill content. It does not update mid-epoch regardless of what happens during training. Inside analyst prompts it appears as:

> **Optimizer Meta Skill**
> This is optimizer-side memory distilled from prior epoch transitions in this environment. Use it to improve how you propose, merge, and rank skill edits. Prefer it when the current evidence is ambiguous, but do not force it if the current trajectories clearly contradict it.
>
> `{meta_skill_content}`

It is also passed to the aggregation, selection, and LR-decision stages — not just reflect.

This means meta skill influences four separate optimizer decisions per step: what edits get proposed, how they get merged, which ones get ranked highest, and how many get applied. A single piece of meta-level advice compounds through the entire pipeline. If the meta skill says "prefer concise rules over detailed ones", it nudges the analyst to propose concise edits, the merger to favour them when combining patches, the ranker to pick them from the merged pool, and potentially the LR decision too. This could be intentional amplification of good insights — or it means a wrong meta skill has four times the damage it should. The scope is broader than "optimizer-side memory for the reflect step" suggests.

**4. Persistence**

Saved to `meta_skill/epoch_XX/meta_skill_result.json`. Loaded fresh at the start of the next epoch. Not gated — always saved, always loaded.

### Key property

Meta skill is **advisory**, not enforced. Analysts are instructed to prefer it when evidence is ambiguous but to override it if current trajectories clearly contradict it. This prevents stale editing advice from overriding fresh signal.

---

## Slow Update vs Meta Skill

| | Slow Update | Meta Skill |
|---|---|---|
| **Target** | The skill document (what the agent does) | The optimizer (how edits are generated) |
| **Content** | Task-facing instructions | Editing strategy principles |
| **Gating** | Force-injected, no gate | Advisory only |
| **When active** | Epoch boundary only | Every step throughout the epoch |
| **Protected region** | Yes — step edits cannot touch it | No |
| **Reflects on** | Previous slow update guidance | Previous meta skill content |
| **Analogy** | Momentum | Meta-learning |

Together they form a two-level learning system: slow update corrects the agent's behavior based on what regressed; meta skill corrects the optimizer's editing strategy based on what edit patterns caused regressions.
