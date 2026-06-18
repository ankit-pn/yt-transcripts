# The KV Cache, From First Principles — an explanation built to last a lifetime

> A companion deep-dive to [The LLM Interview Series #1 transcript](vizuara-llm-interview-1-what-is-the-kv-cache.md). The transcript is the whiteboard walkthrough; this file is the slow, self-contained version that assumes nothing and leaves no gaps. Read it once, carefully, and the KV cache should never confuse you again.

---

## How to read this

Don't skim. Each section earns the next. We use **one running example** the whole way through — the prompt **"A sunset is"** — and the **same tiny numbers**, so nothing ever resets on you. By the end you'll be able to *derive* the KV cache yourself, not just recite it.

The single sentence we are going to *earn* (not memorize) is:

> **A language model generates text one token at a time, and at every step it secretly redoes almost all the work it already did. The KV cache is the realization that the only thing genuinely new each step is the latest token — so you store the rest and never recompute it.**

If that already makes sense, great — the rest is proof. If it doesn't yet, that's exactly what we'll fix.

---

## 1. What "generating text" actually is

A large language model (LLM) of the GPT family is an **autoregressive** model. That word looks scary; it means one simple thing:

> It produces text **one token at a time**, and each new token is predicted from **all the tokens that came before it** (including the ones it just generated).

A *token* is a chunk of text (a word or word-piece). "A sunset is extremely beautiful" might be 5 tokens: `A` · `sunset` · `is` · `extremely` · `beautiful`.

The loop is literally:

```
prompt: "A sunset is"
step 1: read [A, sunset, is]                → predict "extremely"
step 2: read [A, sunset, is, extremely]     → predict "beautiful"
step 3: read [A, sunset, is, extremely, beautiful] → predict "."
... until the model emits a STOP token.
```

Stare at the left column. Step 2 re-reads everything step 1 read, **plus one new token**. Step 3 re-reads everything again, plus one more. This re-reading is where all the waste lives — and where the KV cache will save us. Hold that thought.

Two more facts to nail down before we look inside:

- **Inference vs. training.** All of this is *inference* (using a trained model). Training already happened; the model's weights are **frozen**. The KV cache is purely an inference trick — it has nothing to do with how the model learned.
- **The model can't read letters.** Before anything, each token is turned into a list of numbers — a **vector** — called an *embedding*. In our examples we'll pretend each token becomes a tiny **4-dimensional** vector (real models use thousands of dimensions, but 4 is enough to see everything).

So "A sunset is" becomes a stack of three 4-dimensional vectors — a matrix we'll call **X** with shape **3 × 4** (3 tokens, 4 numbers each).

```
        ┌                    ┐
   A    │  x x x x  │   ← row for "A"
 sunset │  x x x x  │   ← row for "sunset"
   is   │  x x x x  │   ← row for "is"
        └                    ┘
        X  (3 rows × 4 columns)
```

**Rule that will matter forever:** in all these matrices, **one row = one token**. Row order = token order. Burn that in.

---

## 2. Inside one step: attention from first principles

To predict the next token, the model runs the tokens through **attention**. Here is the entire mechanism, slowly. (Real models stack many *layers* of this; we'll do one layer now and add the layers back in Section 10 — the logic is identical.)

### 2a. Three projections: Query, Key, Value

The model owns three fixed weight matrices, learned during training and now **frozen**: **W_Q**, **W_K**, **W_V**. Each is 4 × 4 in our toy size. We multiply our input **X** by each of them:

```
Q = X · W_Q     (3×4 · 4×4 = 3×4)   "queries"
K = X · W_K     (3×4 · 4×4 = 3×4)   "keys"
V = X · W_V     (3×4 · 4×4 = 3×4)   "values"
```

We now have three matrices Q, K, V, each 3 × 4 — one row per token, just like X.

What do they *mean*? The classic analogy: it's like a library search.
- A token's **query** = "here's what I'm looking for."
- A token's **key** = "here's what I'm about / what I can offer."
- A token's **value** = "here's the actual information I'll hand over if you attend to me."

### 2b. Scores: how much should each token attend to each other token?

Multiply the queries by the **transpose** of the keys:

```
Scores = Q · Kᵀ     (3×4 · 4×3 = 3×3)
```

A 3 × 3 matrix. Entry (i, j) is the dot product of token *i*'s query with token *j*'s key — a number saying **"how relevant is token j to token i?"**

```
            key:A   key:sunset  key:is
 query:A    [ .      .          .   ]
 query:sun  [ .      .          .   ]
 query:is   [ .      .          .   ]
```

This scores matrix is **the only place in the whole architecture where tokens look at each other.** Everything else processes each token independently. Remember that — it tells you exactly where "context" is created.

### 2c. The causal mask: you may only look *backward*

In text generation, token *i* must not peek at tokens that come **after** it (they don't exist yet when you're predicting). So before going further, we **mask** the upper triangle of the scores — set those forbidden entries to −∞ so they'll vanish in the next step:

```
            key:A   key:sunset  key:is
 query:A    [  s     -∞         -∞  ]   "A" can only see itself
 query:sun  [  s      s         -∞  ]   "sunset" sees A, sunset
 query:is   [  s      s          s  ]   "is" sees A, sunset, is
```

This **causal** rule is not a detail — in Section 3 it becomes the entire reason the KV cache is even possible. Underline it.

### 2d. Weights: turn scores into proportions

Apply a **softmax** to each row (after dividing by √(key-dimension), a scaling step). Softmax turns each row of raw scores into positive numbers that sum to 1 — i.e., "what fraction of my attention goes to each earlier token." Masked −∞ entries become 0. Call the result **Attention Weights**, still 3 × 3.

### 2e. Context: mix the values using those proportions

Multiply the attention weights by the **values**:

```
Context = Weights · V     (3×3 · 3×4 = 3×4)
```

A 3 × 4 matrix — again **one row per token**. Row *i* is a blend of all the value vectors, weighted by how much token *i* attends to each. This row is token *i*'s **context vector**: its meaning *in light of the tokens around it*.

### 2f. From context to the next token

To actually predict a token, take a context vector and multiply it by the output/unembedding matrix (the **logits** matrix). That produces a score over the whole vocabulary; the highest-scoring entry (roughly) is the next token. For "A sunset is", the model emits **"extremely"**.

**The pipeline of one step, end to end:**

```
X → (×W_Q,W_K,W_V) → Q,K,V → (Q·Kᵀ, mask) → Scores
  → (softmax) → Weights → (×V) → Context → (×logits) → next token
```

That's the whole machine. Now the two insights that turn it into the KV cache.

---

## 3. The property nobody emphasizes: a token's Key and Value are *frozen the moment the token exists*

This is the keystone. If you remember only one paragraph, remember this one.

Look again at how we built the keys and values:

```
K = X · W_K        →   the key for token i is:   K_i = x_i · W_K
V = X · W_V        →   the value for token i is:  V_i = x_i · W_V
```

The key for token *i* depends **only on token *i*'s own embedding x_i** and the frozen weights. It does **not** depend on any later token. Same for the value. (And because attention is **causal** — Section 2c — no later token is ever allowed to change an earlier token's contribution.)

**Consequence:** once you have computed `K_sunset` and `V_sunset`, they are correct *forever*. Adding "extremely" later does not change them. Adding "beautiful" later does not change them.

> The keys and values of past tokens are **immutable**. That is *why* you are allowed to store them and reuse them. The KV cache is not a hack that "happens to work" — it's the direct consequence of keys/values being per-token and attention being causal.

(Queries are different — more on that in Section 9. And yes, position information, e.g. RoPE, is baked into each token's key when it's created; that's fine, because a token's position also never changes.)

Keep this fact in your pocket. We'll cash it in shortly.

---

## 4. The naive generation loop, and the waste

Now generate **with no cache**, exactly as a beginner would.

**Step 1 (prompt "A sunset is", 3 tokens).** Run the whole pipeline. X is 3×4 → Q,K,V are 3×4 → Scores 3×3 → Weights 3×3 → Context **3×4**. Predict "extremely". ✅

**Step 2 (now "A sunset is extremely", 4 tokens).** A beginner reruns the *entire* pipeline from scratch on all four tokens:

```
X is now 4×4 → Q,K,V are 4×4 → Scores 4×4 → Weights 4×4 → Context 4×4 → predict "beautiful"
```

Prefill and this naive decode are **exact copies of each other**, just bigger. Job apparently done.

But now squint. In step 2 we recomputed **Q, K, V for "A", "sunset", "is"** — the first three rows — even though we *already computed those exact same numbers in step 1*. By Section 3, those rows could not possibly have changed. We threw away step 1's work and redid it. Every future step redoes ever more. **This is the waste.** The KV cache is just the decision to stop doing it.

Two insights remove the waste. Here they are.

---

## 5. Insight #1 — to predict the next token, you only need the **last row** of the Context matrix

In step 2, the Context matrix is 4×4: one context row each for `A`, `sunset`, `is`, `extremely`. Which row do we feed to the logits to get the *next* token?

Only the **last one** — the context vector of `extremely`, the most recent token. The next token is whatever should follow the sequence *so far*, and "so far" ends at `extremely`. The context rows for `A`, `sunset`, `is` are not used at all for this prediction.

> **Insight #1: you need exactly one context vector — the last row.** The other rows of the Context matrix are dead weight during generation.

(In *prefill* we still compute all rows, because attention computes them in parallel anyway and it costs nothing extra to get them — but we only *consume* the last one. From the second token onward, computing the other rows is pure waste.)

Now ask the natural question: **what is the minimum needed to produce just that last row?** Reverse-engineer it.

---

## 6. Reverse-engineering the last row

Walk backward through the pipeline, asking at each arrow "what part do I actually need?"

**Context's last row** comes from `Weights · V`. To get **only the last row of Context**, you need **only the last row of Weights**, multiplied by the **whole V**:

```
last Context row  =  (last row of Weights, 1×4)  ·  (whole V, 4×4)  =  1×4   ✅
```

Why the *whole* V but only *one* row of Weights? Because the last token's context is a blend of **all** earlier value vectors (it may attend to any past token), but it has only **one** set of attention proportions — its own.

**The last row of Weights** comes from softmax of **the last row of Scores** (1×4). Easy.

**The last row of Scores** comes from `Q · Kᵀ`. Row *i* of Scores = `Q_i · Kᵀ`. So the **last** row of Scores = **(last query vector) · (whole Kᵀ)**:

```
last Scores row  =  (last query, 1×4)  ·  (whole Kᵀ, 4×4)  =  1×4   ✅
```

Again: only **one** query (the latest token's), but the **whole** K (it scores itself against every past key).

**Collect the shopping list.** To produce the next token in decode, you need exactly:

```
• the query VECTOR of the latest token   (1×4)   ← just one row
• the ENTIRE keys matrix                 (n×4)
• the ENTIRE values matrix               (n×4)
```

Already a win: you do **not** need the whole query matrix — one query row suffices. But the entire K and entire V are still required... so where's the saving on those? Insight #2.

---

## 7. Insight #2 — the entire K and V are *needed*, but they don't need to be *recomputed*

Here's the 4×4 keys matrix we need in step 2. Draw a box around its first three rows:

```
 K (4×4)          rows correspond to:
 ┌─────────┐
 │  K_A    │  ┐
 │ K_sunset│  │  these three rows = the first three tokens
 │  K_is   │  ┘
 │K_extreme│  ← only this row is genuinely new
 └─────────┘
```

Question: **have we computed `K_A`, `K_sunset`, `K_is` before?** Yes — in **step 1's prefill**, when we processed exactly those three tokens. And by Section 3 they are *immutable*. Same story for V.

> **Insight #2: don't recompute the past keys/values — store them.** When you finish a step, keep the K and V you computed. On the next step, the only new key and value to compute belong to the **single new token**. Append them to the stored ones.

That store is the **KV cache**.

```
K_full (this step) =  [ cached K of all earlier tokens ]  +  [ K of the new token ]
V_full (this step) =  [ cached V of all earlier tokens ]  +  [ V of the new token ]
```

Each step: **one** new key vector, **one** new value vector, appended to a growing drawer. Nothing old is ever recomputed.

---

## 8. The real decode loop (with our example)

Put Insights #1 and #2 together. Here is decode **as it actually runs**, predicting "beautiful" after "A sunset is extremely":

1. **Take only the newest token's embedding.** Input is `x_extremely` (1×4) — *not* the whole 4×4 stack. You never re-embed or re-project the old tokens.
2. **Project just that one token:**
   ```
   q_new = x_extremely · W_Q   (1×4)
   k_new = x_extremely · W_K   (1×4)
   v_new = x_extremely · W_V   (1×4)
   ```
3. **Update the cache (append):**
   ```
   K_full = [ K_cache (3×4) ;  k_new (1×4) ]  → 4×4   (only last row is new)
   V_full = [ V_cache (3×4) ;  v_new (1×4) ]  → 4×4   (only last row is new)
   ```
4. **Attention for the new token only:**
   ```
   scores  = q_new · K_fullᵀ      (1×4 · 4×4 = 1×4)
   weights = softmax(scores)       (1×4)
   context = weights · V_full      (1×4 · 4×4 = 1×4)   ← the last context vector
   ```
5. **Predict:** `context · logits → "beautiful"`. Then `k_new`, `v_new` are now part of the cache for the next step.

Compare the two mental pictures:

| | Naive decode | Cached decode |
|---|---|---|
| Input each step | the **whole** sequence (n×4) | **one** token (1×4) |
| Q, K, V computed | for **all** n tokens | for the **1** new token |
| Scores | full n×n | **1×n** (new query vs. all keys) |
| Context | full n×4, use last row | **1×4** directly |
| Old K/V | recomputed every step | **read from cache** |

Same answer, a fraction of the work. That difference *is* the KV cache. Notice there is no magic anywhere — every shortcut is forced by Insight #1 (only the last row matters) and Insight #2 (past K/V are immutable, so cache them).

---

## 9. Why a KV cache and not a QKV cache?

A very common interview follow-up: "Why don't we cache the queries too?"

Look at the shopping list from Section 6:

```
• query: only the latest token's query vector is ever needed
• keys:   the ENTIRE matrix is needed (every step)
• values: the ENTIRE matrix is needed (every step)
```

A past token's **query** (`q_A`, `q_sunset`, …) is used **once** — on the step that token was the newest — and then **never again**, because future steps only ever use the *newest* query. There is nothing to gain by storing it. Keys and values, by contrast, are consulted on **every** future step (every new token scores itself against all past keys and blends all past values). So you cache exactly the things that get reused: **K and V**. Same reasoning kills any idea of a "scores cache" — the scores involve the changing newest query, so they're recomputed (cheaply) each step.

That asymmetry — *queries are write-once, keys/values are read-forever* — is the cleanest one-line justification you can give an interviewer.

---

## 10. Prefill vs. decode, stated precisely (and across many layers)

Now we can define the two phases of inference crisply:

- **Prefill** — you have the whole prompt at once. Process all prompt tokens **in parallel** (one big matrix multiply per projection), build the **initial KV cache** for every prompt token, and emit the **first** new token. It's **compute-bound** (lots of FLOPs, done in parallel). Latency of this phase = **time to first token**.
- **Decode** — repeat, one token at a time: embed the newest token, project it (1 row), append to the cache, attend against the full cache, emit the next token. Each step is cheap in FLOPs but reads the whole growing cache; it's increasingly **memory-bound**. The gap between successive tokens = **inter-token latency**.

**The layer stack (important, so it never confuses you).** Real models have many transformer layers (e.g., 32). Everything above happens **independently at every layer**, and **each layer has its own KV cache**. In cached decode, only the single newest token flows **up** through the stack: at each layer it attends to *that layer's* cached K/V for all past positions, produces one output vector, and passes it to the next layer. You never push the old tokens back through the stack — their per-layer keys and values are already sitting in each layer's cache. So "one token in, one token out, attending to caches all the way up" is the correct picture for the entire model, not just one layer.

---

## 11. How much does it save? (quadratic → linear)

Let `n` = current sequence length, `d` = model dimension.

**Without a cache**, generating the token at position `n` means rebuilding everything for all `n` tokens:
- projections for all tokens: ∝ `n · d²`
- full attention: ∝ `n² · d`

Do that for every position from 1 to N and the totals balloon (roughly `N²·d²` of projection work and `N³·d` of attention work) — you are essentially **re-running prefill at every single step**.

**With a cache**, generating the token at position `n` only:
- projects **one** token: ∝ `d²`
- attends one query against `n` cached keys/values: ∝ `n · d`

So **per-step cost drops from "grows with n² (you redo the whole sequence)" to "one projection + a single 1×n attention."** Across the whole generation, the cache removes an entire factor of the sequence length from the work. The video's shorthand — *"a steep quadratic compute curve becomes a gentle linear one"* — is exactly this. Fewer floating-point operations ⇒ faster time-to-first-token and lower inter-token latency ⇒ a snappier chatbot. That's the whole reason the cache exists.

---

## 12. The price: memory (the "dark side")

The cache isn't free — you trade **compute** for **memory**. Every token, at every layer, you store a key vector and a value vector, and you keep them for the whole sequence. The size of the KV cache:

```
KV bytes ≈ 2 × batch × seq_len × n_layers × n_kv_heads × head_dim × bytes_per_number
            ▲                                              ▲
        K and V                              (n_kv_heads × head_dim = key/value width per layer)
```

The `2` is for **K and V**. `bytes_per_number` is 2 for fp16/bf16. For standard multi-head attention, `n_kv_heads × head_dim` equals the model dimension `d`, so a clean approximation is:

```
KV bytes ≈ 2 × n_layers × d × seq_len × batch × bytes
```

**A concrete number (Llama-2-7B-style):** `n_layers = 32`, `d = 4096`, fp16 (2 bytes).

```
per token, all layers:  2 × 32 × 4096 × 2  ≈  524,288 bytes  ≈  0.5 MB / token
for a 4,096-token context (batch 1):  0.5 MB × 4096  ≈  2 GB
batch of 8 users:                       ≈ 16 GB  — just for the cache
```

That's gigabytes of GPU memory **on top of** the model weights, and it grows **linearly with context length and with the number of concurrent users**. Worse, this cache must be shuttled from high-bandwidth memory (HBM) to the compute units every step, so during decode you're often **bottlenecked by memory bandwidth, not math**. This is why long-context, high-throughput serving is hard: the KV cache, not the model, is frequently the thing that fills the GPU.

---

## 13. What the memory pressure forces (the sequel)

Because the KV cache dominates memory, much of modern LLM-inference engineering is about **shrinking or managing it**:

- **Multi-Query Attention (MQA)** — all attention heads **share one** key/value head. Slashes `n_kv_heads` to 1, shrinking the cache dramatically (at a small quality cost).
- **Grouped-Query Attention (GQA)** — the middle ground: a few KV heads shared across groups of query heads. Used by Llama-2/3-70B and many others; near-MHA quality, much smaller cache.
- **Multi-Head Latent Attention (MLA)** — store a compressed *latent* of K/V and reconstruct on the fly (popularized by DeepSeek). Bigger savings still.
- **Paged KV cache (PagedAttention / vLLM)** — manage the cache in fixed-size "pages" like an operating system manages RAM, to avoid fragmentation and waste across many concurrent requests.
- **Quantizing the cache** — store K/V in 8-bit or 4-bit to halve/quarter the bytes.

Every one of these is a direct answer to Section 12. If you understand *why* the KV cache is big, these all read as obvious consequences rather than a zoo of acronyms.

---

## 14. The mental model that will stick forever

> **Imagine writing a story one word at a time, where to choose each next word you must consider how every earlier word relates to it.**
>
> - The **naive** writer, before adding each new word, **re-reads the entire story from page one and re-annotates every word**. Slow, and it gets slower as the story grows.
> - The **smart** writer gives every word, the moment it's written, two permanent index cards: a **Key card** ("what this word is about") and a **Value card** ("the information this word contributes"). These cards never change — because a word's meaning-as-written doesn't change when later words are added. The writer drops them into a **drawer** (the cache). To add the next word, they write **one** new pair of cards and skim the drawer. No re-reading.
>
> The drawer is the **KV cache**. The reason it works is that the cards are **write-once** (Section 3). The reason you don't also file the **Query** ("what *this* word was searching for") is that a query is used once, the moment its word is the newest, and never consulted again (Section 9). The reason the drawer eventually overflows the desk is Section 12.

Tokens = words. Key/Value cards = the cached K and V. The drawer = the cache. Re-reading from page one = naive decode. Filing one new card pair = cached decode. That picture contains the entire topic.

---

## 15. Common misconceptions, cleared

- **"The KV cache is part of the model / training."** No. The weights are frozen; the cache is an *inference-time* memory of intermediate vectors. Training doesn't use it.
- **"Caching changes the output."** No. It produces the **identical** tokens as the naive loop — it only avoids recomputing immutable values. It's an exact optimization, not an approximation.
- **"We cache the attention scores / weights."** No. Those depend on the *current* newest query, which changes every step; only K and V are reused, so only they are cached.
- **"We cache the queries too."** No (Section 9) — a query is used once and discarded.
- **"There's one cache for the model."** No — there's a separate K/V cache **per layer** (and per attention head). The newest token attends to caches at every layer as it moves up the stack.
- **"It makes everything cheaper."** It makes **compute** cheaper but **costs memory and memory-bandwidth** — which becomes the new bottleneck for long contexts and big batches (Sections 12–13).
- **"It works for any model."** It's specifically for **autoregressive, causally-masked** decoders (GPT-style). The causal mask is what makes past K/V immutable; bidirectional encoders (e.g., BERT) don't generate token-by-token this way.

---

## 16. One-page cheat sheet

```
WHAT
  KV cache = store each token's Key and Value vectors at every layer during
  generation, so you never recompute them for past tokens.

WHY IT'S POSSIBLE
  K_i = x_i·W_K and V_i = x_i·W_V depend only on token i; causal attention
  forbids later tokens from changing them ⇒ past K/V are IMMUTABLE ⇒ cacheable.

THE TWO INSIGHTS
  1. To predict the next token you need only the LAST row of the context matrix.
  2. That last row needs the WHOLE K and V — but past K/V were already computed,
     so cache them and compute only the new token's K and V each step.

WHY KV AND NOT QKV
  Queries are write-once (used only when their token is newest);
  Keys/Values are read on every future step. Cache what's reused.

PHASES
  Prefill: process whole prompt in parallel, build cache, emit 1st token (compute-bound, = time-to-first-token).
  Decode:  per step, 1 token in → project → append to cache → attend over cache → next token (memory-bound, = inter-token latency).

SAVINGS
  Per-step cost goes from "redo the whole sequence (∝ n²)" to "one projection + one 1×n attention (∝ n)."

THE COST (DARK SIDE)
  KV bytes ≈ 2 × n_layers × d × seq_len × batch × bytes_per_number.
  Llama-2-7B fp16 ≈ 0.5 MB/token ≈ 2 GB at 4k tokens. Grows with length & batch;
  becomes memory-bandwidth-bound.

THE SEQUELS (shrink the cache)
  MQA → GQA → MLA, plus paged cache (vLLM) and KV quantization.

ONE-LINE ANSWER FOR AN INTERVIEW
  "Generation redoes work every step; but each past token's key and value are
  immutable, and to predict the next token I only need the latest token's query
  against all keys/values. So I cache past K and V, compute only the new token's
  K/V, and append — turning a quadratic recompute into a linear one, at the cost
  of memory. I'd then write the prefill and decode matrix shapes on the board."
```

---

*If you want the same first-principles treatment for the "dark side" (the memory math, MQA/GQA/MLA, and PagedAttention) as its own file, that's the natural next chapter.*
