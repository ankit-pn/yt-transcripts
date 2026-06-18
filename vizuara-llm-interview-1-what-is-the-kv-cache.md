# The LLM Interview Series #1 — What exactly is the KV Cache?

**Source:** [https://www.youtube.com/watch?v=CxRGWfcGVbs](https://www.youtube.com/watch?v=CxRGWfcGVbs)  
**Channel:** Vizuara  
**Series:** The LLM Interview Series (#1)

> Transcript derived from YouTube's auto-generated captions, then hand-edited for readability and accuracy: punctuation, paragraphing, matrix dimensions, and technical terms (KV cache, prefill/decode, logits, grouped-query attention, etc.) have been corrected. Because this is a whiteboard walkthrough, the wording has been lightly condensed where the spoken delivery was repetitive; the content and order are kept faithful.

---

## Introduction

Hello everyone, and welcome to this video. I'm going to explain the answer to a question that's asked a lot in interviews. I'm starting a series where, in each video, I take one question that comes up a lot in large language model (or AI) interviews and explain it on a whiteboard, from scratch and in detail, so you can really understand what's happening under the hood. Until now, all of Vizuara's lectures have explained foundational concepts in depth, but I thought I'd try this new approach where I write everything down in front of you, so these videos can serve to revise your understanding. Each video is dedicated to one challenging-but-common interview question.

Today's topic is the **key-value (KV) cache**. "What exactly is a key-value cache?" is asked a lot by several companies, and many people struggle to explain it from first principles. So I'm going to write it all down on the board to clarify the answer.

## The KV cache is an inference concept

The first thing to notice: the KV cache shows up during **LLM inference** — not during pre-training, but during the inference pipeline. Remember that.

So what is inference? You take some tokens, pass them to a language model, and the model produces output. Take the example input **"A sunset is"** — three tokens (token 1 = "a", token 2 = "sunset", token 3 = "is"). These pass through an LLM, and I get output, say **"A sunset is extremely beautiful"** — those are my output tokens (there can be many). That whole process is inference.

To understand the KV cache, realize that inference happens in **two stages**:

1. **Prefill stage** — the LLM consumes all the input tokens, forms cross-connections between them using the attention mechanism, and predicts the **first** output token. All input tokens are processed **in parallel**; it's a **compute-bound** operation.
2. **Decode stage** — **one new token** is produced at a time, after the first.

So until the first token is produced we're in prefill; for every subsequent token we're in decode. To explain the KV cache to an interviewer, you first have to explain how prefill and decode work — right down to the matrix-multiplication level, because that's the only way to really understand it.

## The prefill stage, at the matrix level

Our input is "A sunset is." A computer doesn't understand English, so each token becomes a vector — say a **4-dimensional** vector. Stacking the three token vectors gives the input matrix **X**, which is **3×4**.

In prefill, X is first multiplied by three trainable weight matrices — **W_Q, W_K, W_V** (query, key, value). Pre-training is already done, so these weights are fixed. Each is **4×4**. The multiplications:

- X (3×4) · W_Q (4×4) → **Query matrix Q**, which is **3×4**
- X (3×4) · W_K (4×4) → **Key matrix K**, which is **3×4**
- X (3×4) · W_V (4×4) → **Value matrix V**, which is **3×4**

These three matrices are at the heart of understanding the KV cache.

Next, multiply the query matrix by the **transpose of the keys** to get the **attention scores**: Q (3×4) · Kᵀ (4×3) → a **3×3** attention-scores matrix. Simply put, this tells you, for every token, how important the other (preceding) tokens are. With three tokens "a / sunset / is," each element is the attention score between a pair — e.g., the score between "sunset" and "is," or between "a" and "is." The attention-scores matrix is the **only place in the architecture where tokens learn something about their neighbors**.

From the scores we get the **attention weights** by applying a softmax (with causal masking, and dividing by the square root of the key dimension). The attention-weights matrix is again **3×3**.

Then multiply the attention weights by the values: AttnWeights (3×3) · V (3×4) → the **context matrix**, which is **3×4**.

Now the key insight about prediction: **to predict the next token, you only need the last row** of the context matrix — the last context vector. You take that last row and multiply it by the **logits** matrix (you don't need its details right now), which gives you the vector for the next token. Here, the next token is **"extremely."**

So the prefill pipeline is: input matrix → multiply by trainable W_Q, W_K, W_V → get Q, K, V → Q·Kᵀ = attention scores → normalize = attention weights → ·V = context matrix → last row · logits → next token ("extremely"). All input tokens are processed in parallel; one new token is produced.

## The decode stage, done naively

Now we have **four** tokens, because "extremely" was just predicted: "a sunset is extremely." Each is a 4-D vector, so the input matrix is now **4×4** (it was 3×4 in prefill).

Forget the KV cache for a moment — how would you predict the next token? You'd do exactly the same thing:

- X (4×4) · W_Q, W_K, W_V (each 4×4) → Q, K, V, each **4×4**
- Q · Kᵀ → attention scores, now **4×4** (it was 3×3 with three tokens)
- normalize → attention weights, **4×4**
- · V → context matrix, **4×4**
- take the **last row** → · logits → next token, which is **"beautiful."**

So prefill and decode are essentially exact copies of each other. Job well done, you tell the interviewer — but where did the KV cache come in?

## Insight 1: you only need the last row of the context matrix

Let's reverse-engineer it. To predict the next token, you only need the **last row** of the context matrix — none of the other rows. So what do you need to produce just that last row?

The context matrix = attention weights · values. To get only the **last** context row, you only need the **last row of the attention weights** and the **entire values matrix**: (1×4) · (4×4) → (1×4). Why? Every row of the context matrix corresponds to one token (row 1 = "a", row 2 = "sunset", row 3 = "is", row 4 = "extremely"), and likewise every row of the attention weights corresponds to one token. If you want the context vector only of "extremely" (the last token), you only need the attention weights of that last row.

Going further back: to get the last row of attention **weights**, you need the last row of attention **scores** (1×4) — converting scores to weights is easy. And how is each row of the scores computed? Attention scores = Q · Kᵀ, so:
- row 1 = Q₁ · Kᵀ, row 2 = Q₂ · Kᵀ, … row 4 = Q₄ · Kᵀ.

So to get the fourth row, you only need **Q₄** (the query vector for "extremely") — not Q₁, Q₂, Q₃ — and the **entire keys matrix**.

**Putting it together — what you actually need to predict the next token in decode:**
- the **query vector** for the latest token only (1×4),
- the **entire keys matrix** (4×4),
- the **entire values matrix** (4×4).

The pipeline: take the query vector for the latest token ("extremely") → multiply by the entire Kᵀ → attention scores for the latest token → normalize → attention weights for the latest token → multiply by the entire V → context vector for the latest token → logits → next token. The one thing already simplified: you **don't need the entire query matrix, only the query vector** for the new token.

## Insight 2: the earlier keys and values were already computed — cache them

We need the entire keys and values matrices — but do we need to **recompute** them? Look at the 4×4 K and V in decode. The **first three rows** of each (a 3×4 block) correspond to the first three tokens ("a sunset is") — and we **already computed exactly those** in the prefill stage. We already have the key vectors and value vectors for the first three tokens. So why compute them again?

The key idea: **cache the previously computed key and value matrices.** When you do the prefill, you store K and V in memory. When you reach decode — where you need the entire K and entire V — you take the first three (cached) rows and only **compute the last row** anew. Everything before is cached.

## The actual (cached) decode pipeline

This is the real decode (versus the naive version above):

1. You've predicted "a sunset is extremely" and want the next token. Take the input embedding for **only the last token** ("extremely"), which is **1×4** — you don't touch the full 4×4 input at all.
2. Multiply it by W_Q, W_K, W_V (each 4×4) → the **query vector** (1×4), **key vector** (1×4), and **value vector** (1×4) for "extremely" only.
3. You already **cached** the keys and values for the previous three tokens: a **3×4 key cache** and a **3×4 value cache**. **Append** the new key vector → **K_full** (4×4), and append the new value vector → **V_full** (4×4). In each, only the **last row is new**; the previous three rows are cached.
4. Take the query vector for "extremely" (1×4) — we never need the full query matrix — multiply by **K_fullᵀ** → attention scores (1×4) → attention weights (1×4).
5. Multiply by **V_full** → the context vector for the last token only ("extremely") → logits → next token.

So someone who hasn't studied the KV cache pictures decode as computing the full K and V, full attention scores and weights, the entire context matrix, then taking just the last row. Someone who **has** studied it knows two realizations:

1. To get the next token, you only need the **last row** of the context matrix — which, back-calculated, means the entire V, the entire K, and only the **query vector**.
2. To get the entire K and V, you **don't recompute** them — you reuse the **cached** K and V, compute only the key and value vector for the new token, and append. So K_full and V_full = cache + the new vectors. Then query-vector-of-new-token · K_fullᵀ → scores/weights for the latest token → context vector for the latest token → logits → next token.

## Why there's no query cache

Why only a KV cache — why not cache Q too (QKV)? Because, as we saw, the ingredients to compute the next token don't include the **entire query matrix** — you only need the **query vector for the new token**. You **do** need the entire keys and values matrices. That's why caching applies only to keys and values.

## Why it matters

What does the cache do for us? It **saves redundant computation**. Instead of a quadratically increasing compute curve, you get a much slower, roughly **linear** one. (I'll do a separate video on the computational complexity with and without the KV cache.) The KV cache saves a huge number of computations during inference — it makes inference faster by reducing the number of floating-point operations.

That directly helps end-user applications. With a chatbot, you want a low **time-to-first-token** and low **inter-token latency** for a fast experience. If you do redundant computation, decoding each token takes longer because more floating-point operations are needed; the KV cache eliminates that.

## The dark side (and what's next)

Of course, the KV cache has a **dark side**: it takes **memory** to store, and that memory must be transferred from high-bandwidth memory (HBM) to where the compute happens on the GPU — which is not great for inference. I'll cover the dark side in a later video.

So understanding the KV cache comes down to two concepts: (1) you only need the context vector of the **last token**, and (2) you don't need to recompute the full key and value matrices — but for both, you need to understand that inference splits into **prefill** and **decode**.

In front of an interviewer you'll need to write the **matrix-level** implementation — staying high-level isn't enough; the matrix details make it easy to tell who has the foundational knowledge and who doesn't. And if they ask about the tradeoffs (the "dark side"), it's essentially the huge memory requirements of the KV cache — which lead to mechanisms like **multi-query attention**, **grouped-query attention**, and **multi-head latent attention** (topics for another video).

Thank you, everyone. These questions seem straightforward — "to do inference, I just cache the keys and values" — but that's not good enough; you need to explain it at the matrix level like this. I look forward to seeing you in subsequent videos.
