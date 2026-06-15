# Introduction to Reinforcement Learning and PPO for Robotics (VLA for Autonomous Driving series)

**Source:** [https://www.youtube.com/watch?v=tyip9W10pzc](https://www.youtube.com/watch?v=tyip9W10pzc)  
**Channel:** Vizuara  
**Series:** VLA (Vision-Language-Action) for Autonomous Driving

> Transcript derived from YouTube's auto-generated captions, then hand-edited for readability and accuracy: punctuation, paragraphing, and technical terms (PPO, MetaDrive, Gaussian, AlphaGo / Lee Sedol, Isaac Lab, GAE, TurboPi, Claude Code, etc.) have been corrected. Because this is a ~2-hour live teaching session, the wording has been lightly condensed where the spoken delivery was repetitive; the content and order are kept faithful, and audience questions from the chat are folded in as **Q/A** callouts.

---

## Introduction and logistics

Hello everyone, welcome to a new lecture from the VLA for Autonomous Driving series. Today we'll be discussing reinforcement learning in the context of the self-driving work we've been building throughout this series. The only element missing today will be the language intent — we won't add it, because the lecture is already complex as is. Language intent is just an additional layer we can add to the architecture; fundamentally, understanding how reinforcement learning works in this context is a great thing for you to grasp. So today we'll be looking at **vision-action models** (not necessarily vision-language-action models).

Today's lecture has a tremendous focus on **PPO — Proximal Policy Optimization** — one of the most famous RL algorithms. There are several terminologies, notations, and mathematical ideas to keep track of, so to fully wrap your mind around PPO will take real effort. Two tips: (1) keep something to write on, so you can note down the terms and come back to them; (2) pay close attention, because it's easy to lose track of *why* we introduced a given term. If anything is still unclear, just go through the lecture once or twice more — the longer you spend on PPO, the better the intuition gets. Initially you'll feel like you understand it, then you'll notice knowledge gaps that create doubts, and those gaps close with time.

Toward the end I'll describe an **assignment**: training your own car in an environment called **MetaDrive**, using vision input and a reward function, so that your trained policy can navigate the environment. That training is somewhat complex — it takes a lot of time — so we won't run it live; it'll be an assignment to execute afterward, but we'll set up everything you need to reach that stage.

### Where this sits in the series

So far we've discussed: vision-action models (a CNN-based vision embedding used for action — pure behavior cloning); the Action Chunking Transformer (with a conditional variational autoencoder and language embedding); world models; connecting the hardware (TurboPi) with your coding agent; and running the vision-action model and Action Chunking Transformer in Isaac Lab. One thing remaining for the next lecture is using an agent as a policy — using a coding agent itself as the policy rather than a trained neural network. But before that, introducing reinforcement learning is a very good fit for what we're learning.

## What is PPO (preview)

I'm 100% sure you won't fully understand PPO from this slide — the only reason I'm mentioning it up front is so that later, when I keep referring to PPO and its terms, there's at least a minimum of context. PPO stands for **Proximal Policy Optimization**. It was introduced by a team from **OpenAI in 2017**, so it's been around a while, and although other RL algorithms have appeared since, PPO remains one of the most popular.

We have to understand all three words. You won't understand "proximal" yet — we'll come to it. "Policy" simply means the model that results in your car driving: the car has to make a decision, and the policy is the model that produces that decision. "Optimization" is the optimization of that policy's performance — the policy has to get better and better over time. We'll return to this slide once you properly understand PPO.

## Behavior cloning vs. reinforcement learning

What we've been doing so far is **behavior cloning**: there's an expert who collects training data, and the expert's actions help you train a student to become as close to (as good as) the expert as possible. But if the student is trained purely on expert data, the student can never become *better* than the expert — at best only as good. The reason is the training data: even with augmentation and generalization, we're still just minimizing a loss that's smallest when the student's actions mimic the expert's. So there's an inherent ceiling with expert-based behavior cloning.

This is where reinforcement learning comes in. In RL, the agent (the car) can take actions *different* from what the expert showed. It gets different rewards, and the more it tries, the more it figures out which action sequences produce better and better rewards — potentially even higher rewards than the expert's actions would have earned. That's the power of RL, and it's why, for example, **AlphaGo** could beat **Lee Sedol** (the Go champion at the time), and why AI beat world champions at **Dota**. It doesn't matter what the expert is doing; as long as a proper reward function is set up, actions produce rewards from the environment, and the increase or decrease of that reward guides us toward maximizing it.

Reinforcement learning is the broad term; PPO is one RL framework within it. In our case we'll use PPO for a vision-action setup: a car looks at the scene — only the camera image is visible (something like 100×100 pixels) — and acts.

> **MetaDrive example:** This is the MetaDrive environment you'll train your agent in. In the GIF, the agent is failing — driving straight off and falling down — but this is the kind of environment, with a visual image as input, that you'll be working with.

In previous sessions we created image-action pairs (or image–language-intent–action triplets). Here we'll do something very different. First we'll cover the general RL loop — agent, environment, state, action, reward, return, discount, exploration vs. exploitation — then more specific terms (policy, value, advantage), then the REINFORCE algorithm (a very old one), then actor and critic networks, and finally PPO with its extra pieces: clipping, generalized advantage estimation, entropy, KL divergence. Last, we'll cover the vision-PPO racing task you'll do as homework.

### Behavior cloning vs. RL, side by side

In **behavior cloning** you drive the car and collect data, so you have state-action pairs. The **state** is the RGB image — the current situation. The **action** is what you do for that state: go ahead, left, or right (or, for TurboPi, vₓ, v_y, and ω). Then there's a **supervised loss**: a neural network predicts actions for states, and we match the predicted action to the expert's action. The **policy network** is simply the trained neural network — whenever I say "policy," imagine a neural network that predicts an action for a state; that action changes the state, and so on. It's trained to minimize the difference between predicted and expert action.

In **reinforcement learning** we don't do that. We have an **agent** (the car), an **environment** (real or simulated), an **action** (forward, backward, left, strafing…), and an explicitly defined **reward function**. (Recall from the world-model lecture, we had checkpoint-based rewards laid out along the track.) You can reward forward velocity above some threshold, slowing at corners, staying centered in the road, keeping the car's heading aligned with the track — many reward terms adding up to a net total reward. The **state** is whatever the agent senses (camera, plus any other sensors). And a **policy network learns to maximize this reward**. We never compare the policy's action against an expert action, because there is no expert action in the first place.

> **Q (Rajesh):** Can it avoid obstacles with reward functions? — **A:** Yes; if you set up the reward function so obstacle avoidance is part of it, it can. At the end of the day it's about the reward function. Think of playing a computer game: you learn how to play because certain actions earn reward and others earn punishment. If the reward function is set up well, the agent can do essentially any task.

> **Q (Pratik):** So the action is moving left, right, pushing brakes, etc.? — **A:** Yes. In the MetaDrive example there are two actions: the **steering angle** and the **throttle**. The angle decides direction; the throttle is how much accelerator you apply.

> **Q:** Can we modify the reward function after the model is trained, or must we retrain each time? — **A:** Good question. Technically nothing prevents you from modifying it later. The only question is whether, after an abrupt change, the policy you trained so far is still useful or you start from scratch. You can think of it like fine-tuning: you start with a simple environment and reward, train a policy, then the environment gets more complex (more reward terms), and instead of restarting you continue from where you left off. Note that training takes a lot of time — even the basic training here takes 3–5 hours on CPU, which is why we won't do it live.

## Core terminology: state, action, reward, episode

- **State:** RGB frames. When we say "four stacked frames," we mean you look at a history of four frames to predict the next.
- **Action:** steering angle or throttle.
- **Reward:** how much reward you get for the corresponding action.
- **Agent:** the car. **Environment:** the virtual world the car is in — for us, the MetaDrive simulator (there's a GitHub repo; search "MetaDrive GitHub"). It's a lightweight driving simulator — not as lightweight as the one we used in the first three lectures, but really good, and it lets you collect a lot of parameters.

What the environment returns: the **next state**, the **reward** for the transition you made by taking the action, and a **done** flag. An **episode** runs from a reset (the start) until a **terminal state**. In our case the road is like a sky road: if you go off it, you fall down — that's terminal; reaching the end of the road is also terminal. In general, the terminal state depends on the environment (hitting something, getting stuck, etc.).

## Return (G) and discounting (γ)

Note this term down: **return = G**. Whenever I say "return," I mean G. It's the **discounted return** considering the current action plus future actions. Suppose at time steps t = 0, 1, 2, 3 you're at different states, take actions, and receive rewards (say 0.5, then 74, 9, 6, …). If, sitting at t = 0, you had a way to calculate the rewards your policy would earn over the next several steps, you could compute the total return for the episode:

> **Gₜ = Rₜ + γ·Rₜ₊₁ + γ²·Rₜ₊₂ + … + γⁿ·Rₜ₊ₙ**

This is very similar to **net present value**. $100 given today is worth more than $100 given five years from now; the future amount must be multiplied by a discount factor to get its present value. So G is like the net present value of all future rewards.

γ is always between 0 and 1. If **γ = 0**, none of the future rewards matter — only the present reward Rₜ. If **γ = 1**, you give equal importance to all rewards, present and future. Typically γ is between 0 and 1 — say 0.9 — so the current reward gets full weight and future rewards are discounted. γ¹⁰⁰ at γ = 0.9 is tiny, meaning a reward 100 steps out barely counts, because the future is uncertain and you're not even sure you'll get there. So Gₜ is: if you could simulate the future actions in your mind and collect their rewards, what is the net present value of all those rewards.

> **Q:** A student sets γ = 0.5 to focus on the near term; the agent learns to lurch forward at every step and then immediately falls off the sky bridge. Why? — **A:** With γ = 0.5, γ² = 0.25, and higher powers shrink fast, so future actions have very little influence on the return. The model isn't prioritizing the future — that reward is discounted by γⁿ, which is tiny for small γ — so it primarily optimizes the current action. If you want it to "see" the future, keep γ higher.

## Exploration vs. exploitation; stochastic vs. deterministic policy

First, note that a **policy is a probabilistic model**: for the same input it won't always give the same output. It samples from a probability distribution — like a Gaussian. If you sample 100 times from a Gaussian with mean 0 and standard deviation 1, most samples land near the mean, a few land far away (because the std is non-zero). Similarly, for the exact same state (the exact same image the car sees), the action depends on what we sample from a distribution, and the shape of that distribution is what the policy *is*. So a policy is essentially a probability distribution from which you sample an action for a given state.

- **Exploitation:** you always pick at the peak (the mean) of the distribution — the highest-probability outcome — maximizing the immediate reward.
- **Exploration:** you sample from anywhere in the distribution, possibly picking a suboptimal action now — because maybe it leads to a better future reward or overall return.

**Deterministic vs. stochastic policy:** a deterministic policy gives the same action for the same state every time; these are almost never used. A **stochastic policy** samples from a distribution, so the same input yields different actions at different times. For us, the action is a pair: one number for steering, one for throttle. Each comes from its own Gaussian (defined by a mean and std), so plotting the two gives a 2-D diagonal normal distribution. We never use deterministic policies here.

> **Q:** Which earlier model also gave different outputs for the same input? — **A:** A CNN with fixed weights and biases always gives the same output for the same input. The example where the same input doesn't give the same output is the **conditional variational autoencoder** (in the Action Chunking Transformer), or a **diffusion model** — there we don't predict the output directly; we predict a μ and a σ defining a distribution, then sample from it. Same input → same μ and σ, but different sampled output. Whenever you sample from a probability distribution, that's the case — same here.

## State value (V) and action value (Q)

**State value, V(Sₜ):** if I'm at a state and follow the policy until the terminal state, I get a return G for that rollout. Because the policy is probabilistic, running many rollouts from the *same* state with the *same* policy gives *different* paths and *different* returns (G₁, G₂, G₃, …). Average them (the expectation) and you get V — how good the current state is, **irrespective of the specific action**, assuming you follow the policy. Higher V = a good state to be in; lower V = a bad position.

For intuition: is the **start** of the track a high- or low-value state? Low — because from the start the car must traverse a lot of distance, so under a probabilistic policy the likelihood of failing (falling, big negative reward) is highest. Closer to the finish line (a high-reward terminal state), the car is much more likely to reach it, so the state value is higher. V is simply an estimate of how good the current state is.

> **How V is computed (sketch):** From state Sₜ, run an experiment forward, say 100 steps, to reach Sₜ₊₁₀₀; the 100 rewards give a return G_t¹. Repeat to get G_t², G_t³, … These differ every time, because each reward comes from the environment in response to an action sampled from the (non-deterministic) policy π. Averaging G_t¹…G_t^N gives the expectation of the return for that state — that's V. (Remember: G = return; R = reward — don't confuse them.)

**Action value, Q(Sₜ, A):** unlike V, Q is a function of **both** state *and* a chosen action. The idea: from the current state, take *one particular action A first*, then follow the policy to the terminal state, and compute the resulting return. So if from state S₁ you reach S₂ via action A, B, or C and then follow the policy, you get returns G_t^A, G_t^B, G_t^C — these are Q(S₁,A), Q(S₁,B), Q(S₁,C).

The difference from V: for V you **never** pick an explicit action — you always follow the policy. For Q you pick a particular action first, *then* follow the policy. V is **state value** (action-agnostic); Q is **action value** (action-specific).

## Advantage (A)

**Advantage = Q − V.** Conceptually: how much better is this particular action than, on average, just following the policy. If Q(S,A) − V(S) is **positive**, taking that specific action in the current state is better than blindly following the policy; if **negative**, it's worse.

> **Advantage graph intuition:** Say at state S_B you have three actions — left, straight, right. The dotted line is V(S_B), say 61.6 (the state value, action-agnostic). Each bar is the Q for one action. If going right gives a Q of 48.2, the advantage is negative — maybe turning right caused a collision or a fall (very bad reward). The advantage is the gap between each bar and the dotted line. We'll come back to this term.

To recap three terms: **V** = state value, **Q** = action value, **A** = advantage (how much better a particular action is than the policy-mediated average action).

## The objective J(θ) and the REINFORCE gradient

We want to maximize the return G (higher G ⇒ better current and future actions). We have a neural network — the **policy** — with weights and biases **θ**. We write the policy as **π_θ** (π = policy; θ = its parameters); think of it as "neural network as a function of its weights and biases." The network predicts an action for a state; the action leads to the next state and a reward; repeat to get all future rewards and hence a return. The goal: **how do I modify the policy's weights through training so that the overall return is maximized?**

We call the return objective **J(θ)** — think of it like G, but written to show it depends on the network's parameters. We want the global maximum of J(θ). So we perform **gradient ascent**: move in the direction of the positive gradient of J, climbing the return landscape. The REINFORCE gradient comes out (approximately) as:

> **∇J(θ) = Σₜ Aₜ · ∇ log π_θ(aₜ | sₜ)**

where A is the advantage and the sum is over time steps. There's a relatively simple derivation from J(θ) to this expression (I'll share a PDF), but it would take ~15 minutes, so I'll skip it. The key reading: the gradient of the return is a function of the gradient of the log-policy *and* the advantage. If the advantage is positive, it's good — move along the positive gradient (favor that action more). If negative, it's not good for that step (favor that action less).

> **Why "approximate"?** In the original equation it's Gₜ (the return itself) rather than the advantage A; using A is a good approximation. This algorithm/framework is called **REINFORCE** — a very old one (proposed in **1992**), not used directly now, but instructive.

> **Policy update visualization:** A policy is a probability distribution parameterized by θ. When θ changes through training, the distribution you sample actions from changes — the mean shifts, the std shrinks, a skew appears — yielding a better policy distribution. So "policy update" just means: after a training step, the distribution you sample your action from is slightly different.

So REINFORCE says: for each time step, compute the advantage and the gradient of log π. Positive advantage contributes positively to ∇J (promoting that action); negative advantage contributes negatively (discouraging it). This naturally pushes the policy toward actions that maximize Gₜ.

> **Quick check:** A policy has V(S) = 50, Q(S,A₁) = 65, Q(S,A₂) = 45. What does the policy gradient do? — Advantage of A₁ = +15 (positive ⇒ favored), advantage of A₂ = −5 (negative ⇒ less favored). The network shifts weights to pick A₁ more and A₂ less.

## Actor–critic

**Actor–critic** is simply this: an **actor head** that predicts the **action**, and a **critic head** that predicts the **value V** for a state. Both share the network up to a point. Note the terms: actor head → action; critic head → V(s). We'll see how this plugs into the final architecture.

## From REINFORCE to PPO: the noisy-update problem

PPO contains the actor–critic framework plus everything else we discussed, with a few additions. Here's the problem with naive REINFORCE updates. Suppose one particular action at the current state produces a huge reward. That action gets a big advantage A, and since |∇J| scales with |A|, the gradient is big — so the network's parameters change a lot. That means your **policy changes drastically because of one good (or one bad) action**. The update is very noisy and overly dependent on the reward of a single action. That noisiness is the core problem with naive REINFORCE, and PPO is designed to fix it — to make updates smoother.

## PPO ingredients: π_old vs. π_new, importance ratio, clipping, KL

We track two versions of the policy: **π_old** (before updating θ) and **π_new** = π_θ (after). At every gradient step, we look at how different π_new is from π_old. How do you compare two distributions? **KL divergence.** If the KL divergence is high, the two policies are very different — which we *don't* want, because the policy shouldn't drift too much in one step. So we introduce a **KL-divergence cutoff**: if the KL divergence exceeds, say, **0.03**, we skip that update and only update when the policy changes slightly.

> **Q:** Isn't a hardcoded 0.03 a bit arbitrary/bad? — **A:** For sure it's arbitrary, and I don't love it, but it's what people found works well. Other algorithms/papers have found better numbers. You'll see another hardcoded number shortly.

There's also the **importance ratio**, rₜ(θ) = π_new(aₜ|sₜ) / π_old(aₜ|sₜ) — the ratio of the new policy to the old policy for a given state-action. It measures, for a given state, how much more (or less) likely the new policy is to favor action aₜ compared with the old one.

Two steps back: the REINFORCE gradient (∇J = Σ A·∇log π) was too dependent on individual action quality, causing very noisy updates. PPO fixes this by tracking how much the policy changes and **clipping** the importance ratio to a range between **0.8 and 1.2** (the other arbitrary number, like the 0.03). If the ratio exceeds 1.2 (≈20% more likely to pick an action under the new policy) or drops below 0.8 (≈20% less likely), that change is not taken into account. In other words, we only allow a limited range of modifications to the policy.

## The PPO clip (surrogate) objective

PPO optimizes the **clipped surrogate objective**:

> **L = E[ min( rₜ·Aₜ , clip(rₜ, 0.8, 1.2)·Aₜ ) ]**

where E is expectation, rₜ is the importance ratio, and Aₜ is the advantage. (Don't try to memorize this now.) The whole point is to **prevent the policy from being updated by too much in a single step** — so that no single action (or few actions) dominates how the policy is modified. It's called a **surrogate** objective because it replaces the previous objective: REINFORCE maximized J(θ) directly; PPO defines a slightly different objective that also accounts for the importance ratio.

## Generalized Advantage Estimation (GAE)

Instead of computing the advantage simply as A = Q − V, PPO uses **Generalized Advantage Estimation (GAE)**. First, the **TD (temporal-difference) residual**:

> **δₜ = Rₜ + γ·V(Sₜ₊₁) − V(Sₜ)**

This resembles Q − V: Rₜ is the specific reward for the action taken; γ·V(Sₜ₊₁) is the (discounted) value of the next state — an alternative to the Q term — and we subtract V(Sₜ). GAE then sums these residuals over future steps with an extra discounting factor **λ** that smooths the estimate:

> **Aₜ = Σₖ (γλ)ᵏ · δₜ₊ₖ**

We won't go deeper — it's a lot — but the idea is: instead of computing the advantage naively as Q − V, GAE incorporates several discounted future steps, producing a smoother advantage estimate.

## Entropy bonus

The action predicted by the policy isn't deterministic — it's sampled from a distribution with a mean μ and std σ. This is exactly where **exploration vs. exploitation** lives. If σ is very low, samples cluster at the mean (no exploration — pure exploitation); we want σ somewhat higher so the policy occasionally tries suboptimal actions and explores.

How do we encourage the network to predict a not-too-small σ? We add an **entropy** term: higher σ ⇒ higher entropy. Entropy enters the loss with a **negative multiplier**, so a higher-variance (higher-entropy) distribution *lowers* the loss — incentivizing the network not to collapse σ toward zero. (Too high is also bad, but the bonus keeps σ from getting too low.) This is the last new term: the **entropy bonus**.

## Putting the loss together (recap of all terms)

Laying out everything: state/environment/action/reward → behavior cloning vs. RL → deterministic vs. stochastic policy → state value V vs. action value Q → advantage A = Q − V → the REINFORCE objective J(θ) (maximize the return), with gradient Σ A·∇log π. Because a single high-advantage action could change the policy too much, **PPO ("proximal")** keeps the new policy in *proximity* to the old one — hence "proximal." ("Policy" = the neural network; "optimization" = finding the maximum, but smoothly, never allowing huge one-step changes.) Plus **GAE** (a better advantage estimate using discounted future steps) and **entropy** (higher σ → higher entropy → lower loss → more exploration). And **KL divergence** flags when π_old and π_new differ too much (if KL is too high, skip the update).

The **final PPO loss** has three parts:
1. **Clipped surrogate (PPO/actor) loss** — keeps the π_new/π_old ratio within 0.8–1.2.
2. **Critic (value) loss** — predicted V vs. the ground-truth value of the state.
3. **Entropy bonus** — entropy × a negative coefficient (it's a *bonus*, not a penalty: high entropy lowers the loss, gently rewarding the model for predicting a not-too-low σ).
Plus the constraints: KL divergence < 0.03, and the importance ratio within its threshold.

## The neural network architecture

> **Q:** Why does naive REINFORCE lead to large policy swings? — **A:** Exactly the reason we don't prefer it. (Demo: the MetaDrive **sky bridge** — called that because the physics make the car fall if it leaves the bridge. A bad policy fails to find the turn and falls off; falling is terminal. The car must follow the track, take the turn, and reach the terminal state.)

The environment is the MetaDrive sky bridge, and the car's view is roughly a 94×94-pixel image — a pure vision model, with no other input. (We could use Isaac Lab, but MetaDrive is free for everyone and easy to try, and it exposes lots of parameters; this could be done in Isaac Lab too.)

**Reward function (intuition):** higher for driving fast; lower for driving fast in a corner; higher for staying centered; higher for progress along the road; high bonus for reaching the end. In the homework you'll define your own custom reward, aiming to get the car from start to finish in minimum time while staying on the road.

**Flow:** the observation (input image) → an **agent network** (a CNN) → an **action** → fed back into the environment → next state/observation and a reward → back into the network → next action, and so on. At **inference**, you simply have the trained policy: collect observation, act, repeat.

> **Q:** Would it work with more than one car? — **A:** In principle yes, but to race/win you'd need to train with other cars present, add a reward term penalizing collisions, *and* make sure the state actually shows the other cars (otherwise the policy can't react). The current setup has one car for simplicity.

**Training** is where the network is optimized: we compute Q, V, A, the loss, KL divergence, and entropy. Let's walk through the pieces.

### CNN encoder

The input is the last **four frames**, each an RGB image (so a stack of four images, each ~92×92). A sequence of **convolution layers** shrinks the spatial dimensions and grows the channel dimension; after a final projection you get a single **512-dimensional vector** representing the current frame plus the past three frames.

### Actor head(s)

The agent (policy) network must predict **two things**: an **action** and a **value**. For the action, how many numbers? In MetaDrive there are **two actions**: **steering** and **throttle**. (For the earlier Three.js simulator there were four — forward/backward/left/right; for TurboPi, vₓ, v_y, ω.) But the model does **not** predict the action directly. It predicts, for each of the two actions, a **μ** and a **σ**:

- μ and σ for **steering** → a Gaussian for the steering value
- μ and σ for **throttle** → a Gaussian for the throttle value

So there are two sub-heads — one predicts the means (steer-mean, throttle-mean), one predicts the stds (steer-σ, throttle-σ). You **sample** from each Gaussian to get the raw steer and throttle. Then a **tanh** squashes the raw values into the **[−1, +1]** range (normalized outputs train better). This squashed value is the action fed into the environment — recursively, until a terminal state (one episode). (At inference you can sample at the mean for pure exploitation, or away from it for exploration — exploration comes from σ.)

> **Q:** Why scale *after* tanh, not right after sampling from the Gaussian? — **A:** Network parameters train better when confined to a sensible range; μ/σ outputs could in theory be huge (−∞ to ∞), so we normalize with tanh first, then apply an environment-specific **scaling factor** (e.g., MetaDrive may expect values in a different range) before applying the action.

> **Q:** Why a Gaussian? — **A:** For sampling a single parameter, a Gaussian is the simplest approximation (as in diffusion models, where we approximate the data distribution as Gaussian with some μ and σ). The true distribution of actions might differ slightly, but the Gaussian is a good approximation.

### Critic head

The critic head takes the **same** 512-dimensional input (in parallel with the actor) and predicts **V**, the state value. V is used in two places: (1) a **loss** comparing predicted V vs. the actual value of the state, and (2) inside the **GAE** formula.

## The rollout buffer (where ground-truth V/Q come from)

> **Q:** The critic predicts V, but where does the *ground-truth* V come from? To compute Gₜ I need Rₜ, Rₜ₊₁, Rₜ₊₂, … — but I'm still at time t, so how do I know future rewards and the true state value to compare against?

The answer: a **rollout buffer**. A large number of episodes are played out and stored. One buffer corresponds to one policy π: it rolls out a fixed number of time steps, recording, at each step, the state, the action (from the same policy), and the reward. With that full table available, you can compute **Vₜ** (the expected return for each state) and **Qₜ** (for a given action) just like a cumulative sum in a spreadsheet. So the rollout buffer is a **pre-computed table** for a given policy, and it gives the ground-truth Vₜ that the critic's prediction is compared against (one of the loss terms). When the policy is updated to π_new, the table is no longer valid and must be **regenerated**.

GAE is likewise computed from the rollout-buffer data (anything from the future is read from the buffer), and it too depends on the current π — when π changes, the buffer changes.

## The final architecture, end to end

The observation is four stacked RGB frames (X×Y resolution). Three convolution layers turn it into a 512-D vector, which feeds **both** the actor head and the critic head in parallel:
- **Actor head** → 2-D output (steer, throttle), via two sub-heads (μ and σ). Build the Gaussian π_θ(s) from μ and σ; compute **entropy** from σ; **sample**, then **squash** with tanh to [−1, 1]; then apply the **environment scaling** before stepping the environment, which returns a new reward and state.
- **Critic head** → 512→1 (it predicts only V).

The three losses (clipped surrogate, value/critic, entropy) are computed using the rollout buffer; the buffer is built by rolling out a future horizon (e.g., 512 steps) and is **regenerated whenever π changes**. Also enforced: KL divergence < 0.03 and the importance ratio within threshold.

> **Where do the parameters live?** Primarily in the linear projection 512→2 (for μ and σ: 512×2 + 2 weights/biases) and the critic's 512→1 (512 weights — it predicts a single value). And if the CNN encoder is **not** pre-trained, its parameters are updated too. So the "policy" is primarily the actor-head parameters, but in practice the CNN encoder + actor head + critic head are all updated as the policy improves.

## The assignment

There's a GitHub repo and a website (built with the help of a teammate, "Naman") that acts as an arena where you train a car with PPO. **You define your own reward**, and you must train a policy until it can actually drive. To reach a baseline you'll need at least **3–5 hours of CPU training**, so I recommend **RunPod** if you have access — it's much faster and lets you try more. The repo (linked from the site) has all the instructions; you can run the commands via **Claude Code** or **Codex**, or directly in the terminal. It will clone the repo and run training.

The thing you customize is the **reward** (in `reward_config.py`) — if you don't change it, everyone's performance ends up roughly the same, so find a smarter reward function. The default reward combines: **progress** (moving forward), **centering**, **heading** (facing the road's curvature), **speed**, **control** (penalizing random idling/braking), and a **bonus for reaching the terminal state**. This reward is where the environment plugs into the network: when you apply an action, the environment produces the reward, which feeds the computation of V, Q, and the surrogate loss. At the end of the day, **how you define the reward** is what differentiates your performance from someone else's — so spend most of your time there. (If you want, you can even swap PPO for another RL algorithm, but for a fair apples-to-apples comparison, stick with PPO. Submission instructions are in the repo.)

> **Q:** How do we define rewards manually — is it scalable? — **A:** In complex environments, defining reward manually is very difficult. Ideally a neural network predicts a reward/value (recall the world-models lecture, where a network both "dreamed" the trajectory *and* predicted the reward). But a lot of manual work is unavoidable, because only humans know what we want — we know the car must drive a certain way, or the game must be played a certain way, or we're not happy. So a basic reward function has a large human-defined component, with the rest predicted by a neural network. In the real world it gets complex because there are effectively infinite reward/penalty combinations.

## Closing and next lecture

This is the second-to-last lecture of the VLA for Autonomous Driving series. Next time I'll show something fairly new and interesting: **using an agent as a policy** — a coding agent itself as the policy, instead of a large trained model (a world model, a VLA model, or π0). The idea: can an agent directly produce motor commands in one shot, just by looking at the camera and observing the environment? It's slow — latency is much higher than a world model or VLA — but agents are already very good at digital tasks, so in principle they could also be good at physical tasks: observe the environment, calibrate geometry and physics, then predict what to do. The field is very nascent, but it's a lot of fun, so we'll explore it in the next (final) lecture.

I know we covered a couple of dozen terms today — it's very mathematically involved, which is exactly why RL books are hard to digest: so many ideas at once. So please rewatch, take good notes, read from different sources, ask ChatGPT for definitions, and work out the interconnections. It's not hard to understand — it just takes time and deep-work sessions. Continue the discussion on Discord, and if you make good notes in any format (a compiled PDF, anything), please share them so others can benefit. Thank you so much — we'll meet next week and wrap up the series. See you soon.
