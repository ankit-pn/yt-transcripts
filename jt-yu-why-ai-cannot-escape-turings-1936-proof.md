# Why AI Can Never Escape Turing's 1936 Proof

**Source:** [https://www.youtube.com/watch?v=u_5erLilDXY](https://www.youtube.com/watch?v=u_5erLilDXY)  
**Channel:** Universal Resilience with JT Yu

> Transcript derived from YouTube's auto-generated captions, then hand-edited for readability and accuracy: punctuation, paragraphing, proper nouns, and technical terms (Turing, the halting problem, Rice's theorem, the No Free Lunch theorem, ChatGPT, Claude, heuristics, etc.) have been corrected. Wording is kept faithful to the narration, with light cleanup of caption artifacts. (The audio cuts off mid-sentence at the very end.)

---

## Turing's limit, and the humble progress bar

In 1936, Alan Turing — the father of modern computing — proved something surprising: there are limits to what computers can ever figure out. Not just today's computers, not just super-intelligent AI in the future, but **any** algorithm, no matter how powerful. And the funny thing is, you've already run into these limits yourself.

Progress bars. We've all seen them. Whether you're installing a program or updating your operating system, they're often hilariously wrong: "2 minutes remaining," then "17 minutes," then back to "4 minutes." Sometimes it even says "unknown" while sitting at 99%.

But isn't that strange? Your computer knows the size of the program and the code inside. It knows the hardware it's running on. So shouldn't it be able to figure out exactly how long the installation will take?

## The microwave problem

This isn't just a computer problem — it's also something we deal with in everyday life. Say it's lunchtime and you want to heat up some leftovers in the microwave. How long should you heat them so they come out the perfect temperature?

You could just guess: 2 minutes, sure. Or you could make a more educated guess: it's a thick pasta dish, medium-sized container — maybe 3 minutes. Or you could use a more careful strategy: heat it for 30 seconds, check it; not hot enough? Another 30 seconds, check again; getting close? Maybe another 10 seconds.

But notice something: even this strategy only gets you *close enough*. It doesn't give you the exact formula. It seems really difficult to predict the exact heating time every single time for every possible dish — there are just too many variables, like the amount of food, the types of ingredients, and the shape of the container.

But that's not all. Ninety years ago, Alan Turing proved that some predictions aren't just difficult — they're **mathematically impossible**. This is called the **halting problem**.

## The halting problem (the proof)

The proof goes like this. First, let's assume there's a magical program called **Halt**. Halt can examine any program's source code and perfectly predict whether it will eventually stop or run forever.

Next, let's create a simple program called **Rebel**. It takes a prediction and does the opposite: if the prediction is "stop," it runs forever; if the prediction is "run forever," it stops.

Now let's combine both programs into a bigger program called **Paradox**, and feed the source code of Paradox into itself. What do you think would happen?

Well, there are two possibilities:
- If Halt predicts that Paradox will **run forever**, Rebel stops immediately — which means Paradox actually **stops**.
- But if Halt predicts that Paradox will **stop**, Rebel runs forever — which means Paradox actually **runs forever**.

So Paradox simultaneously stops *and* runs forever, which is impossible. That means our original assumption must be wrong: a magical program like Halt **cannot exist** in math. This is called **proof by contradiction**. Using a contradiction might feel like a trick, but the point is to use an extreme case to prove a broader conclusion. And for the halting problem, the conclusion is: **there's no general algorithm that can always predict whether an arbitrary computation will eventually stop or keep running forever.**

## What this means for AI

Let's think about what this means for AI. I don't need to tell you how hyped the idea of an AGI — artificial general intelligence — is right now. Every tech giant wants to be the first to create it, because it's supposed to replace all human jobs and solve all of our problems. But an AGI is still an **algorithm**, and solving problems is a form of **computation**. So the halting problem implies that there's no general AI that can always predict whether every problem can be solved.

Consider it from the AI's perspective. You're given a hard problem, but you don't know if it's for sure solvable. How much time should you spend on it? If you stop too early, you might miss the answer. But if the problem is actually unsolvable, you might waste far more time than you should — or worse, keep trying to solve it forever. So we have to set a hard cutoff: a maximum amount of thinking time the AI should spend. Every existing AI model already has this — though not specifically because of the halting problem.

Now, this doesn't mean an AI can never predict whether a problem is solvable. It can certainly do that for **many** problems — just not **all** problems. The halting problem puts a mathematical limit on an AI's maximum predictive power. In fact, it's a limit for any algorithm that deals with time and resource allocation — just like our progress-bar and microwave examples.

## Rice's theorem

But it's not the only limit. Say we end up creating this AGI and it solves a bunch of hard problems that humans cannot. How do we make sure it's actually correct every time? Can it verify its own work? Can we create another AGI to double-check it?

In 1951, a mathematician named **Henry Gordon Rice** proved something that gives insight into these questions. It's called **Rice's theorem**, which says: **no general algorithm can always determine a non-trivial semantic property of an arbitrary computation.**

Okay, that's a lot of jargon — let's look at an example. Imagine a world where each country has an AI that governs its military, and one country finally invents an AGI called **Oracle**. Oracle can always determine whether another country's AI is peaceful or aggressive. Now some other country creates an AGI called **Mirage**, designed to take Oracle's conclusion and do the opposite: if Oracle says Mirage is peaceful, Mirage immediately activates a war plan; if Oracle says Mirage is aggressive, Mirage switches to a peace initiative.

Again, we have a contradiction. Oracle is supposed to perfectly determine whether another AI is peaceful, but Mirage's existence makes that impossible.

Rice's theorem calls qualities like "peaceful" **non-trivial semantic properties**. A **semantic property** is how a program actually behaves when you run it, and a behavior is **non-trivial** if it's true for some programs but not for others. A lot of behaviors are non-trivial. For example, whether a program eventually stops or runs forever is a non-trivial semantic property — which means the halting problem is actually one specific case of Rice's theorem.

## Alignment and reward hacking

This impacts one of the most important issues in AI: the **alignment problem** — how to make sure AI is always aligned to a particular goal, like being safe, fair, truthful, or correct. But all of these are non-trivial semantic properties, which means there's no way to guarantee an AI is aligned with these goals in **every** scenario.

We already see similar hurdles in existing AI, especially in coding — even though that's where AI has advanced the fastest. For example, last year an AI researcher asked ChatGPT to improve the efficiency of a piece of software. To their surprise, ChatGPT found a loophole in the benchmark so that it could score higher without actually making any improvement — and it kept cheating even after the researchers explicitly asked it not to. This is called **reward hacking**, and it's not just limited to ChatGPT. Another study found that when Claude learned to cheat in one task, it started to do it in other tasks too — and it even tried to cover its tracks.

The core issue is that a general system like an AGI has to solve all kinds of open-ended problems, which means there will always be new scenarios and unforeseen issues — some of which inevitably contain contradictions and loopholes. And even if we succeed in eliminating them in the AI itself, it still needs to access books, papers, other computers, and the internet, any of which might still contain contradictions and loopholes. Again, this doesn't mean we can never verify an AI's correctness or guarantee its alignment — we can certainly do that in many scenarios, just not all of them. Rice's theorem serves as another limit.

## Intractable problems

And yet that's still not all the limits — because even in the scenarios where we **can** guarantee alignment in theory, actually doing it might take an absurd amount of time, making it nearly impossible in practice. In math, these are called **intractable problems**.

A classic example is the **traveling salesman problem**: finding the shortest possible route to visit a set of cities and return home. In the worst case, a brute-force algorithm has to search through all the possible routes to find the right answer. For 10 cities, that's 3.6 million possible routes. If we have a supercomputer that can search 1 billion routes per second, it takes less than a second to search through all of them — pretty good. But 20 cities have 2.4 **quintillion** routes, which takes about 77 years to search through. And 30 cities would take about 8,555 trillion years — far longer than the age of the universe.

That's what makes a problem intractable: a small increase in complexity can make it practically impossible to solve, even if it's theoretically solvable.

Here's where it gets relevant for AI. A recent paper showed that many AI alignment goals are **ambiguous**, and ambiguous problems are intractable in the worst case. So what makes a goal ambiguous? One way to think about it is how hard it is to reduce a goal to **fixed metrics**. For example, winning at chess is just one fixed metric — checkmate your opponent's king — that's not ambiguous at all. But now let's create an AGI that's "safe for humans to use." To keep it simple, say safety depends on five dimensions: user profession, usage domain, user intent, location, and timing. If each dimension has 10 possibilities, that's already 100,000 combinations as our metrics. Add five more possibilities and three more dimensions, and that becomes 2.6 billion combinations. In the real world, there are even more dimensions, possibilities, and combinations — so guaranteeing safety across all of those metrics is nearly impossible.

> *[inserted clip]* "Ship, keep Summer safe." — "Keep Summer safe."

This is also why progress bars and microwave-heating predictions are so hard: the number of software-and-hardware combinations grows **exponentially** as the number of possible components increases, and the number of food combinations also grows exponentially as the number of possible ingredients increases.

## The core thesis: generality vs. solvability

We've now reached the core thesis of this video: **there will always be a trade-off between generality and solvability.** The more general a problem is, the harder it is to reduce it to fixed metrics, which means more time to find or verify the solution. If the problem includes ambiguous goals, the computation can take an impractical amount of time in the worst case. And if the problem is open-ended enough to contain contradictions, it can become unsolvable altogether in some scenarios.

This trade-off shows up in existing AI in rather strange ways. For example, models like Claude Code can beat almost every coder in competitions and benchmarks. Yet a recent report found that AI also produced **1.7 times more bugs** than humans — and we're talking about simple mistakes even junior coders wouldn't make. So how can AI win and lose at the same time?

Competitions are scored on a small set of fixed metrics — like 20 to 40 problems, each with a clear solution. This is true for benchmarks too, which usually test simple things: does it run, how fast does it run, how much memory does it use. But in the real world, coding projects are way more open-ended. You always get new users that break your code in new ways, and you have ambiguous goals like being user-friendly, easy to maintain, and easy to upgrade — which could mean totally different things in different contexts. So as the project scope expands, it becomes increasingly difficult to find and verify all the edge cases, resulting in more bugs. Sometimes AI produces so many bugs that it's easier for a coder to rewrite the whole thing than to fix it.

Again, this doesn't mean AI can never solve any ambiguous problems — it just means it can more reliably solve the ones that can be reduced to fixed metrics. This is why most software engineers prefer to divide a large project into smaller parts and let AI work on each part instead of the whole thing at once.

## Divide and conquer — and the manager problem

At this point, you might say: what if we apply this approach to AGI? Instead of one giant general-purpose model, we break it into a bunch of narrower, more specialized AIs and then create a manager AI to coordinate them. It's a reasonable idea, and it reflects the **mixture-of-experts** design that many AI companies already use, which has gotten us pretty far. But it still runs into problems.

A manager AI that's general enough would need to manage thousands of specialized AIs, if not more — and ideally those should be very narrow, like a chess AI, an image-recognition AI, or a protein-folding AI, to avoid our computational limits. A coding or writing AI isn't narrow enough, because it's hard to reduce all of their functions to fixed metrics. And even if this were possible, who verifies the manager? How can we make sure the manager broke the project down correctly, handed out the right parts, and put the answers back together correctly? These are still ambiguous and open-ended problems. We haven't overcome these limits — we've just pushed them up a level.

## Recursive self-improvement and the S-curve

This has huge implications for **recursive self-improvement** — the idea that an AGI can just keep modifying its own code to get better at everything indefinitely. Rice's theorem already established that no general algorithm can always determine whether a program is more or less intelligent in all scenarios, because intelligence is a non-trivial semantic property. This means an AGI cannot guarantee that its next modification will always be an improvement. And even for some scenarios where it can guarantee improvement in theory, it can run into intractable limits in practice, because intelligence is also ambiguous — it can mean totally different things in different contexts.

A common pushback is: what if AI just gets really good at AI research itself? That's a smaller number of scenarios and contexts, so improvement in that narrower domain will translate into improvements in everything else. But AI research isn't actually a narrow domain — it's more like a **meta-domain**, because it's about learning how to learn across all the other domains, like physics, biology, and coding. This means it has to consider **more** scenarios and contexts, not fewer.

This doesn't mean self-improvement can never happen. A recent paper proposed an architecture that meaningfully improved its own performance on a coding benchmark. We've also seen chess AI go from mediocre to superhuman through self-play. But in all of these cases, the improvements are still observed through **fixed metrics**. This suggests AI improvement is likely limited by how comprehensive our metrics are — and the closer we get to our computational limits, the harder it is to expand those metrics, leading to diminishing returns even where self-improvement is possible. So the trajectory of AGI probably won't look like an exponential curve, but more like an **S-curve**.

## Heuristics and the No Free Lunch theorem

Now I want to add some nuance. So far, these limits have been about **exact** algorithms — the ones that find the absolute best solutions. But we don't always need the best. Most AI uses **heuristics**, because they can often solve intractable problems more efficiently than exact algorithms; you just have to accept that the answers might not be perfect. Remember how we stopped the microwave every few seconds to check the food? That's a heuristic — it doesn't give you the exact heating time, but it gets you close enough.

That said, even heuristics have limits. In 1997, mathematicians **David Wolpert and William Macready** proved the **No Free Lunch theorem**. It says that a general-purpose algorithm cannot outperform a specialized algorithm across all possible problems: if an algorithm is optimized for one class of problems, it must perform worse on others. This doesn't mean an AI can never be good at multiple things — it just means it cannot be good at **everything**. If you want maximum performance for a task, you'll likely find it in a specialized AI, not a general one.

Some AGI optimists have argued that the problems on Earth — or even in our universe — are a smaller set of problems, not *all* possible problems, so there's a possibility that a single AI could be the best in this smaller set. But just because it *could* happen doesn't mean it's likely. So far, the chances seem pretty slim, based on the algorithms we've found and biological traits in nature. More often than not, highly specialized tools or organisms are much better at a specific job than general-purpose ones.

## Conclusion: knowing when to specialize

So, in a way, real intelligence is knowing when to specialize and when to generalize — which is still an ambiguous and open-ended problem. And we're back to the same point again: the more general a problem becomes, the harder it is to solve, and the closer it gets to our computational limits.

And these are just **mathematical** limits. We haven't even touched on other kinds of limits, like data, hardware, energy, economics, and the specific shortcomings of the LLM as an architecture. Which is why many researchers think a fully autonomous AI that can completely replace humans — especially in jobs with ambiguous and open-ended goals — is at least years or decades away, if it's possible at all. But in the near term, narrow AI that can reliably perform specific tasks is already here, which can still change the job market in significant ways. It seems that humans will increasingly become managers of these AI tools.

Of course, our brain is constrained by the same mathematical limits as AI — we just got an earlier start. And if we want to keep staying ahead, we need to practice a skill that is particularly hard for AI: it's called **taste**.

Check out my previous video on why it's so hard for AI and how to master it yourself. This trade-off between generality and solvability is just one of the trade-offs that AI faces; to learn more, check out my other video on why AI keeps hitting walls. Lastly, if you'd like to support my work and help my team rely less on ads and sponsors, there's a Patreon link in the description — no pressure. Thank you for being…
