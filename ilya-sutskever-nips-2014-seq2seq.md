# Ilya Sutskever @ NIPS 2014 — Sequence to Sequence Learning with Neural Networks

**Source:** [https://www.youtube.com/watch?v=9l9xGoyDhMM](https://www.youtube.com/watch?v=9l9xGoyDhMM)  
**Speaker:** Ilya Sutskever (with Oriol Vinyals and Quoc Le, Google)  
**Event:** NIPS 2014  
**Channel:** AI Weekly

> Transcript derived from YouTube's auto-generated captions, then hand-edited for readability and accuracy: punctuation, paragraphing, proper nouns, and technical terms have been corrected. Timestamps mark roughly where each paragraph begins. Wording is kept faithful to the talk, with light cleanup of spoken filler.

---

## Introduction

**[00:00]** *Moderator:* So our next speaker is Ilya Sutskever. He's going to present a paper titled "Sequence to Sequence Learning with Neural Networks," and his co-authors are Oriol Vinyals and Quoc Le from Google.

*Ilya Sutskever:* Thank you very much for your patience. So, my name is Ilya Sutskever, and this is work that I've done with my excellent co-authors Oriol Vinyals and Quoc Le. I suppose I could start with a joke about how many PhDs it takes to set up a projector.

So, the foundation of this work is the deep neural network, and deep neural networks have two properties that make them very desirable and attractive.

## Why deep neural networks

**[00:46]** The first property of deep neural networks is that they are **powerful** — which means they can perform an astonishingly wide range of computations, and there's a very good chance that they can compute the function we care about. The second property is that they're **trainable**. The fact that they're both powerful and trainable means that when we have a difficult problem, we can find the best possible neural network that solves it, and we'll get good results.

One important point is that it is absolutely essential to use a powerful model. If the model we use is not powerful, then it really doesn't matter how much data we have or how good our learning algorithm is.

**[01:31]** For example, if you have a linear model, a linear SVM, or a small neural network, it's basically impossible to get good results on really difficult problems. Small convolutional networks are the same — they have a small number of neurons, so they cannot compute the right kind of functions that we want.

The good thing about deep neural networks is that we can be reasonably confident there exists a good setting of the parameters that achieves high performance, especially on perception tasks. This is something I call the **deep learning hypothesis**, although it's been known to connectionists since the 1970s. The argument goes like this: human beings can solve a lot of very interesting problems in a fraction of a second — especially perception, vision, and speech — and yet our neurons are very slow.

**[02:17]** Which means it is possible to solve a very complicated perception problem using on the order of ten massively parallel steps. So we can conclude that anything a human can do in a fraction of a second, a big ten-layer neural network can do as well. This gives us confidence that if you have a hard problem, you just need to take a big deep neural network — and if you have a lot of data, you'll solve it. And very conveniently, it is possible to train these models with stochastic gradient descent.

## The limitation: sequences

Now, despite their great power, deep neural networks cannot learn to map sequences to sequences — they can only map vectors to vectors. For example, in visual recognition it's pretty clear: you've got an image as a vector and a set of categories on the output. Or in speech recognition, neural networks are

**[03:03]** typically used to map a little chunk of speech into a little category that tells us what kind of phoneme it is. Mapping sequences to sequences is not something neural networks can do — or at least could not do at the time we were doing this work. And there are many interesting applications for it: machine translation, speech recognition, the recent image caption generation, question answering, summarization. All of these problems can be expressed as: you take a sequence as input, and you produce a sequence as output.

So our goal in this work is to solve the general sequence-to-sequence problem. We take an extremely simple approach, which I'll describe soon, and we achieve pretty good results on machine

**[03:49]** translation. In this paper, our performance is close to the winning entry of the WMT'14 English-to-French task, which is a real, large-scale machine translation problem — and we surpass that result in later work. But the real take-home message is not that we can do machine translation; it's that we can do *any* sequence-to-sequence problem. Machine translation is already pretty good — if you use Google Translate you get good translations — and yet our system could match it, and it doesn't really assume anything about translation. If you have any other sequence-to-sequence problem you wish to solve, you can just plug it in without any thinking and get reasonable, even good, results.

So what's the problem with deep neural networks? How is it that they cannot solve the sequence-to-sequence problem? The

**[04:35]** answer lies in the fact that different neurons have different connections. For example, you can see that this neuron has its own set of connections, and the neuron next to it has a different set of connections. What this means is that the neural network doesn't have a good way of generalizing across temporal patterns: if you see a pattern at one time step and then see it at a different time step, it looks totally different.

So you might say: okay, what about the recurrent neural network? The recurrent neural network deals with sequences — can't you solve your sequence-to-sequence problem with it? Here's a recurrent neural network in this figure. An RNN is basically a regular

**[05:20]** conventional neural network where you use the same connections at each time step. So you can take a sequence of inputs, map it into a sequence of hidden units, and get a sequence of outputs. In a sense, you also map a sequence to a sequence with a recurrent neural network — and yet I claim it's not adequate for our needs. Why not? What's wrong with them?

The main problem with recurrent neural networks is that there's a **one-to-one correspondence between the inputs and the outputs**. As you can see, for each input there is an output. You may have a sequence of inputs and produce a sequence of outputs, but both sequences are going to be the same length and have the same alignment. Another problem with the

## Training RNNs: exploding and vanishing gradients

**[06:05]** recurrent neural network is that they're somewhat difficult to learn, because they have the vanishing gradient problem and the exploding gradient problem — which were discovered by Hochreiter, and a little later by Bengio et al. Luckily, the state of neural network research has advanced, and now we can address both problems. I'll quickly present the way we solve them.

The exploding gradient problem means that sometimes your recurrent neural network exhibits very large sensitivity to perturbations, so your gradient is very large. You take a very simple approach: you just say, hey, if the gradient is very large, you shrink it — like

**[06:50]** that. When it comes to the vanishing gradient problem, we use the Long Short-Term Memory, the LSTM. The LSTM is a certain RNN architecture that differs from the regular RNN architecture in such a way that the gradients basically don't want to vanish. The model is not different from the RNN — it's the same model and can compute the same functions — but it's set up in a different way so that the gradients simply don't want to vanish at all.

The way it works is very simple. If you think about an RNN, what it does is compute a totally new hidden state at each time step. If you go back to the figure, you'll see you get a totally

**[07:36]** new hidden state: you combine the hidden state with the weights and pass it through the nonlinearity, which gives you a new hidden state. With the LSTM, instead of computing a new hidden state, we compute a **delta** to the hidden state, which we add to it. It looks like a minor difference — in the LSTM you compute a delta that you add to the hidden state — but the implication is pretty significant. At the end of the sequence, the final hidden state is going to be a sum of many little deltas. So even if you added a little delta to the hidden state at the beginning of your sequence, because the sum distributes its gradients evenly, every time step gets a gradient, and the gradients do not vanish because of the sum.

Now, the LSTM is still sensitive to the order of its input. You might say,

**[08:21]** "but it's a sum, so how does it know about the order?" The reason it knows about the order is that the LSTM decides what to add based on the sequence — it's smart about what to add, but it's still a sum, so the gradients get distributed back evenly.

The LSTM looks like this — a pretty intimidating picture — which is why, even though it was invented almost 20 years ago, it hasn't become really popular until recently. But nonetheless, once you implement the LSTM and get your gradient check to pass, you don't have to think about it anymore. It just works. It *wants* to work, and that's a very good property.

## The main idea

**[09:06]** Now we're ready to understand the main idea behind this work, which is extremely simple. The main idea looks like this: we use an LSTM, we feed it the input sequence, and then we ask it to predict the output sequence. And that's it. There you go — this is the main technical contribution of this work. We use minimum innovation for maximum results. In this figure, you get a sequence of inputs and you just feed it to the hidden state of the LSTM; then you feed it a special symbol that says "hey, the sequence has ended"; and then you just say, okay, now please predict the output sequence. It's really extremely simple.

**[09:52]** From the modeling perspective, there isn't a whole lot going on here. But now let's try to understand what it really means. From an objective-function point of view, we define a distribution over output sequences given the input sequences, and we maximize the log-probability of the correct answer using stochastic gradient descent.

Now, you may notice there's a bottleneck — the hidden state. You've got this long input sequence, which is mapped into a vector, and then you need to produce the target sequence. That may seem like a lot to ask of a hidden state — to remember so much. The way we address it is by using a **large hidden state**. We make sure the model looks like this: we use a deep LSTM. That's

**[10:38]** the minor departure from the previous picture, but again the model is extremely uniform — just this big deep LSTM. The hidden state here is large, so it has a lot of capacity to remember whatever input you want. So that's it — this is the model. We just train it, and because deep neural networks are so powerful, it figures out how to solve the problem.

## Related work

Now there's important related work I need to mention. The first publication that discussed these kinds of models was by Nal Kalchbrenner and Phil Blunsom, which used almost the same model — but they used a convolutional neural network for the encoder and a recurrent network for the decoder. Then later, Cho et al. used almost the same model; this work is simultaneous to ours, with the

**[11:24]** difference that it was primarily used for rescoring — rescoring the proposals of a statistical machine translation system. And finally, the other relevant work is by Dzmitry Bahdanau and Yoshua Bengio, which used an attention mechanism for the same problem, so that they don't need to use a large hidden state. That's very interesting work, a very interesting model.

The main emphasis of our work is simply to show that this simple, uniform, uncreative model can achieve good results if the model is large. Our main motivation was to really try hard to produce the translations directly from the model. The way you should think about it is that this work is a proof that the naive approach just

## Experiments

**[12:11]** works. I'll describe the dataset on which we conducted our experiment: the WMT'14 English-to-French dataset. In machine translation, WMT runs these competitions every year — they release datasets and many teams compete. This dataset is about 700 million words, although we trained on only ~30% of the training data, for a technical/historical reason I won't get into now.

The model we used for the large experiments was not small. It had 160,000 input words and 80,000 output words, and four layers of LSTMs with 1,000 cells — which means you've got an 8,000-dimensional state (1,000 LSTM cells means a 2,000-dimensional state per layer). We also

**[12:56]** used a different LSTM for the input and the output, which gives a total of 384 million parameters. So this is how the model looks — this giant thing: one LSTM for the encoder, a different one for the decoder, a vocabulary of 160,000 words on the input, 80,000 words on the output, and a softmax. We just used a straight softmax of size 80,000, with a thousand dimensions feeding into it. This is probably one of the largest naive softmaxes ever attempted. So — just a big model, and we just train it with stochastic gradient descent. Really straightforward, really simple.

One really important point is that this model **wants to work**. It's no longer the case that neural network training and tuning is

**[13:41]** black magic. The amount of tuning required for this model was small, and all of it is described in this one slide: the batch size is 128; the learning rate is 0.7; the initial scale is uniform between −0.08 and +0.08; we control the norm of the gradient so it's no greater than 5; and we halve the learning rate every half-epoch after 5 epochs. That's all it takes to get this thing to work.

This is really important. In general, if you feel like you want to get into neural nets and you say "oh my God, this tuning is so complicated, what do I do?" — actually, in many cases the tuning is not complicated. There are only four things you need to know to tune your neural network in 90% of cases:

**[14:28]** you need to make sure your learning rate is all right; that the initial scale of the random initialization is all right; that you clip the norm of the gradients if you use a recurrent neural network; and that you don't forget to lower your learning rate at the end. Do these four things and that's it — that's all we did here. We didn't even use momentum. It's a very simple, barebones approach, which was sufficient here.

## Parallelization

Another thing we did to train larger models faster was parallelization. In this work, we used a single 8-GPU machine, and we parallelized the model by placing each layer of the LSTM on a different GPU. This way we were able to get a 3.5× speedup — nothing crazy, but not too bad. And this

**[15:14]** gives us eight times more RAM. However, I should point out that this model can be run on a single K40, which has 12 GB of RAM — RAM is the main limitation. You'd probably get something about two times slower, because the K40 is a faster GPU than the ones we had.

The parallelization is quite neat. Each layer of the LSTM lives on a different GPU, as shown in this figure. Each layer computes its activations independently and sends them to the next LSTM. So you get this kind of computation — it looks like that — and this is how you get the 3.5× speedup. Again, this is a bit of an engineering detail if you want maximum speed, but the model itself, like I said, is just a deep

**[15:59]** LSTM — extremely simple, extremely uniform.

## Visualizing the embeddings

Now, one nice property of this model is that we embed a sequence into a vector — we've got this nice way of embedding a variable-sized object into a single embedding. This thing is a vector, and we can try to see what it does. We get pretty interesting, nice-looking effects. For example, we took six sentences — "Mary admires John," "Mary is in love with John," "Mary respects John," "John admires Mary," "John is in love with Mary," "John respects Mary" — we computed their vector representations, took a two-dimensional PCA, and we got this nice automatic clustering.

**[16:44]** It's apparent that the model has learned something non-trivial about embedding. It has preserved the order of, I believe, the verbs, and at the same time it cares about the order of the people. Likewise, in this example it's pretty interesting, because here we have three ways of saying the same thing and three ways of saying something different: "I was given a card by her in the garden," "In the garden, she gave me a card," "She gave me a card in the garden." All these sentences say the same thing in different ways — and these other sentences say the opposite — and it figured out to cluster them. It was smart

**[17:29]** enough to cluster them automatically. So this is a nice effect. What it means is that if you have an objective function where you want to embed a sequence into a vector, you can just use a deep LSTM — or even a shallow LSTM — and it will probably be very reasonable.

## Results

The results: we use the BLEU score to evaluate our performance, where higher is better. The winner of the WMT'14 contest got 37.0, and an ensemble of five of our LSTM models got 34.8 — which means we're pretty close to the best entry of WMT'14, but not there yet.

We can do a bit of error analysis and ask: what happens on long sequences? Does

**[18:15]** performance deteriorate when you look at longer sentences? The answer is basically no. Even when a sentence has 35 words, it does just fine, and only when you go all the way to 79 words is there a small drop in performance. The red curve is a baseline — an untuned phrase-based system.

Where our system suffers is on **out-of-vocabulary words**. When we sort the test sentences by the number of out-of-vocabulary words they have, you see that our performance degrades very rapidly, and this is a problem. I want to point out a follow-up work, done primarily by Thang Luong when he was an intern at Google — along with myself, Quoc Le, Oriol Vinyals, and others — where we used a very simple idea to address the out-of-vocabulary problem that required very little coding, which

**[19:01]** brought us all the way to 37.5 BLEU points, whereas the best entry of WMT'14 was 37.0. So that's quite nice. The paper is available on arXiv and on our websites.

## Conclusion

To conclude: we presented a general, simple, primitive approach for solving sequence-to-sequence problems, and we showed that it works on machine translation — and works well. Machine translation is a problem where you already have good solutions, so this means that if you have a sequence-to-sequence problem, or something you want to formulate as one, you just apply this and it will work. A notable recent application of ideas like this is the recent image caption generation. I really see this as a step toward completely solving the supervised learning problem.

**[19:47]** So the real conclusion is: if you have a large dataset and you train a very large deep neural network, then success is guaranteed. Thank you very much.

## Q&A

**[19:47]** *Moderator:* We have time for a couple of quick questions.

**Q:** Just a quick question — what if there's a big mismatch between the dimensions of the input sequence and the output sequence?

**A:** What do you mean by mismatch?

**Q:** For example, the input sequence is a musical note and the output sequence is a huge analog signal.

**A [20:35]:** Basically, the main limitation of this model is that it doesn't like very long sequences. As long as your sequences aren't very long, it's going to be okay. It doesn't care — it will just find a way.

**Q:** Thanks for a very nice talk. There are many things you might try to do with these representations after you've learned them — like predicting properties of the input sentence, recapitulating the input sentence, or extracting anything that an NLP system might want to extract. So I have two questions, both of which are probably follow-up work. One: once you've trained the system to do machine translation or auto-encoding, what can you get from these representations if you freeze the

**[21:21]** weights? And the other question: can you improve performance if you do multitask learning — so you're not just doing translation, but you may have supervision on some sentences?

**A:** Okay. The answer is that we haven't investigated very thoroughly the properties of the representations. When it comes to unsupervised representations, the situation is always a bit tricky, because the objective function doesn't really try to make a representation that's convenient for other tasks. It is definitely true that multitask learning will help, especially in situations where you don't have a lot of data. If you *do* have a lot of data, I think multitask learning might even hurt, because you just want to focus on the big dataset and dedicate all

**[22:06]** the network's resources to the problem you want to solve.

**Q:** I might expect that the more different the source and target languages are, the more interesting linguistic features might have to be encoded in order to transform one into the other.

**A:** It's definitely very possible. The network — I think it would be interesting to understand exactly how it works — and it is likely that it will encode very interesting linguistic features.

**[22:06]** *Moderator:* Last question.

**Q:** I was curious about your grand claim that this can solve any sequence-to-sequence model, given that you only have like four things to do in training. What other sequence-to-sequence problems have you run this on?

**A [22:52]:** Well, we'll have a paper online in a few weeks. It's pretty cool.

**Q:** You can't just say the task?

**A:** A few weeks, I promise.

*Moderator:* Thanks, let's thank Ilya again.
