---
layout: post
title: Evaluating Machine Evaluation Metrics
subtitle: Automatic evaluation metrics explained
tags: [cube, algorithms]
category: Explanations
published: true
mathjax: true
---

Over the years, neural networks have only become smarter and better at understanding the world around us. Consider language, for example. Today, these giants very well possess the ability to understand human languages. So much better that they've already started to transform and mould human languages in different domains. Machine translation is a wonderful example.

At a finer level, these giants do not seem to possess any form of intelligence. They literally seem to be just number crunching machines! But zoom out a little bit, and there you would find them excelling at the hardest of tasks.

With the ever increasing focus on machine translation, human evaluations were difficult and time-consuming. One couldn't wait for weeks in order to test a system that changes every day. Of course there was this factor of subjectivity too, so we needed some kind of automated evaluation scheme to compare various models.

## Precision and Recall

If you're dealing with NLP, you better know the terms _precison_ and _recall_. If that's not the case, however, read on! There's always got to be a first time!

Let's say I want to test your memorization skills, so I ask you to play a brain game. The game is simple. I show you a bunch of objects, and all you have to do is remember as much of them as possible.
Easy? Cool. Time's up. Let's test you out.

Let's say you call out a bunch of items $$ H = \{ h_1, h_2, \dots h_m \} $$, while I actually showed you $$ T = \{ t_1, t_2, \dots t_n \} $$.

Recall, in this case, is quite simply the number of items that you can correctly recall! So out of the $$ H $$ items you think you saw, you only recall $$ H \cap T $$ of them correctly. How many of the $$ T $$ items did you actually remember? Yup, that's your recall rate.

$$ Recall = \frac{|H \cap T|}{|T|} $$

Precision, again, is intuitive. It's just how often do you get your judgements correct. So out of the 
$$ | H | $$ attempts you make, you were right only $$ | H \cap T | $$ times. Your precision?

$$ Precision = \frac{|H \cap T|}{|H|} $$

Easy, right? Okay, let's move on to BLEU now. But if you ever have a hard time with these terms, just remember the brain game we played. :simple_smile:

One the most popular evaluation metrics was BLEU, and this post is an attempt to explain one of the most popular metrics for evaluating machine translation.

## Introducing BLEU

BLEU, short for **B**i**L**ingual **E**valuation **U**nderstudy, was originally introduced in 2002 by [Papineni et al.](http://www.aclweb.org/anthology/P02-1040.pdf). The idea here is simple. We compare the predicted hypothesis from our system (H) with an already existing target translation (T). The closer they are, the better.

So, how do we measure _closeness_? A simple way is to compare words from H and T one at a time. The more matches we find, the closer they are. Fair enough? Let's try it out.

Let's say our task is to translate sentences from Chinese to English. We feed in the chinese sentence, and get the following output (H).

```
H : There is a cat on the mat
```

A professional human translator, however, suggests that the translation should rather be (T).

```
T : The cat is on the mat
```