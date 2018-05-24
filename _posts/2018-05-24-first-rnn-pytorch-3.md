---
layout: post
title: Building your first RNN
subtitle: with PyTorch 0.4! (Part 3/3)
tags: [rnn, pytorch]
category: [learn]
published: true
mathjax: true
---
> This is part three of a three-part series on getting started with RNNs using PyTorch. Part two can be accessed at [Building your first RNN - Part 2]({% post_url 2018-05-24-first-rnn-pytorch-2 %}). Part one is available at [Building your first RNN - Part 1]({% post_url 2018-05-24-first-rnn-pytorch-1 %})

Having built the network in [Part 2]({% post_url 2018-05-24-first-rnn-pytorch-2 %}), have a look at the training script below. Most of the code should make sense on its own. There are a few helper operations like `torch.squeeze` and `torch.transpose` whose function can be inferred from the comments. You can also refer to the [docs](https://pytorch.org/docs/stable/torch.html) for more information.

{% gist 13243631f8ed219167ccd3866ce3204e module-train.py %}

After every training iteration, we need to evaluate the network. Have a look at the validation script below. After calculating the scores as in the training script, we calculate a softmax over the scores to get a probability distribution in line 9. We then aggregate the characters with the maximum probability in line 11. We then compare the predicted output `batch_out` with the target output `original` in line 15. At the end of the epoch, we calculate the accuracy in line 18.

{% gist 13243631f8ed219167ccd3866ce3204e module-valid.py %}

Notice that the predicted outputs are still in the form of indices. Converting them back to characters is left as an exercise.

But before you go, congratulations! You've built your first RNN in PyTorch! The complete code for this post is available as a [GitHub gist](https://gist.github.com/nikhilweee/13243631f8ed219167ccd3866ce3204e). You can test the network by simply running the [training script](https://gist.github.com/nikhilweee/13243631f8ed219167ccd3866ce3204e#file-train-py). Thanks for sticking around.
