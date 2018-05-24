---
layout: post
title: Building your first RNN
subtitle: with PyTorch 0.4! (Part 2/3)
tags: [rnn, pytorch]
category: [learn]
published: true
mathjax: true
---
> This is part two of a three-part series on getting started with RNNs using PyTorch. Part one can be accessed at [Building your first RNN - Part 1]({% post_url 2018-05-24-first-rnn-pytorch-1 %}). Part three is available at [Building your first RNN - Part 3]({% post_url 2018-05-24-first-rnn-pytorch-3 %})

Having described the problem and built the dataset in [Part 1]({% post_url 2018-05-24-first-rnn-pytorch-1 %}), let's finally start building our model. It's a good idea to first have a general overview of what we aim to achieve. One might think of something along the following lines.

>On a very high level, the first step in a general workflow will be to feed in inputs to an LSTM to get the predictions. Next, we pass on the predictions along with the targets to the loss function to calculate the loss. Finally, we backpropagate through the loss to update our model's parameters.

Hmm, that sounds easy, right? But how do you actually make it work? Let's dissect this step by step. We'll first identify the components needed to build our model, and finally put them to gether as a single piece to make it work.

<div class="note" markdown="1">

### The PyTorch paradigm

... before diving in, it's important to know a couple of things. PyTorch provides implementations for most of the commonly used entities from layers such as LSTMs, CNNs and GRUs to optimizers like SGD, Adam, and what not (Isn't that the whole point of using PyTorch in the first place?!). The general paradigm to use any of these entities is to first create an instance of `torch.nn.entity` with some required parameters. As an example, here's how we instantiate an `lstm`. 

```python
# Step 1
lstm = torch.nn.LSTM(input_size=5, hidden_size=10, batch_first=True)
```

Next, we call this object with the inputs as parameters when we actually want to run an LSTM over some inputs. This is shown in the third line below.

```python
lstm_in = torch.rand(40, 20, 5)
hidden_in = (torch.zeros(1, 40, 10), torch.zeros(1, 40, 10))
# Step 2
lstm_out, lstm_hidden = lstm(lstm_in, hidden_in)
```

This two-stepped process will be seen all through this tutorial and elsewhere. Below, we'll go through step 1 of all the modules. We'll connect the dots at a later stage.

</div>

Getting back to code now, let's dissect our 'high level' understanding again.

## 1. Prepare inputs

>... **feed in inputs** to an LSTM to get the predictions ...

To feed in inputs, well, we first need to prepare the inputs. Remember the embedding matrix $$ E $$ we described [earlier]({% post_url 2018-05-24-first-rnn-pytorch-1 %}#the-dataset-finally)? we'll use $$ E $$ to convert the pair of indices we get from `dataset()` into the corresponding embedding vectors. Following the general paradigm, we create an instance of `torch.nn.Embedding`.

The [docs](https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding) list two required parameters - `num_embeddings: the size of the dictionary of embeddings` and `embedding_dim: the size of each embedding vector`. In our case, these are `vocab_size` $$ V $$ and `embedding_dim` $$ D $$ respectively.

```python
# Step 1
embed = torch.nn.Embedding(vocab_size, embedding_dim)
```

Later on, we could easily convert any input tensor `ecrypted` containing indices of the encrypted input (like the one we get from `dataset()`) into the corresponding embedding vectors by simply calling `embed(encrypted)`.

As an example, the word `SECRET` becomes `ERPDRF` after encryption, and the letters of `ERPDRF` correspond to the indices `[4, 17, 15, 3, 17, 5]`. If `encrypted` is `torch.tensor([4, 17, 15, 3, 17, 5])`, then `embed(encrypted)` would return something similar to the following.

```python
# Step 2
>>> encrypted = torch.tensor([4, 17, 15, 3, 17, 5])
>>> embedded = embed(encrypted)
>>> print(embedded)
tensor([[ 0.2666,  2.1146,  1.3225,  1.3261, -2.6993],
        [-1.5723, -2.1346,  2.6892,  2.7130,  1.7636],
        [-1.9679, -0.8601,  3.0942, -0.8810,  0.6042],
        [ 3.6624, -0.3556, -1.7088,  1.4370, -3.2903],
        [-1.5723, -2.1346,  2.6892,  2.7130,  1.7636],
        [-1.8041, -1.8606,  2.5406, -3.5191,  1.7761]])
```


## 2. Build an LSTM

>... feed in inputs **to an LSTM** to get the predictions ...

Next, we need to create an LSTM. We do this in a similar fashion by creating an instance of `torch.nn.LSTM`. This time, the [docs](https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM) list the required parameters as `input_size: the number of expected features in the input` and `hidden_size: the number of features in the hidden state`. Since LSTMs typically operate on variable length sequences, the `input_size` refers to the size of each entity in the input sequence. In our case, this means the `embedding_dim`. This might sound counter-intuitive, but if you think for a while, it makes sense.

`hidden_size`, as the name suggests, is the size of the hidden state of the RNN. In case of an LSTM, this refers to the size of both, the `cell_state` and the `hidden_state`. Note that the hidden size is a hyperparameter and _can be different_ from the input size. [colah's blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) doesn't explicitly mention this, but the equations on the PyTorch [docs on LSTMCell](https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell) should make it clear. To summarize the discussion above, here is how we instantiate the LSTM.

```python
# Step 1
lstm = torch.nn.LSTM(embedding_dim, hidden_dim)
```

<div class="note" markdown="1">

### A note on dimensionality

During step 2 of the [general paradigm](#the-pytorch-paradigm), `torch.nn.LSTM` expects the input to be a 3D input tensor of size `(seq_len, batch, embedding_dim)`, and returns an output tensor of the size `(seq_len, batch, hidden_dim)`. We'll only feed in one input at a time, so `batch` is always `1`. 

As an example, consider the input-output pair `('ERPDRF', 'SECRET')`. Using an `embedding_dim` of 5, the 6 letter long input `ERPDRF` is transformed into an input tensor of size `6 x 1 x 5`. If `hidden_dim` is 10, the input is processed by the LSTM into an output tensor of size `6 x 1 x 10`.

</div>

Generally, the LSTM is expected to run over the input sequence character by character to emit a probability distribution over all the letters in the vocabulary. So for every input character, we expect a $$ V $$ dimensional output tensor where $$ V $$ is 27 (the size of the vocabulary). The most probable letter is then chosen as the output at every timestep.

If you have a look at the output of the LSTM on the example pair `('ERPDRF', 'SECRET')` [above](#a-note-on-dimensionality), you can instantly make out that the dimensions are not right. The output dimension is `6 x 1 x 10` - which means that for every character, the output is a $$ D $$ (10) dimensional tensor instead of the expected 27.

So how do we solve this?

## 3. Transform the outputs

>... feed in inputs to an LSTM to **get the predictions** ...

The general workaround is to transform the $$ D $$ dimensional tensor into a $$ V $$ dimensional tensor through what is called an affine (or linear) transform. Sparing the definitions aside, the idea is to use matrix multiplication to get the desired dimensions.

Let's say the LSTM produces an output tensor $$ O $$ of size `seq_len x batch x hidden_dim`. Recall that we only feed in one example at a time, so `batch` is always `1`. This essentially gives us an output tensor $$ O $$ of size `seq_len x hidden_dim`. Now if we multiply this output tensor with another tensor $$ W $$ of size `hidden_dim x embedding_dim`, the resultant tensor $$ R = O \times W $$ has a size of `seq_len x embedding_dim`. Isn't this exactly what we wanted?

To implement the linear layer, ... you guessed it! We create an instance of `torch.nn.Linear`. This time, the [docs](https://pytorch.org/docs/stable/nn.html#torch.nn.Linear) list the required parameters as `in_features:  size of each input sample` and `out_features:  size of each output sample`. Note that this only transforms the last dimension of the input tensor. So for example, if we pass in an input tensor of size `(d1, d2, d3, ..., dn, in_features)`, the output tensor will have the same size for all but the last dimension, and will be a tensor of size `(d1, d2, d3, ..., dn, out_features)`.

With this knowledge in mind, it's easy to figure out that `in_features` is `hidden_dim`, and `out_features` is `vocab_size`. The linear layer is initialised below. 

```python
# Step 1
linear = torch.nn.Linear(hidden_dim, vocab_size)
```

With this we're preddy much done with the essentials. Time for some learning!

## 4. Calculate the loss

> Next, we pass on the predictions along with the targets to the loss function to calculate the loss.

If you think about it, the LSTM is essentially performing multi-class classification at every time step by choosing one letter out of the 27 characters of the vocabulary. A common choice in such a case is to use the cross entropy loss function `torch.nn.CrossEntropyLoss`. We initialize this in a similar manner. 

```python
loss_fn = torch.nn.CrossEntropyLoss()
```

You can read more about cross entropy loss in the excellent [blog post by Rob DiPietro.](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/)

## 5. Optimize

> Finally, we backpropagate through the loss to update our model’s parameters.

A popular choice is the Adam optimizer. Here's how we initialize it. Notice that almost all torch layers have this convenient way of getting all their parameters by calling `module.parameters()`.

```python
optimizer = torch.optim.Adam(list(embed.parameters()) + list(lstm.parameters())
                             + list(linear.parameters()), lr=0.001)
```

To summarize, here's how we initialize the required layers.

{% gist 13243631f8ed219167ccd3866ce3204e module-model.py %}

We'll wrap this up and consolidate the network in [Part 3]({% post_url 2018-05-24-first-rnn-pytorch-3 %})
