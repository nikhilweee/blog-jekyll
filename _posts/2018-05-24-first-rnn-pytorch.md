---
layout: post
title: Building your first RNN
subtitle: with PyTorch 0.4!
tags: [rnn, pytorch]
category: Explanations
published: true
mathjax: true
redirect_from:
  - /posts/2018/first-rnn-pytorch-1/
  - /posts/2018/first-rnn-pytorch-2/
  - /posts/2018/first-rnn-pytorch-3/
---
If you have some understanding of recurrent networks, want to get your hands dirty, but haven't really tried to do that on your own, then you are certainly at the right place. This tutorial is a practical guide about getting started with recurrent networks using PyTorch. We'll solve a simple cipher using PyTorch 0.4.0, which is the latest version at the time of this writing.  

You are only expected to have some understanding of recurrent networks. If you don't, here's the link to the [golden resource](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Chris Olah's post on Understanding LSTMs. We'll use a single layer LSTM for the task of learning ciphers, which should be a fairly easy exercise.

## The Problem

Before starting off, let's first define the problem in a concrete manner. We wish to decrypt secret messages using an LSTM. For the sake of simplicity, let's assume that our messages are encrypted using the [Caesar Cipher](https://en.wikipedia.org/wiki/Caesar_cipher), which is a really simple substitution cipher.  

Caesar cipher works by replacing each letter of the original message by another letter from a given alphabet to form an encrypted message. In this tutorial we'll use a right shift of 13, which basically means that the encrypted version of each letter in the alphabet is the one which occurs 13 places to the right of it. So `A`(1) becomes `N`(1+13), `B`(2) becomes `O`(2+13), and so on. Our alphabet will only include uppercase English characters `A` through `Z`, and an extra letter, `-`, to represent any foreign character.

With all of these in mind, here's the substitution table for your reference.

```
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z -
N O P Q R S T U V W X Y Z - A B C D E F G H I J K L M
```

The first row shows all the letters of the alphabet in order. To encrypt a message, each letter of the first row can be substituted by the corresponding letter from the second row. As an example, the message `THIS-IS-A-SECRET` becomes `FUVEMVEMNMERPDRF` when encrypted.

[^why-nn]Aside : but [why use neural networks for this problem?](#fn:why-nn)

[^why-nn]: **But why Neural Networks?** You might be wondering why do we use neural networks in the first place. In our use case, it sure makes more sense to decrypt the messages by conventional programming because we _know_ the encryption function beforehand. _This might not be the case everytime_. You might have a situation where you have enough data but still have no idea about the encryption function. Neural networks fit quite well in such a situation. Anyways, keep in mind that this is still a toy problem. One motivation to choose this problem is the ease of generating loads of training examples on the fly. So we don't really need to procure any dataset. Yay!

## The Dataset

Like any other neural network, we'll need data. Loads of it. We'll use a parallel dataset of the following form where each tuple represents a pair of (encrypted, decrypted) messages. 
```
('FUVEMVEMNMERPDRF', 'THIS-IS-A-SECRET')
('FUVEMVEMN-AFURDMERPDRF', 'THIS-IS-ANOTHER-SECRET')
...
```
Having defined our problem, we'll feed the `encrypted` message as the input to our LSTM and expect it to emit the original message as the target. Sounds simple right?

It does, except that we have a little problem. Neural networks are essentially number crunching machines, and have no idea how to hande our encrypted messages. We'll somehow have to convert our strings into numbers for the network to make sense of them.

## Word Embeddings

The way this is usually done is to use something called as word embeddings. The idea is to represent every character in the alphabet with its own $$ D $$ dimensional **embedding vector**, where $$ D $$ is usually called the embedding dimension. So let's say if we decide to use an `embedding_dim` of 5. This basically means that each of the 27 characters of the alphabet, `ABCDEFGHIJKLMNOPQRSTUVWXYZ-`, will have their own embedding vector of length 5.

Often, these vectors are stored together as $$ V \times D $$ dimensional **embedding matrix**, $$ E $$, where each row $$ E[i] $$ of the matrix represents the embedding vector for the character with index $$ i $$ in the alphabet. Here $$ V $$ is the length of the vocabulary (alphabet), which is 27 in our case. As an example, the whole embedding matrix $$ E $$ might look something like the one shown below.

```
[[-1.4107, -0.8142,  0.8486,  2.8257, -0.7130],
 [ 0.5434,  3.8553,  2.9420, -2.8364, -4.0077], 
 [ 1.6781, -0.2496,  2.5569, -0.2952, -2.2911],
 ...
 [ 2.7912,  1.3261,  1.7603,  3.3852, -2.1643]]
```
$$ E[0] $$ then represents the word vector for `A`, which is `[-1.4107, -0.8142,  0.8486,  2.8257, -0.7130]`.

[^char-embedding]Aside : but [I read something different!](#fn:char-embedding)

[^char-embedding]: **I think I read something different!** Strictly speaking, what I just described here is called a _character embedding_, beacause we have a vector for each _character_ in the alphabet. In case we had a vector for each _word_ in a vocabulary, we would be using _word embeddings_ instead. Notice the analogy here. An alphabet is the set of all the letters in a language. Similarly, a vocabulary is the collection of all the words in a language. 

P.S. I'll be using alphabet and vocabulary interchangably throughout this tutorial. Similarly, word embeddings, word vectors, character embeddings, or simply embeddings will mean the same thing.

## The Cipher

Now that we have enough background, let's get our hands dirty and finally jump in to writing some code. The first thing we have to do is to create a dataset. And to do that, we first need to implement the cipher. Although we implement it as a simple function, it might be a good idea to implement the cipher as a class in the future.

{% gist 13243631f8ed219167ccd3866ce3204e module-cipher.py %}

We create the `encode` function which uses the parameters `vocab` and `key` to encrypt each character. Since we're working with letters, `vocab` in this context simply means the alphabet.  The encryption algorithm should be fairly easy to understand. Notice how we use the modulo operator in line `8` to prevent the indexes from overflowing.

To check the implementation, you can check for some random inputs. For example, ensure that `encrypt('ABCDEFGHIJKLMNOPQRSTUVWXYZ-')` returns `NOPQRSTUVWXYZ-ABCDEFGHIJKLM`.

## The Dataset (Finally!)

Okay, let's finally build the dataset. For the sake of simplicity, we'll use a random sequence of characters as a message and encrypt it to create the input to the LSTM. To implement this, we create a simple function called `dataset` which takes in the parameter `num_examples` and returns a list of those many (input, output) pairs.

{% gist 13243631f8ed219167ccd3866ce3204e module-batch.py %}

There's something strange about this function though. Have a look at line 24. We're not returning a pair of strings. We're first converting strings into a list of indices which represent their position in the alphabet. If you recall the section on [word embeddings](#word-embeddings), these indices will later be used to extract the corresponding embedding vectors from the embedding matrix $$ E $$. We're then converting these lists into a pair of tensors, which is what the function returns.

## Tensors?

This brings us to the most fundamental data type in PyTorch - the Tensor. For users familiar with NumPy, a tensor is the PyTorch analogue of `ndarray`. If you're not, a tensor is essentially a multidimensional matrix which supports optimized implementations of common operations. Have a look at the [Tensor Tutorial](http://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html) on the PyTorch website for more information. The takeaway here is that we'll use tensors from now on as our go to data structure to handle numbers. Creating a tensor is really easy. Though there are a lot of ways to do so, we'll just wrap our list of integers with `torch.tensor()` - which turns out the easiest amongst all.

You can satisfy yourself by having a look at what this function does. A quick call to `dataset(1)` should return something similar to the following. You can also verify that the numbers in the second tensor are right shifted by 13 from the numbers in the first tensor. `20 = (7 + 13) % 27`, `3 = (17 + 13) % 27` and so on.

```python
[[tensor([ 20,   3,  21,   0,  14,   4,   2,   4,  13,  12,   8,  23,
         12,  10,  25,  17,  19,   1,   2,  22,  12,  15,  16,   3,
         13,  10,  20,  23,  25,  15,  19,   4]), 
  tensor([  7,  17,   8,  14,   1,  18,  16,  18,   0,  26,  22,  10,
         26,  24,  12,   4,   6,  15,  16,   9,  26,   2,   3,  17,
          0,  24,   7,  10,  12,   2,   6,  18])]]
```

With this we're done with the basics. Let's start building the network. It's a good idea to first have a general overview of what we aim to achieve. One might think of something along the following lines.

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

To feed in inputs, well, we first need to prepare the inputs. Remember the embedding matrix $$ E $$ we described [earlier](#the-dataset-finally)? we'll use $$ E $$ to convert the pair of indices we get from `dataset()` into the corresponding embedding vectors. Following the general paradigm, we create an instance of `torch.nn.Embedding`.

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

Let's wrap this up and consolidate the network. Have a look at the training script below. Most of the code should make sense on its own. There are a few helper operations like `torch.squeeze` and `torch.transpose` whose function can be inferred from the comments. You can also refer to the [docs](https://pytorch.org/docs/stable/torch.html) for more information.

{% gist 13243631f8ed219167ccd3866ce3204e module-train.py %}

After every training iteration, we need to evaluate the network. Have a look at the validation script below. After calculating the scores as in the training script, we calculate a softmax over the scores to get a probability distribution in line 9. We then aggregate the characters with the maximum probability in line 11. We then compare the predicted output `batch_out` with the target output `original` in line 15. At the end of the epoch, we calculate the accuracy in line 18.

{% gist 13243631f8ed219167ccd3866ce3204e module-valid.py %}

Notice that the predicted outputs are still in the form of indices. Converting them back to characters is left as an exercise.

But before you go, congratulations! You've built your first RNN in PyTorch! The complete code for this post is available as a [GitHub gist](https://gist.github.com/nikhilweee/13243631f8ed219167ccd3866ce3204e). You can test the network by simply running the [training script](https://gist.github.com/nikhilweee/13243631f8ed219167ccd3866ce3204e#file-train-py). Thanks for sticking around.
