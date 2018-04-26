---
layout: post
title: Getting started with LSTMs
subtitle: using PyTorch!
tags: [rnn, pytorch]
category: [learn]
published: false
---
> PyTorch 0.4 has just been released. What a good time to write this blog post!

If you have some understanding of recurrent networks, want to get your hands dirty, but haven't really tried to do that on your own, you are at the right place. This tutorial is a practical guide about getting started with recurrent networks using PyTorch. We'll solve a simple cipher using PyTorch 0.4.0, which is the latest version at the time of this writing.  

You are only expected to have some understanding of recurrent networks. If you don't, here's the link to the [golden resource](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) (Chris Olah's post on Understanding LSTMs). I'll use a single layer LSTM for the task of learning ciphers, which should be a fairly easy exercise.

## The Problem

Before starting off, let's first define the problem in a concrete manner. Let's say we wish to decrypt secret messages using an LSTM. For the sake of simplicity, let's assume that our messages are encrypted using the [Caesar Cipher](https://en.wikipedia.org/wiki/Caesar_cipher), which is a simple substitution cipher.  

The way Caesar cipher works is by replacing each letter of the original message by another letter from the alphabet to form the encrypted message. In this tutorial we'll use a right shift of 13, which basically means that `A`(1) becomes `N`(1+13), `B`(2) becomes `O`(2+13), and so on. We'll also include an extra character in our alphabet, the `-`, to represent all non-alphanumeric characters.

With all of these in mind, here's the substitution table for your reference. Each letter of the first row is replaced with the corresponding letter from the second row.

```
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z -
N O P Q R S T U V W X Y Z - A B C D E F G H I J K L M
```

As an example, the message `THIS-IS-A-SECRET` becomes `FUVEMVEMNMERPDRF` when encrypted. Now you might be wondering why do we need neural networks in the first place. You're right. In our use case, it makes more sense to decrypt the messages by conventional programming because we _know_ the encryption function beforehand. _This may not be the case everytime_. Another simple reason to choose this problem is that we could generate loads of training examples on the fly. So we don't really need to procure any dataset. Yay!

## The Dataset

Talking about data, let's say we have a parallel dataset of the following form, where each tuple represents a pair of (encrypted, decrypted) messages. Having defined our problem, we'll feed the `encrypted` message as the input to our LSTM and expect it to emit the original message as the target. Sounds simple right?
```
('FUVEMVEMNMERPDRF', 'THIS-IS-A-SECRET')
('FUVEMVEMN-AFURDMERPDRF', 'THIS-IS-ANOTHER-SECRET')
...
```

It does, except that we have a little problem. Neural networks are essentially number crunching machines, and have no idea how to hande our encrypted messages. We'll somehow have to convert our strings into numbers for the network to make sense of them.

## Word Embeddings
The way this is usually done is to use something called as _word embeddings_. The idea is to represent every character in the alphabet with its own `N` dimensional vector. Note that `N` is usually called the embedding dimension. So let's say if we decide to use an `embedding_dim` of 20, this basically means that each of the 27 characters of the alphabet, `ABCDEFGHIJKLMNOPQRSTUVWXYZ-`, will have their own vector of size 20.

Strictly speaking, what I just described here is called a _character embedding_, beacause we have a vector for each _character_. In case we had a vector for each _word_ in the sentence, we would be using _word embeddings_ instead.  

## The Cipher
Now that we have enough background, let's get our hands dirty and finally jump in to writing some code. The first thing we have to do is to create a dataset. And to do that, we first need to implement the cipher. Although we implement it as a simple function, it might be a good idea to implement the cipher as a class in the future.

We create the `encode` function which uses the parameters `vocab` and `key` to encrypt each character. The encryption algorithm should be fairly easy to understand. Notice how we use the modulo in line `8` to prevent the indexes from overflowing.

{% gist 13243631f8ed219167ccd3866ce3204e cipher.py %}

To check the implementation, you can check for some random inputs. For example, ensure that `encrypt('ABCDEFGHIJKLMNOPQRSTUVWXYZ-')` returns `NOPQRSTUVWXYZ-ABCDEFGHIJKLM`.

## The Dataset (Again!)
Okay, let's finally build the dataset. For the sake of simplicity, we'll use a random sequence of characters as a message and encrypt it to create the input to the LSTM. To implement this, we create a simple function called `dataset` which takes in the parameter `num_examples` and returns a list of those many (input, output) pairs.

{% gist 13243631f8ed219167ccd3866ce3204e batch.py %}

As described earlier, we want to train our network to decrypt secret messages. So we'll feed in the encrypted message as the input and expect our model to correctly predict the original message - essentially decrypting the encrypted message.

Notice line 24. We're not returning a pair of strings. We're returning a pair of tensors. 

_Tensors?!_ :hushed: Keep calm! These are just some inbuilt pytorch data structures to speed up numerical computations. For users familiar with NumPy, a tensor is the PyTorch analogue of `ndarray`. If you're not, a tensor is essentially an multidimensional array which supports optimized implementations of common operations. Have a look at the [Tensor Tutorial](http://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html) on the pytorch website for more information. The takeaway here is that we'll use tensors from now on whenever we think of numbers. Creating a tensor is really easy. Though there are a lot of ways to do so, we'll just wrap our list of integers with `torch.tensor()` - which turns out the easiest way to do so.


But why aren't we returning strings? Good question. Save that for now. I'll answer it later on. For now, you can satisfy yourself by having a look at what this function does. A quick call to `dataset(1)` should return something similar to this. You can also verify that the numbers in the second tensor are right shifted by 13 from the numbers in the first tensor. `20 = (7 + 13) % 27`, `3 = (17 + 13) % 27` and so on.

```
[[tensor([ 20,   3,  21,   0,  14,   4,   2,   4,  13,  12,   8,  23,
         12,  10,  25,  17,  19,   1,   2,  22,  12,  15,  16,   3,
         13,  10,  20,  23,  25,  15,  19,   4]), 
  tensor([  7,  17,   8,  14,   1,  18,  16,  18,   0,  26,  22,  10,
         26,  24,  12,   4,   6,  15,  16,   9,  26,   2,   3,  17,
          0,  24,   7,  10,  12,   2,   6,  18])]]
```

# Let's get building!

Let's finally start building our model! Let's first have a general overview of what we aim to achieve. One might think of something along the following lines.

>On a very high level, the first step in a general workflow will be to feed in inputs to our LSTM to get the predictions. Next, we pass on the predictions along with the targets to the loss function to calculate the loss. Finally, we backpropagate through the loss to update our model's parameters.

Hmm, that sounds easy, right? But how do you actually make it work? Let's dissect the problem step by step. We'll first identify the components needed to build our model, and finally put them to gether as a single piece to make it work.

> ... feed in inputs to our LSTM ...

To feed in inputs, we first need to have an LSTM. Pytorch makes it really easy to do so. You simply create an instance of `torch.nn.LSTM` as shown in line `6`. `nn.LSTM` implements   Out of the many possible parameters listed in the [docs for nn.LSTM](http://pytorch.org/docs/stable/nn.html#torch.nn.LSTM) Notice that I've passed on two parameters - `embedding_dim` and `hidden_dim`. 

{% gist 13243631f8ed219167ccd3866ce3204e model.py %}

Let's say you want to feed in an encrypted string into the LSTM to see what it predicts. In such a case, our input could be something like `FUVEMVEMNMERPDRF` and the corresponding target could be `THIS-IS-A-SECRET`. Since we're using character embeddings, our _input sequence_ is a vector containing the corresponding character embedding for each character in our input `FUVEMVEMNMERPDRF`.

In PyTorch, an LSTM is initialized by creating an instance of `torch.nn.LSTM` as shown. The corresponding arguments are explained below.

```
lstm = torch.nn.LSTM(input_size=20, hidden_size=10, num_layers=1)
```

* `input_size: The number of expected features in the input 'x'`  
   The `input_size` here refers to the size of each entity in the `input_sequence`. Since we're using an `embedding_dim` of 20, each character will be represented by a vector of size 20, which is the `input_size`

* `hidden_size: The number of features in the hidden state 'h'`  
   This basically asks for the size of the hidden vector. In the context of an LSTM, this refers to the size of both, the cell state `c` and the hidden state `h`. This also means that both the states have to be of the same size. In this tutorial, we're using a `hidden_size` of 10.

* `num_layers: Number of recurrent layers.`  
   Using an number greater than 1 basically means we're using a stacked LSTM, where the output of each layer is the input for the next layer, and the final output is that from the last layer.


## Guide to Mini Batching in PyTorch


