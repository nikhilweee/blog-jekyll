---
layout: post
title: Building your first RNN
subtitle: with PyTorch 0.4! (1/2)
tags: [rnn, pytorch]
category: [learn]
published: false
mathjax: true
---
> This is part one of a two-part series on getting started with RNNs using PyTorch. Part two can be accessed at [Building your first RNN - Part 2]({% post_url 2018-04-25-first-rnn-pytorch-2 %})

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

{% gist 13243631f8ed219167ccd3866ce3204e cipher.py %}

We create the `encode` function which uses the parameters `vocab` and `key` to encrypt each character. Since we're working with letters, `vocab` in this context simply means the alphabet.  The encryption algorithm should be fairly easy to understand. Notice how we use the modulo operator in line `8` to prevent the indexes from overflowing.

To check the implementation, you can check for some random inputs. For example, ensure that `encrypt('ABCDEFGHIJKLMNOPQRSTUVWXYZ-')` returns `NOPQRSTUVWXYZ-ABCDEFGHIJKLM`.

## The Dataset (Finally!)

Okay, let's finally build the dataset. For the sake of simplicity, we'll use a random sequence of characters as a message and encrypt it to create the input to the LSTM. To implement this, we create a simple function called `dataset` which takes in the parameter `num_examples` and returns a list of those many (input, output) pairs.

{% gist 13243631f8ed219167ccd3866ce3204e batch.py %}

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

With this we're done with the basics. We'll build the actual network in [Part 2]({% post_url 2018-04-25-first-rnn-pytorch-2 %})
