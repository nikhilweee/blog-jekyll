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

Talking about data, let's say we have a parallel dataset of the following form, where each tuple represents a pair of (encrypted, decrypted) messages. Having defined our problem, we'll feed the `encrypted` message as the input to our LSTM and expect it to emit the `decrypted` message as the target. Sounds simple right?
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

<script src="https://gist.github.com/nikhilweee/13243631f8ed219167ccd3866ce3204e.js"></script>

To check the implementation, you can confirm that `encrypt('ABCDEFGHIJKLMNOPQRSTUVWXYZ-')` returns `NOPQRSTUVWXYZ-ABCDEFGHIJKLM`.

## Bring on them Minibatches
Okay, let's finally build the dataset. 


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


