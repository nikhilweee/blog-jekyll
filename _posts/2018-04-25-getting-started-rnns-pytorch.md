---
layout: post
title: Building your first RNN
subtitle: with PyTorch 0.4!
tags: [rnn, pytorch]
category: [learn]
published: false
mathjax: true
---
If you have some understanding of recurrent networks, want to get your hands dirty, but haven't really tried to do that on your own, then you are certainly at the right place. This tutorial is a practical guide about getting started with recurrent networks using PyTorch. We'll solve a simple cipher using PyTorch 0.4.0, which is the latest version at the time of this writing.  

You are only expected to have some understanding of recurrent networks. If you don't, here's the link to the [golden resource](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) - Chris Olah's post on Understanding LSTMs. We'll use a single layer LSTM for the task of learning ciphers, which should be a fairly easy exercise.

## The Problem

Before starting off, let's first define the problem in a concrete manner. We wish to decrypt secret messages using an LSTM. For the sake of simplicity, let's assume that our messages are encrypted using the [Caesar Cipher](https://en.wikipedia.org/wiki/Caesar_cipher), which is a really simple substitution cipher.  

Caesar cipher works by replacing each letter of the original message by another letter from a given alphabet to form an encrypted message. In this tutorial we'll use a right shift of 13, which basically means that the encrypted version of each letter in the alphabet is the one which occurs 13 places to the right of it. So `A`(1) becomes `N`(1+13), `B`(2) becomes `O`(2+13), and so on. Our alphabet will include uppercase English characters `A` through `Z`, and an extra letter, `-`, to represent any foreign character.

With all of these in mind, here's the substitution table for your reference. The first row shows all the letters of the alphabet in order. To encrypt a message, each letter of the first row can be substituted by the corresponding letter from the second row. As an example, the message `THIS-IS-A-SECRET` becomes `FUVEMVEMNMERPDRF` when encrypted.

```
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z -
N O P Q R S T U V W X Y Z - A B C D E F G H I J K L M
```

**But why Neural Networks?**  
You might be wondering why do we use neural networks in the first place. In our use case, it sure makes more sense to decrypt the messages by conventional programming because we _know_ the encryption function beforehand. _This may not be the case everytime_. We might have enough data but still have no idea about the encryption function. Neural networks fit quite well in such a situation. Another simple reason to choose this problem is that we could generate loads of training examples on the fly. So we don't really need to procure any dataset. Yay!

## The Dataset

Talking about data, we'll use a parallel dataset of the following form where each tuple represents a pair of (encrypted, decrypted) messages. Having defined our problem, we'll feed the `encrypted` message as the input to our LSTM and expect it to emit the original message as the target. Sounds simple right?
```
('FUVEMVEMNMERPDRF', 'THIS-IS-A-SECRET')
('FUVEMVEMN-AFURDMERPDRF', 'THIS-IS-ANOTHER-SECRET')
...
```

It does, except that we have a little problem. Neural networks are essentially number crunching machines, and have no idea how to hande our encrypted messages. We'll somehow have to convert our strings into numbers for the network to make sense of them.

## Word Embeddings
The way this is usually done is to use something called as word embeddings. The idea is to represent every character in the alphabet with its own $$ D $$ dimensional **embedding vector**, where $$ D $$ is usually called the embedding dimension. So let's say if we decide to use an `embedding_dim` of 5, this basically means that each of the 27 characters of the alphabet, `ABCDEFGHIJKLMNOPQRSTUVWXYZ-`, will have their own embedding vector of length 5.

Often these vectors are stored together as $$ V \times D $$ dimensional **embedding matrix** $$ E $$, where each row $$ E[i] $$ of the matrix represents the embedding vector for the character with index $$ i $$ in the alphabet. Here $$ V $$ is the length of the vocabulary (alphabet), which is 27 in our case. As an example, the whole embedding matrix $$ E $$ might look something like the one shown below. $$ E[0] $$ then represents the word vector for `A`, which is `[-1.4107, -0.8142,  0.8486,  2.8257, -0.7130]`.


```
[[-1.4107, -0.8142,  0.8486,  2.8257, -0.7130],
 [ 0.5434,  3.8553,  2.9420, -2.8364, -4.0077], 
 [ 1.6781, -0.2496,  2.5569, -0.2952, -2.2911],
 ...
 [ 2.7912,  1.3261,  1.7603,  3.3852, -2.1643]]
```

Strictly speaking, what I just described here is called a _character embedding_, beacause we have a vector for each _character_ in the alphabet. In case we had a vector for each _word_ in a vocabulary, we would be using _word embeddings_ instead. Notice the analogy here. An alphabet is the set of all the letters in a language. Similarly, a vocabulary is the collection of all the words in a language. I'll be using alphabet and vocabulary interchangably throughout the tutorial. Similarly, word embeddings, word vectors, character embeddings, or simply embeddings will mean the same thing.

## The Cipher
Now that we have enough background, let's get our hands dirty and finally jump in to writing some code. The first thing we have to do is to create a dataset. And to do that, we first need to implement the cipher. Although we implement it as a simple function, it might be a good idea to implement the cipher as a class in the future.

{% gist 13243631f8ed219167ccd3866ce3204e cipher.py %}

We create the `encode` function which uses the parameters `vocab` and `key` to encrypt each character. Since we're working with letters, `vocab` in this context simply means the alphabet.  The encryption algorithm should be fairly easy to understand. Notice how we use the modulo operator in line `8` to prevent the indexes from overflowing.

To check the implementation, you can check for some random inputs. For example, ensure that `encrypt('ABCDEFGHIJKLMNOPQRSTUVWXYZ-')` returns `NOPQRSTUVWXYZ-ABCDEFGHIJKLM`.

## The Dataset (Again!)
Okay, let's finally build the dataset. For the sake of simplicity, we'll use a random sequence of characters as a message and encrypt it to create the input to the LSTM. To implement this, we create a simple function called `dataset` which takes in the parameter `num_examples` and returns a list of those many (input, output) pairs.

{% gist 13243631f8ed219167ccd3866ce3204e batch.py %}

There's something strange about this function though. Have a look at line 24. We're not returning a pair of strings. We're first converting strings into a list of indices which represent their position in the alphabet. If you recall the section on [word embeddings](#word-embeddings), these indices will later be used to extract the corresponding embedding vectors from the embedding matrix $$ E $$. We're then converting these lists into a pair of tensors, which is what the function returns.

**Tensors?**  
These are just some inbuilt data structures in PyTorch to speed up numerical computations. For users familiar with NumPy, a tensor is the PyTorch analogue of `ndarray`. If you're not, a tensor is essentially a multidimensional matrix which supports optimized implementations of common operations. Have a look at the [Tensor Tutorial](http://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html) on the PyTorch website for more information. The takeaway here is that we'll use tensors from now on as our go to data structure to handle numbers. Creating a tensor is really easy. Though there are a lot of ways to do so, we'll just wrap our list of integers with `torch.tensor()` - which turns out the easiest amongst all.

For now, you can satisfy yourself by having a look at what this function does. A quick call to `dataset(1)` should return something similar to the following. You can also verify that the numbers in the second tensor are right shifted by 13 from the numbers in the first tensor. `20 = (7 + 13) % 27`, `3 = (17 + 13) % 27` and so on.

```python
[[tensor([ 20,   3,  21,   0,  14,   4,   2,   4,  13,  12,   8,  23,
         12,  10,  25,  17,  19,   1,   2,  22,  12,  15,  16,   3,
         13,  10,  20,  23,  25,  15,  19,   4]), 
  tensor([  7,  17,   8,  14,   1,  18,  16,  18,   0,  26,  22,  10,
         26,  24,  12,   4,   6,  15,  16,   9,  26,   2,   3,  17,
          0,  24,   7,  10,  12,   2,   6,  18])]]
```

# Let's get building!

Let's finally start building our model! Let's first have a general overview of what we aim to achieve. One might think of something along the following lines.

>On a very high level, the first step in a general workflow will be to feed in inputs to an LSTM to get the predictions. Next, we pass on the predictions along with the targets to the loss function to calculate the loss. Finally, we backpropagate through the loss to update our model's parameters.

Hmm, that sounds easy, right? But how do you actually make it work? Let's dissect the problem step by step. We'll first identify the components needed to build our model, and finally put them to gether as a single piece to make it work.

>... **feed in inputs** to an LSTM to get the predictions ...

To feed in inputs, well, we first need to prepare the inputs. Remember the embedding matrix $$ E $$ we described earlier? we'll use $$ E $$ to convert the pair of indices we get from `dataset()` into the corresponding embedding vectors. Fortunately PyTorch provides a convenient way to do so. We just have to create an instance of `torch.nn.Embedding`.

The [docs](https://pytorch.org/docs/stable/nn.html#torch.nn.Embedding) list two required parameters - `num_embeddings: the size of the dictionary of embeddings` and `embedding_dim: the size of each embedding vector`. In our case, these are `vocab_size` $$ V $$ and `embedding_dim` $$ D $$ respectively.

```python
embed = torch.nn.Embedding(vocab_size, embedding_dim)
```

Later on, we could easily convert any input tensor `ecrypted` containing indices of the encrypted input (like the one we get from `dataset()`) into the corresponding embedding vectors by simply calling `embed(encrypted)`. As an example, if `encrypted` is `torch.tensor([ 20,   0,  21,   0])`, then `embed(encrypted)` would return something similar to the following.

```python
>>> encrypted = torch.tensor([ 20,   0,  21,   0])
>>> print(embed(encrypted))
tensor([[ 0.4344,  0.3980,  1.6350,  2.7416, -1.5250],
        [ 1.0940,  1.0992, -2.0170,  0.1433,  2.8728],
        [-0.4051,  0.0461,  0.3688,  1.0386,  0.2903],
        [ 1.0940,  1.0992, -2.0170,  0.1433,  2.8728]])
```

>... feed in inputs **to an LSTM** to get the predictions ...

Next, we need to create an LSTM. In PyTorch, we do this in a similar fashion by creating an instance of `torch.nn.LSTM`. This time, the [docs](https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM) list the required parameters as `input_size: the number of expected features in the input` and `hidden_size: the number of features in the hidden state`. Since LSTMs typically operate on variable length sequences, the `input_size` refers to the size of each entity in the input sequence. In our case, this means the `embedding_dim`. This might sound counter-intuitive, but if you think for a while, it makes sense. `hidden_size`, as the name suggests, is the size of the hidden state of the LSTM. Note that the hidden size can be different from the input size. [colah's blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) doesn't explicitly mention this, but the [docs on LSTMCell](https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell) should make it clear.


Once we have prepared the inputs, we will now pass them , 

>... feed in inputs to an LSTM to get the predictions ...



We'll define something called an **Embedding Matrix**, $$E \in \mathbb{R}^{D \times V}$$. This is a $$ D \times V $$ dimensional matrix where $$ D $$ is the `embedding_dim` and $$ V $$ is the length of the vocabulary. Each column $$ E[:, i] $$ of this matrix is the word vector for the 



 (27 in our case). We'll arbitrarily choose $$ D $$ as 5.
 




 PyTorch makes it really easy to do so. You simply create an instance of `torch.nn.LSTM` as shown in line `6`. Out of the many possible parameters listed in the [docs for nn.LSTM](http://pytorch.org/docs/stable/nn.html#torch.nn.LSTM), there are two required ones - the `input_size` and `hidden_size`. 

* `input_size: The number of expected features in the input 'x'`  
   The `input_size` here refers to the size of each entity in the `input_sequence`. Since we're using an `embedding_dim` of 5, each character will be represented by a vector of size 5, which is the `input_size`

* `hidden_size: The number of features in the hidden state 'h'`  
   This basically asks for the size of the hidden vector. In the context of an LSTM, this refers to the size of both, the cell state `c` and the hidden state `h`. This also means that both the states have to be of the same size. In this tutorial, we're using a `hidden_size` of 10.

The general idea of working with PyTorch is a two step process. 

1. Initialize the corresponding class with some important parameters.
2. Call the class object with the input tensor as arguments. 


Notice that I have set them to `embedding_dim` and `hidden_dim` respectively. 

{% gist 13243631f8ed219167ccd3866ce3204e model.py %}

Let's say you want to feed in an encrypted string into the LSTM to see what it predicts. In such a case, our input could be something like `FUVEMVEMNMERPDRF` and the corresponding target could be `THIS-IS-A-SECRET`. Since we're using character embeddings, our _input sequence_ is a vector containing the corresponding character embedding for each character in our input `FUVEMVEMNMERPDRF`.

In PyTorch, an LSTM is initialized by creating an instance of `torch.nn.LSTM` as shown. The corresponding arguments are explained below.

```
lstm = torch.nn.LSTM(input_size=5, hidden_size=10, num_layers=1)
```

* `input_size: The number of expected features in the input 'x'`  
   The `input_size` here refers to the size of each entity in the `input_sequence`. Since we're using an `embedding_dim` of 5, each character will be represented by a vector of size 5, which is the `input_size`

* `hidden_size: The number of features in the hidden state 'h'`  
   This basically asks for the size of the hidden vector. In the context of an LSTM, this refers to the size of both, the cell state `c` and the hidden state `h`. This also means that both the states have to be of the same size. In this tutorial, we're using a `hidden_size` of 10.

* `num_layers: Number of recurrent layers.`  
   Using an number greater than 1 basically means we're using a stacked LSTM, where the output of each layer is the input for the next layer, and the final output is that from the last layer.


## Guide to Mini Batching in PyTorch


```
graph TD
y_i["y_{i} output"]
y_ii["y_{i-1} prev_output"]
s_i["s_{i} decoder_hidden"]
c_i["c_{i} context"]
s_ii["s_{i-1} prev_decoder"]
h_j["h_{j} encoder_hidden"]
alpha_ij["alpha_{ij} attn_weights"]
e_ij["e_{ij} attn_energies"]
subgraph 
    a[feedforward]
end
subgraph 
    softmax[softmax]
end
subgraph 
    wsum[weighted_sum]
end
subgraph 
    rnn[RNN]
end
subgraph 
    out[feedforward]
end

s_ii --> a
h_j --> a
a --> e_ij
e_ij --> softmax
softmax --> alpha_ij
alpha_ij --> wsum
h_j --> wsum
wsum --> c_i
y_ii --> rnn
s_ii --> rnn
c_i --> rnn
rnn --> s_i
y_ii --> out
s_i --> out
c_i --> out
out --> y_i
```