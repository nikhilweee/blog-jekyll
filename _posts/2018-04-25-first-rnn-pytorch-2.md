---
layout: post
title: Building your first RNN
subtitle: with PyTorch 0.4! (2/2)
tags: [rnn, pytorch]
category: [learn]
published: false
mathjax: true
---
> This is part two of a two-part series on getting started with RNNs using PyTorch. Part one can be accessed at [Building your first RNN - Part 1]({% post_url 2018-04-25-first-rnn-pytorch-1 %})

Having described the problem and built the dataset in [Part 1]({% post_url 2018-04-25-first-rnn-pytorch-1 %}), let's finally start building our model. It's a good idea to first have a general overview of what we aim to achieve. One might think of something along the following lines.

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

This two-stepped process will be seen all through this tutorial and elsewhere.

</div>

Getting back to code now, let's dissect our 'high level' understanding again.

## 1. Prepare inputs

>... **feed in inputs** to an LSTM to get the predictions ...

To feed in inputs, well, we first need to prepare the inputs. Remember the embedding matrix $$ E $$ we described earlier? we'll use $$ E $$ to convert the pair of indices we get from `dataset()` into the corresponding embedding vectors. Following the general paradigm, we create an instance of `torch.nn.Embedding`.

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

`hidden_size`, as the name suggests, is the size of the hidden state of the LSTM. Note that the hidden size _can be different_ from the input size. [colah's blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) doesn't explicitly mention this, but the [docs on LSTMCell](https://pytorch.org/docs/stable/nn.html#torch.nn.LSTMCell) should make it clear. To summarize the discussion above, here is how we instantiate the LSTM.

```python
# Step 1
lstm = torch.nn.LSTM(embedding_dim, hidden_dim)
```

<div class="note" markdown="1">

### A note on dimensionality

During step 2 of the [general paradigm](#the-pytorch-paradigm), `torch.nn.LSTM` expects the input to be a 3D input tensor of size `(seq_len, batch, embedding_dim)`, and returns an output tensor of the size `(seq_len, batch, hidden_dim)`. We'll only feed in one input at a time, so `batch` is always `1`. 

As an example, consider the input-output pair `('FUVEMVEMNMERPDRF', 'THIS-IS-A-SECRET')`. Using an `embedding_dim` of 5, the 16 letter long input, `FUVEMVEMNMERPDRF` is transformed to an input tensor of size `16 x 1 x 5`. If the `hidden_dim` is 10, the input is processed by the LSTM into an output tensor of size `16 x 1 x 10`.

</div>

Pay attention to the output size of the embedding layer.

```python
>>> embedded.size()
torch.Size([6, 5]) 
``` 

Notice that we have a problem here. For every character in the input, we expect the LSTM to output probabilities corresponding to every possible character in the vocabulary. This way we could simply choose the character with the highest probability to be the output corresponding to the given input. an output matrix of the size `(seq_len, batch, vocab_size)` in order to compute a softmax over all the possible characters in the vocabulary.
How do we correct this?



>... feed in inputs to an LSTM to **get the predictions** ...

Once we get the outputs from the LSTM, the next step is to simply find out the most probable character for every character in the input sequence. This is usually done by computing a softmax over the set of all possible characters in the vocabulary. But wait, there's a catch (again).

A simple solution is to use matrix multiply. Let's say the output from the LSTM is denoted by $$ x \in \mathbb{R}^H $$ where $$ H $$ is the `hidden_size`. The idea is to multiply $$ x $$ with another matrix $$ W $$ such that the resulting matrix $$ y = W \times x $$ is  If we matrix is $$ out $$, and the 


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