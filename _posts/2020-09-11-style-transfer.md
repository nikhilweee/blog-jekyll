---
layout: post
title: Intro to Style Transfer
subtitle: Image Style Transfer Using Convolutional Neural Networks
category: Papers
mathjax: true
---
## Image Style Transfer Using Convolutional Neural Networks

This paper introduced neural style transfer. It uses a VGGNet pre-trained on the ImageNet dataset for the purpose. The key idea is to be able to separate content and style from the representations of the network. Once that is done, a new image is synthesized from white noise where two different kind of losses are minimized - a style loss between the style image and the hybrid image, and a content loss between the content image and the hybrid image.

## Intuitions

Since the VGGNet was originally trained for object classification, the layers towards the end of the network should have enough information to be able to recognise the object while still being invariant to its lower level features like position, style, etc. Therefore, we use the higher layers for the purposes of extracting content from the image, and the lower layers for the purposes of capturing the style of the image. This intuition is formalised in the way the losses are calculated.

## Content Loss

The representation of every image $\vec{x}$ in each layer $l$ of a CNN can be encoded by the feature response $F^l$ to the layer. Therefore, for the hybrid image $\vec{x}$ to be able to match the content of the original image $\vec{p}$, their respective feature representations $F^l$ and $P^l$ should be similar. This gives rise to the content loss, which is simply the squared error loss between the two. Note that $F^l \in \mathbb{R}^{N_l \times M_l}$ where $F_{ij}^{l}$ is the activation of the $i$th filter at the $j$th position in layer $l$ with $N_l$ distinct features each of size $M_l$

$$ L_{content} (\vec{p}, \vec{x}, l) = \frac{1}{2} \sum_{i, j} (F_{ij}^l - P_{ij}^l) $$

## Style Loss

To capture the style of an image, we might just want to capture the feature responses from the lower layers of the image. But note that those layers also contain spatial information about the content of the image which are later used by the higher layers of the network. Therefore, there is a need to decouple that information from the style of the image. To do so, we use a matrix of feature correlations built on top of the feature responses of each layer in the CNN. The feature correlations are given by something called as the Gram matrix $G^l \in \mathbb{R}^{N_l \times N_l}$, where $G_{ij}^l$ is the inner product between the feature maps $i$ and $j$ in layer $l$. 

$$
G_{ij}^l = \sum_k F_{ik}^l F_{jk}^l
$$

If $\vec{a}$ and $\vec{x}$ are the style image and the hybrid image respectively, then just like the content loss, we want the gram matrix of the style image $A^l$ to be as close as possible to the gram matrix of the hybrid image $G^l$. The style loss $E_l$ for every layer $l$ of the network is then defined as the squared loss between the two gram matrices. The total style loss is the weighted sum of the individual layer losses $E_l$ where $w_l$ are the weighing factors described separately in the paper.

$$
L_{style} (\vec{a}, \vec{x}) = \sum_{l=0}^{L} w_l E_l
\hspace{2cm}
E_l = \frac{1}{4 N_l^2 M_l^2} \sum_{i,j} (G_{ij}^l - A_{ij}^l)^2
$$

The total loss $L_{total}$ is the weighted sum of the content and style losses with weights $\alpha$ and $\beta$ respectively.

$$
L_{total} (\vec{p}, \vec{a}, \vec{x}) = \alpha L_{content} (\vec{p}, \vec{x}) + \beta L_{style} (\vec{a}, \vec{x})
$$