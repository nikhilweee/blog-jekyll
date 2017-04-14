---
layout: post
title: Visual Cube
subtitle: A visual explanation to popular algorithms for the rubik's cube
# bigimg:
# - "https://i.imgur.com/j5GcL6G.jpg" : "Dawn from Triund Hill"
# - "https://i.imgur.com/KRRbzAk.jpg" : "Sunset at the Triund Hill"
tags: []
category: []
published: false
---

The rubik's cube needs no introduction. I've personally seen these since my childhood but I was never quite interested in solving the cube. As a kid, I used to play around, knowing that this is impossible for me to solve. I gave up easily. This was nothing more than a fidgeting toy.

Only recently has this toy caught my interest, and it was a bit too late when I tried to learn how to actually solve a cube. Most of the tutorials on the interweb throw out algorithms at you, but without an intuitive understanding of how they actually work, simply memorizing them does no good.

Here's an attempt to simplify all those `FR'UDD'R`s. Feel free to drag the cube around for a complete view.

_A big thanks to [Gregor Adams](https://codepen.io/pixelass) for his beautiful model of the [RubiCSS cube](https://codepen.io/pixelass/pen/CsItl)._  
_All these algorithms can be seen on  [how-to-solve-a-rubix-cube.com](https://how-to-solve-a-rubix-cube.com/)_

## Correcting the edges

`F R D' R F F`

This might be the first "difficult" step for many newbies. Pay attention to the highlighted edge. Notice how this algorithm cleverly corrects the orientation of the red-green edge.

<object id="base" type="text/html" style="width: 100%; height: 500px;" data="/resources/cube/algorithms/frdrff"></object>

## The Top Cross

`F R U R' U' F`

Once you've made the second layer, these algorithms start to get a bit longer. And why not? These algorithms also have the additional responsibility to maintain the bottom two layers. Here's the popular `fruruf`.

There are a few interesting observations about this algorithm.  
Notice the "L" and "Rod" shapes on the top edge that are commonly talked about. Also notice how there's a counterclockwise follow up to every move. There's an `R'` for an `R`, an `F'` for an `F`, and a `U'` for a `U`. These are certainly necessary to maintain the bottom two layers. Also notice that the reverse order is  not `U'R'F'` for obvious reasons.
Notice how six iterations of the same algorithm bring you back to where you started. Also notice the "L" and "Rod" shapes formed on the top face of the cube.

<object id="base" type="text/html" style="width: 100%; height: 500px;" data="/resources/cube/algorithms/fruruf"></object>

## The Last Layer Edges

`R U R' U R U U R' U`

Swapping the edges on the last layer can be tricky to understand. Take a look below.
Notice how the consecutive `UU`s are important to bring the top edges back to position.
<object id="base" type="text/html" style="width: 100%; height: 500px;" data="/resources/cube/algorithms/rururuuru"></object>

## Orient the Last Layer Corners

`R' D' R D`

Swapping the edges on the last layer can be tricky to understand. Take a look below.
Notice how the orienttion of the four corners is changed after an even iteration.
<object id="base" type="text/html" style="width: 100%; height: 500px;" data="/resources/cube/algorithms/rdrd"></object>

## Position the Last Layer Corners

`U R U' L' U R' U' L`

Swapping the edges on the last layer can be tricky to understand. Take a look below.
Corner 1 always stays in the same place, and in the same orientation. Notice how the bottom layers and the top corners are gracefully in place.
Also note the positions of the four corners at the end of every iteration. The algorithm repeats itself after three iterations. The highlighted corners can only be in one of the three possibilities.

<object id="base" type="text/html" style="width: 100%; height: 500px;" data="/resources/cube/algorithms/urulurul"></object>
