---
title: Recommender systems for ML interviews (WIP)
blog_type: ml_notes
excerpt: Notes on recommender systems to present in ML interviews.
layout: post_with_toc_lvl3
last_modified_at: 2023-02-12
---

<hr/>

### 1. Links

1. [Tensorflow recommendation systems YT playlist](https://www.youtube.com/playlist?list=PLQY2H8rRoyvy2MiyUBz5RWZr5MPFkV3qz)
2. Wide and deep blog: [blog](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html)
3. Youtube paper on deep recommendations: [paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf)
4. Netflix recommendation case study: [paper](https://ojs.aaai.org/index.php/aimagazine/article/view/18140/18876)

<hr/>

### 2. Notes from TF Recommendation YT playlist
Recommender system: Connect one item with another. Simplest example would be to
match a user with a movie they may perhaps be interested in watching next.


#### Challenges

* High cardinality sparse features (very few examples for cross features)
* Usually conflicting objectives that are difficult to filter down to one
* Long-term effects are challenging to elicit from A/B tests
* Biased data. The videos shown to a user influence the videos seen by a user. Also
  imagine we only show highly rated videos and use the data to train a model. Predicting
  on low-rated reviews would be out of domain extrapolation which can yield low-fidelity
  results. We might have to deliberately display low probability of click videos to
  collect diverse data.
* **Scale** \
  Usually broken down into stages:
  1. Retrieval: Fast retrieval from O(millions) -> O(thousands)
  2. Ranking: O(1000) -> O(100)
  3. Post-Ranking and Ordering: O(100) -> O(10)

#### Solutions
1. For retrieval, it's usually some sort of Approximate Nearest Neighbours technique
   based on query and candidate embeddings which are usually made to be in the same
   embedding space. TF provides for ScaNN (Scalable Nearest Neighbours).
2. For privacy and speed, on-device smaller models are also common.


<hr/>

### 3. General notes

#### Dimension 1: Architecture
1. Collaborative filtering and matrix/tensor factorisation
2. Model as probability of interaction (click)\
   For probability of interaction we can build this as any binary classification problem
    where for every user and a preselected set of items, we predict the probability of
    click. The preselection comes from L1-1 ranking (recall phase).
3. Model as pairwise ranking (compare alternatives and make the model select one)

#### Dimension 2: Type of model
Modelling as probability of click (or pairwise ranking) can be done through
any classification model such as logistic regression, xgboost or neural nets.
Neural net approaches
* **Wide and deep:** Wide has sparse cross features which are memorised (overfit) and then
  the deep is a sequence of MLP layers or other carefully crafted layers that let the
  model generalise better. The wide part is for exceptions and deep part is for
  generalisations.
* **Two tower model:** One tower for user embeddings and meta features and another
  tower for the item. The final dimensions of the two towers match, we simply
  dot and take the sigmoid to compute the probability.
* **Sequence model:** User actions are first classified into different buckets. Each
  of these gets an action embedding. Users have a general user embedding. We then concat
  these and learn a sequence model such as an LSTM or transformers where we predict the
  action in the next time step. This apparently was a game-changer for Netflix.
