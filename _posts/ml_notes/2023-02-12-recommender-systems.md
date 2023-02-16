---
title: Recommender systems for ML interviews (WIP)
blog_type: ml_notes
excerpt: Notes on recommender systems to present in ML interviews.
layout: post_with_toc_lvl3
last_modified_at: 2023-02-12
---

### Dimension 1: Architecture
1. Collaborative filtering and matrix/tensor factorisation
2. Model as probability of interaction (click)\
   For probability of interaction we can build this as any binary classification problem
    where for every user and a preselected set of items, we predict the probability of
    click. The preselection comes from L1-1 ranking (recall phase).
3. Model as pairwise ranking (compare alternatives and make the model select one)

### Dimension 2: Type of model
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

### Links
1. Wide and deep blog: [blog](https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html)
2. Youtube paper on deep recommendations: [paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45530.pdf)
3. Netflix recommendation case study: [paper](https://ojs.aaai.org/index.php/aimagazine/article/view/18140/18876)


### Notes from papers

#### Deep Learning for Recommender Systems: A Netflix Case Study
