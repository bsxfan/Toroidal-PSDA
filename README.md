# Toroidal PSDA
A probabilistic scoring backend for length-normalized embeddings. 

*Toroidal* PSDA is a generalization of the original PSDA model that was published in our Interspeech 2022 paper:
> [Probabilistic Spherical Discriminant Analysis: An Alternative to PLDA for length-normalized embeddings](https://arxiv.org/abs/2203.14893)

We now refer to the original PSDA as *Simple* PSDA. A paper describing the new model has not been published yet, but we will link that paper here when it becomes available. 

This repo supercedes the [original PSDA repo](https://github.com/bsxfan/PSDA) and it contains an updated version of the Simple PLDA implementation, as well as the new Toroidal PSDA implementation.

Probabilistic _Linear_ Discrimnant Analysys (PLDA) is a trainable scoring backend that can be used for things like speaker/face recognition or clustering, or speaker diarization. PLDA uses the self-conjugacy of multivariate Gaussians to obtain closed-form scoring and closed-form EM updates for learning. Some of the Gaussian assumptions of the PLDA model are violated when embeddings are length-normalized.

With PSDA, we use [Von Mises-Fisher](https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution) (VMF) instead of Gaussians, because they may give a better model for this kind of data. The VMF is also self-conjugate, so we enjoy the same benefits of closed-form scoring and EM-learning.



## Install
Dependencies are numpy, scipy and [PYLLR](https://github.com/bsxfan/PYLLR).

To install, put the root (the folder that contains the package tpsda) on your python path.

## Demo

- A working demo is here:
<https://github.com/bsxfan/Toroidal-PSDA/blob/main/tpsda/toroidal/toroid_vs_cosred.py>.
It can be run as a script. It makes synthetic data and demonstrates training and scoring.

- Further insight into the model and the training em-algorithm can be gained by running this demo script:
<https://github.com/bsxfan/Toroidal-PSDA/blob/main/tpsda/toroidal/test_em.py>.
It plots low-dimensional data on an interactive rotatable globe (if your plotting backend allows). 
