# Research on Sparse Coding-based Face Recognition under Complex Environment

+ Proposed 2 strategies for learned ISTA to improve the efficiency of image restoration.
+ Apply improved unrolled ISTA in sparse representation-based face recognition. 

## A short Abstract

Sparse coding is an important paradigm to represent visual information, however, many representation-based recognition tasks are still limited by sparse solvers due to their effective computational cost. In this paper, we study data-driven learning for iterative algorithms to accelerate the approximation of sparse solutions. We propose a strategy for sparse dictionary learning combined with automatic tuning of step sizes in the iterative soft thresholding algorithm (ISTA) via its unfolding into a computational graph. In order to make it practical in large-scale applications, we suggest learning the step sizes instead of tuning them. Furthermore, following the assumption of the existence of a data-dependent "optimum" in the acceleration technique implemented in the fast iterative soft thresholding algorithm (FISTA), we introduce a learned momentum scheme to achieve "optimal" gain. Extensive encoding and decoding results confirm our hypothesis and prove the effectiveness of our method.

## Proposed tuned steps LISTA
![tuned steps LISTA]()

## Proposed learned steps LISTA
![learned steps LISTA]()

## Proposed "learned momentum" FISTA
![learned momentum FISTA]()

