# Research on Sparse Coding-based Face Recognition under Complex Environment

+ Proposed 2 strategies for learned ISTA to improve the efficiency of image restoration.
+ Apply improved unrolled ISTA in sparse representation-based face recognition. 

## A short Abstract
Sparse coding is an important paradigm to represent visual information, however, many representation-based recognition tasks are still limited by sparse solvers due to their effective computational cost. In this paper, we study data-driven learning for iterative algorithms to accelerate the approximation of sparse solutions. We propose a strategy for sparse dictionary learning combined with automatic tuning of step sizes in the iterative soft thresholding algorithm (ISTA) via its unfolding into a computational graph. In order to make it practical in large-scale applications, we suggest learning the step sizes instead of tuning them. Furthermore, following the assumption of the existence of a data-dependent "optimum" in the acceleration technique implemented in the fast iterative soft thresholding algorithm (FISTA), we introduce a learned momentum scheme to achieve "optimal" gain. Extensive encoding and decoding results confirm our hypothesis and prove the effectiveness of our method.

## A short Introduction
Consider the constrained minimization problem for a smooth function 
$$
f:\mathbb{R}^{m} \rightarrow \mathbb{R}
$$

and a convex function 
$$
g:\mathbb{R}^{n} \rightarrow \mathbb{R}
$$
which may be non-differentiable:
$$
\arg \min _{x} f(x)+g(x)
$$

If 
$$
f(x) = \frac{1}{2} \| b -Ax \|_{2}^{2}, 
g(x) = \lambda \| x \|_{1}
$$
where 
$$
b \in  \mathbb{R}^{m},A \in  \mathbb{R}^{m\times n},m\ll n, x \in \mathbb{R}^{n}
$$
$ \lambda $  is a positive scalar that balances the strength between regularized least-squares and the regularization. This minimization problem is called the LASSO regression and is the central issue in this paper.

Iterative shrinkage thresholding algorithm(ISTA, \cite{daubechies2004iterative}), as an extension of the proximal gradient method, is one of the most well-known schemes to approach this issue. Later, by binding to it "Nesterov acceleration", a novel fast iterative shrinkage thresholding algorithm (FISTA, \cite{beck2009fast}) was proposed, and achieved improved complexity results of $O(\frac{1}{k^2})$ compared to $O(\frac{1}{k})$ for ISTA, where $k$ is the number of iterations. The approximate message passing (AMP, \cite{donoho2009message}) algorithm was applied to solve the problem. Other schemes \cite{afonso2010fast, li2009coordinate} also showed their effectiveness. 

However, the past decade has seen a raising general interest in implementing these iterative algorithms as DNNs, i.e., taking traditional iterative algorithms into data-driven learning mechanisms to obtain accelerated sparse solvers, which we focus on in this paper. LISTA \cite{gregor2010learning} shows a representational way of unfolded iterative algorithms as neural networks. It only needs few iterations to achieve comparative results with the baseline algorithm ISTA. Following LISTA and its usefulness and versatility, a variety of other algorithms, such as LISTA-based or its variants \cite{sprechmann2013efficient,hershey2014deep,sprechmann2015learning,borgerding2017amp,chen2018theoretical,ablin2019learning,ito2019trainable,aberdam2020ada}, have been developed. At the same time, many other extensions have been put forward to solve other minimization problems. For example, \cite{wang2016learning,xin2016maximal} researched the $\ell_0$-based sparse approximation and \cite{wang2016learningb} investigated the $\ell_\infty$-constrained representation by unrolling the alternating direction method of multipliers (ADMM) algorithm.