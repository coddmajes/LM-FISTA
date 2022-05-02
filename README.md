# LM-FISTA 

----
This repository reproduced many iterative algorithms in folded and unfolded ways.
e.g., [ISTA](#[1]), [FISTA](#[2]), [LISTA](#[3]), [LISTA(tied)](#[4]), [LAMP](#[4]), [LFISTA](#[5]),
[LISTA-CP](#[6]), [TiLISTA](#[7]), [ALISTA](#[7]), [oracle_ISTA](#[8]), [GLISTA](#[9]),
 
In addition, we proposed two strategies, tuned stepsize LISTA and learned momentum FISTA respectively.
More details can be seen in the paper [English](doc/thesis/Adaptive_Accelerations_for_Learning-based_Sparse_Coding.pdf)|[中文](doc/thesis/Research_on_Sparse_Coding-based_Face_Recognition_under_Complex_Environment.pdf).

We got some inspirations from [LISTA-CP](#[6])'s work, their repository can be found at [VITA-Group/LISTA-CPSS](https://github.com/VITA-Group/LISTA-CPSS.git).
To be specific, we used their [config.py](config.py), [tf.py](utils/tf.py), and their ways to show the results 
while training models. Their code is implemented using tensorflow 1.0 framework, our work
is in tensorflow 2.0. And, we found that their way to predict may seem "confusing".
So, we changed it. More details can be seen in our code.

@author: coddmajes@gmail.com

## Code guide

----

### 1. env requirement
```
python 3.6
```

### 2. env installation

-----
To run the whole project, you need to install required packages at first.
```
pip install -r requirements.txt
```

### 3. run algorithm demo (LISTA(tied) as an example)

```
python main.py --M 250 --N 500 --support 0.1 -l 0.4 -T 16 --tbs 64 --vbs 1000 --is_unshared True --numbers 16 \
--task_type str --net LISTA_en_de --scope LISTA_en_de --exp_id 0
```

> more details can be seen in [setting.py](setting.py) | [setting_f.py](setting_f.py)

### 4. dataset

> refers to [README](data/README.md)

### 5. results

> Latest results refer to [README](doc/results/README.md)

## References
[1] <span id="[1]"> Daubechies I ,  Defrise M ,  Mol C D . An iterative thresholding algorithm for linear inverse problems with a sparsity constraint[J].  2003. </span>

[2] <span id="[2]"> Bech A . A fast iterative shrinkage-thresholding algorithms for linear inverse problems[J]. SIAM J. Imaging Sciences, 2009, 2. </span>

[3] <span id="[3]"> Gregor K ,  Lecun Y . Learning Fast Approximations of Sparse Coding[C]// International Conference on International Conference on Machine Learning. Omnipress, 2010.</span>

[4] <span id="[4]"> Borgerding M, Schniter P. Onsager-corrected deep learning for sparse linear inverse problems[C]//2016 IEEE Global Conference on Signal and Information Processing (GlobalSIP). IEEE, 2016: 227-231.</span>

[5] <span id="[5]"> Moreau T ,  Bruna J . Understanding Neural Sparse Coding with Matrix Factorization[J].  2016.</span>

[6] <span id="[6]"> Chen X ,  Liu J ,  Wang Z , et al. Theoretical Linear Convergence of Unfolded ISTA and its Practical Weights and Thresholds[J].  2018.</span>

[7] <span id="[7]"> Liu J ,  Chen X ,  Wang Z , et al. ALISTA: ANALYTIC WEIGHTS ARE AS GOOD AS LEARNED WEIGHTS IN LISTA[C]// International Conference on Learning Representations (ICLR). 2019.</span>

[8] <span id="[8]"> Ablin P ,  Moreau T ,  Massias M , et al. Learning step sizes for unfolded sparse coding[J].  2019.</span>

[9] <span id="[9]"> Wu K, Guo Y, Li Z, et al. Sparse coding with gated learned ISTA[C]//International Conference on Learning Representations. 2019.</span>

## Citation
周筝. 复杂环境下基于稀疏编码的人脸识别方法研究[D]. 福州:福州大学, 2021.
