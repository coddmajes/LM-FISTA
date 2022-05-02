# Test Results
> Code: [Simulation](plot_results.ipynb) | [Face](plot_results_face.ipynb)

## Algorithms:

> ISTA, FISTA, LISTA(shared), LISTA(tied), LISTA(unshared), LISTA-CP, TiLISTA, ALISTA, LAMP, LFISTA, GLISTA, 

## Group:

### (1) traditional iterative algorithms:

> ISTA, FISTA, 

### (2) weight shared, unshared, tied:

> LISTA(shared), LISTA(tied), LISTA(unshared), LAMP(shared), LAMP(tied),

- shared weights means that all weights are learned through all iterations：

  for example,  W=A^(T)/L as the initial value, after the first iteration, W ->W', for the second iteration, W' is the initial value;

- unshared weights means that each iteration has their independent trainable weights:

  for example, W=A^(T)/L as the initial value, after the first iteration, W ->W', for the second iteration, w is the initial value;

- tied weights means that matrix weights are shared and the rest are unshared:

  for example, W=A^(T)/L, theta=lambda/L, to be to be specific, W is shared and theta is unshared.

### (3) weight coupling & analytical weight

> LISTA-CP, TiLISTA, ALISTA, LISTA-CP-ss, TiLISTA-ss, ALISTA-ss

-ss means support selection scheme is added

### (4) accelerations or gain added:

> LFISTA, GLISTA

### (5) proposed algorithms:

> tuned step LISTA(TsLISTA), learned step LISTA(LsLISTA), learned momentum FISTA(LMFISTA)

TsLISTA and LsLISTA are improved by calculating or learning "optimal" step size; 

LMFISTA is a strategy to learn acceleration automatically by just adding one learnable parameter.


## Comparison:

### V1
> verification of different weight design: shared, unshared and tied: 

- V1-1, LISTA(shared), LISTA(tied), LISTA(unshared), LAMP(shared), LAMP(tied)
- V1-2, taking LISTA as an example to see if weights really need to be unshared.
  - for example, 1st and 2nd iterations are using shared weight W_1, and 3rd and 4th iterations are using shared weight W_2, the initial value for W_1 and W_2 is A^(T)/L; 
  - for a 16-layers network, using LISTA_1 to indicate unshared weights, LISTA_16 stands for tied weight, LISTA denotes LISTA with shared weights, LISTA_4 means each 4 iterations or layers are using shared matrix weight, e.g., 1st-4th layers are using W_1, 5th-8th layers are using W_2;
- more details can be seen in [English](/doc/thesis/Adaptive_Accelerations_for_Learning-based_Sparse_Coding.pdf)
  
### results:

#### V1-1:

![V1-1](images/v1-1.png)

| Models               | LISTA(shared)   | LISTA(unshared)   | LISTA(tied)     | LAMP(unshared) | LAMP(tied) |
| -------------------- | --------------- | ----------------- | --------------- | -------------- | ---------- |
| Trainable parameters | O(mn + n^2 + 1) | O(Kmn + Kn^2 + K) | O(mn + n^2 + K) | O(Kmn + K)     | O(mn + K)  |

K indicates the layers/iterations, m, n are the sensing Matrix A s dimensions, row and column respectively.

#### conclusion: 
Tied scheme shows the best performances, and has comparatively less trainable parameters.

#### V1-2:

![V1-2](images/v1-2.png)

| Models               | LISTA(shared)   | LISTA-1(unshared) | LISTA-2                | LISTA-4                | LISTA-8                | LISTA-16(tied)  |
| -------------------- | --------------- | ----------------- | ---------------------- | ---------------------- | ---------------------- | --------------- |
| Trainable parameters | O(mn + n^2 + 1) | O(Kmn + Kn^2 + K) | O((k/2)(mn + n^2) + K) | O((k/4)(mn + n^2) + K) | O((k/8)(mn + n^2) + K) | O(mn + n^2 + K) |

#### conclusion: 
Matrix weights to be unshared for each layer is unneccessary,
but threshold has to be unshared.

### <span id="V2"> V2 </span>

> verification of adding Nesterov's acceleration, LISTA(tied), LISTA-CP, ALISTA,and proposed LsLISTA; (-NA means Nesterov's acceleration is added)

- V2-1, LISTA(tied), LISTA(tied)-NA, LISTA-CP, LISTA-CP-NA, ALISTA, ALISTA-NA, LsLISTA, LsLISTA-NA, SNR=inf
- V2-2, LISTA(tied), LISTA(tied)-NA, LISTA-CP, LISTA-CP-NA, ALISTA, ALISTA-NA, LsLISTA, LsLISTA-NA, SNR=20

### results:

#### V2-1:

![V2-1](images/v2-1.png)

#### V1-2:

![V2-2](images/v2-2.png)

### conclusion:
After adding Nesterov's acceleration, proposed LsLISTA-NA shows the bese performance, 
however, ALISTA-NA became worth, shows that pre-calculated weights can not adapt 
to added momentum. ALISTA is less adaptive.

### V3

> validation of gamma, which is the learnable parameter for LMFISTA:

-  LM-LISTA-CP, LM-ALISTA, LM-FISTA, LM-LISTA-CPss, LM-ALISTA-ss, LM-FISTA-ss, r'$\frac{t_{(k)}-1}{t_{(k+1)}}$

### results:

![V3](images/v3-1.png)

### conclusion: 
- Learned momentum scheme added models (LM-LISTA-CP, LM-ALISTA, LM-FISTA) show 
their gamma learned to be the "optimal" just after 2 or 3 iterations;
however, the original Nesterov's acceleration is relatively slow;

- After add support selection, LM-LISTA-CPss and LM-ALISTA-ss show
their gamma to be worse, means that these two models (LM-LISTA-CP, LM-ALISTA)
with support selection maybe causes overfitting? LM-FISTA still shows 
learned a stable gamma after a few layers.

### V4

> comparison between traditional iterative algorithms and proposed unrolled algorithms:

- V4-1, ISTA, FISTA, TsLISTA, LsLISTA, LMFISTA SNR=inf
- V4-2, ISTA, FISTA, TsLISTA, LsLISTA, LMFISTA SNR=20
- V4-3, ISTA, FISTA, TsLISTA, LsLISTA, LMFISTA SNR=10

### results:

#### V4-1:

![V4-1](images/v4-1.png)

#### V4-2:

![V4-2](images/v4-2.png)

#### V4-3:

![V4-3](images/v4-3.png)

### conclusion:
Proposed unrolled algorithms (TsLISTA, LsLISTA, LMFISTA) outperform 
traditional iterative algorithms.

### V5

> comparison between benchmark algorithm LISTA(tied),  LAMP(tied), and proposed unrolled algorithms:

- V5-1, LISTA(tied), LAMP(tied), TsLISTA, LsLISTA, LMFISTA SNR=inf
- V5-2, LISTA(tied), LAMP(tied), TsLISTA, LsLISTA, LMFISTA SNR=20
- V5-3, LISTA(tied), LAMP(tied), TsLISTA, LsLISTA, LMFISTA SNR=10

### results:

#### V5-1:

![V5-1](images/v5-1.png)

#### V5-2:

![V5-2](images/v5-2.png)

#### V5-3:

![V5-3](images/v5-3.png)

| Models               | LISTA(tied)     | LAMP(tied) | TsLISTA    | LsLISTA   | LMFISTA       |
| -------------------- | --------------- | ---------- | ---------- | --------- |---------------|
| Trainable parameters | O(mn + n^2 + K) | O(mn + K)  | O(Kmn + K) | O(mn + K) | O(mn + K + 1) |

### conclusion:
Proposed LMFISTA outperforms LISTA(tied) and LAMP(tied).
Proposed LMFISTA only added one learnable parameter gamma compare to
proposed LsLISTA, but outperforms it.

### V6

> comparison between weight coupling & analytical weight and proposed unrolled algorithms:

- V6-1, LISTA-CP, TiLISTA, ALISTA, TsLISTA, LsLISTA, LMFISTA SNR=inf
- V6-2, LISTA-CP, TiLISTA, ALISTA, TsLISTA, LsLISTA, LMFISTA SNR=20
- V6-3, LISTA-CP-ss, TiLISTA-ss, ALISTA-ss, LsLISTA-ss, LMFISTA-ss SNR=inf
- V6-4, LISTA-CP-ss, TiLISTA-ss, ALISTA-ss, LsLISTA-ss, LMFISTA-ss SNR=20

### results:

#### V6-1:

![V6-1](images/v6-1.png)

#### V6-2:

![V6-2](images/v6-2.png)

#### conclusion:
Proposed LMFISTA outperforms other models.

#### V6-3:

![V6-3](images/v6-3.png)

#### V6-4:

![V6-4](images/v6-4.png)

| Models               | LISTA-CP    | TiLISTA       | ALISTA | TsLISTA     | LsLISTA   | LMFISTA   |
| -------------------- | ----------- | ------------- | ------ | ----------- | --------- | --------- |
| Trainable parameters | O(Kmn  + K) | O(mn + K + K) | O(K)   | O(Kmn  + K) | O(mn + K) | O(mn+K+1) |

#### conclusion:
After adding support selection, all algorithms show almost the same performance.
Except for ALISTA-ss, proposed LMFISTA has less trainable parameters than the others,
but ALISTA is hard to apply in real applications, less adaptive (also see in [V2](#V2))than LMFISTA.


### V7

> comparison between accelerations or gain added and proposed LMFISTA:

- V7-1, LFISTA, GLISTA, LMFISTA, SNR=inf
- V7-2, LFISTA, GLISTA, LMFISTA, SNR=20

### results:

#### V7-1:

![V7-1](images/v7-1.png)

#### V7-2:

![V7-2](images/v7-2.png)

| Models               | LFISTA                   | GLISTA-exp                       | LMFISTA   |
| -------------------- | ------------------------ | -------------------------------- | --------- |
| Trainable parameters | O(Kn^2 + Kn^2 + Kmn + K) | O(Kmn + Kn^2 + Kn^2 + K + K + K) | O(mn+K+1) |

### conclusion:
LFISTA, GLISTA and proposed LMFISTA are unrolled algorithms with accelerated schemes,
proposed LMFISTA outperforms the other two, and has the least parameters.
The rest two are too heavy in parameter setting.

### V8

> comparison between having learned momentum added and without:

- V8-1, LISTA-CP, LM-LISTA-CP, ALISTA, LM-ALISTA, LsLISTA, LMFISTA, SNR=inf
- V8-1, LISTA-CP, LM-LISTA-CP, ALISTA, LM-ALISTA, LsLISTA, LMFISTA, SNR=20
- V8-1, LISTA-CP, LM-LISTA-CP, ALISTA, LM-ALISTA, LsLISTA, LMFISTA, SNR=10

### results:

#### V8-1:

![V8-1](images/v8-1.png)

#### V8-2:

![V8-2](images/v8-2.png)

#### V8-3:

![V8-3](images/v8-3.png)

| Models               | LISTA-CP   | LM-LISTA-CP     | ALISTA | LM-ALISTA | LsLISTA | LMFISTA   |
| -------------------- | ---------- | --------------- | ------ | --------- | ------- | --------- |
| Trainable parameters | O(Kmn + K) | O(Kmn  + K + 1) | O(K)   | O(K + 1)  | O(mn+K) | O(mn+K+1) |

### conclusion:
After adding learned momentum scheme, which only need to add one parameter, 
these models (LM-LISTA-CP, LM-ALISTA, LMFISTA ) are show better results than 
original models ((LISTA-CP, ALISTA, LsLISTA).
