# Active Reward Learning and Iterative Trajectory Improvement from Comparative Language Feedback (Journal Paper Under Review)
**Authors:** [Eisuke Hirota^*](https://ei5uke.github.io/), [Zhaojing Yang^*](https://yang-zj1026.github.io), [Ayano Hiranaka](https://misoshiruseijin.github.io/), [Miru Jun](https://github.com/lemonlemonde), [Jeremy Tien](https://www.linkedin.com/in/jeremy-tien), [Stuart Russell](https://www.cs.berkeley.edu/~russell/), [Anca Dragan](https://people.eecs.berkeley.edu/~anca/), [Erdem Bıyık](https://ebiyik.github.io)

# Set-Up
First, let's create a virtual env using pyvenv or conda:
```bash
conda create -n lal python=3.8
conda activate lal
```

Next, we'll clone the repo and set our requirements up:
```bash
git clone https://github.com/USC-Lira/language_active_learning.git
cd language_active_learning/
pip install -r requirements.txt
```

# Run
There are 6 different methods to run:
1. Comparison (non-active, random queries)
2. Language (non-active, random queries)
3. ActiveLanguage (active, information gain, linear nn)
4. PurelyBayesian (active, information gain, Bayesian ML)
5. BALD (active, information gain, deep ensemble)
6. QbC (active, uncertainty, deep ensemble)

Methods 3-5 use some form of approximate Bayesian inference or variational inference. While this repo provides code for such methods, the paper only leverages Laplace approximation for speed purposes. The usage of other algorithms may consist of refactoring the code. Other algorithms consist of Metropolis-Hastings, Metropolis within Gibbs, and Expectation Propagation.

To run the code, we suggest using any of the slurm scripts like follows:
```bash
/home/user/language_active_learning$ sbatch scripts/nn_active_rs.slurm
```
