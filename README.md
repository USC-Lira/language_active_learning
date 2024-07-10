# language_active_learning
Active Preference-Based Reward Learning using Human Language Feedback

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
There are 4 different ways to run:
1. Sampling Language from Embedding Space + Sampling Reward Weights
2. Sampling Language from LLMs + Sampling Reward Weights
3. Sampling Language from the Dataset + Sampling Reward Weights
4. Randomly choose questions

Henceforth, for 1, we must choose two sampling methods: one for the embedding space, and another for the reward weight.
For 2 and 3, we only choose a sampling method for the reward weight.
For 4, no sampling method is chosen.

The sampling methods available are:
1. Metropolis-Hastings
2. Gibbs (Metropolis within Gibbs)
3. Laplace Approximation
4. Expectation Propagation

So, for example:
```bash
/home/user/language_active_learning$ python lal/pref_learning/active_pb.py --active=1 --reward=2 --lang=1
```
This would run 1 (Sampling Language from Embedding Space + Sampling Reward Weights) using 1 (Metropolis-Hastings) to sample language and 2 (Gibbs) to sample reward weights.

Another example, but for 2 (Sampling Language from LLMs + Sampling Reward Weights) or 3 (Sampling Language from the Dataset + Sampling Reward Weights):
```bash
/home/user/language_active_learning$ python lal/pref_learning/active_pb.py --active=2 --reward=4
```
In this case, we run 2 (Sampling Language from LLMs + Sampling Reward Weights) using 4 (Expectation Propagation) to sample reward weights. We do not call ```--lang``` because we use the LLM to sample language.

# Acknowledgement
Much of this source code comes from:
* https://github.com/USC-Lira/language-preference-learning
* https://github.com/Stanford-ILIAD/easy-active-learning