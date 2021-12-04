r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1. **The shape of the jacobian will be (128 x 1024 x 2048)** 
2.  **1 GB** = 1,073,741,824 byte = (1024 input features) x (2048 output features) x (128 vectors) x  (4 byte/element)
"""
# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    wstd, lr, reg = 0.01, 0.005, 0.0
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0, 0, 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = 0.01, 1e-05, 5e-06, 0, 5e-05
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # ====== YOUR CODE: ======
    wstd = 1e-1
    lr = 1e-3
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Not done - related to bonus question:**
"""

part2_q2 = r"""
**Yes, it is possible, since they measure different things**. <Br>
<Br>
Accuracy: $\dfrac{number of correct predictions}{Total number of predictions}$ <Br>
<Br>
Cross antropy: $\ell_{\mathrm{CE}}(\vec{y},\hat{\vec{y}}) = - {\vectr{y}} \log(\hat{\vec{y}})$ <Br>
<Br>
**Example for both accuracy and loss going up at the same time:** <Br>

**Scenario 1:**

|| Ground_truth class | P(a) | P(B) | P(C) | Cross antropy | Accuracy ||
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sample 1 | A | 0.7 | 0.15 | 0.15| 0.357 | 1 |
| Sample 2 | A | 0.7 | 0.15 | 0.15| 0.357 | 1 |
| Sample 3 | A | 0.33 | 0.33 | 0.34| 1.109 | 0 |
    
**Loss**: 1.82
**Accuracy**: 0.667

**Scenario 2:**

|| Ground_truth class | P(a) | P(B) | P(C) | Cross antropy | Accuracy ||
| --- | --- | --- | --- | --- | --- | --- | --- |
| Sample 1 | A | 0.5 | 0.25 | 0.25| 0.693 | 1 |
| Sample 2 | A | 0.5 | 0.25 | 0.25| 0.693 | 1 |
| Sample 3 | A | 0.5 | 0.25 | 0.25| 1.693 | 1 |
    
**Loss**: 2.07
**Accuracy**: 1.00
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

"""

part3_q2 = r"""
**Your answer:**

"""

part3_q3 = r"""
**Your answer:**

"""

part3_q4 = r"""
**Your answer:**

"""

part3_q5 = r"""
**Your answer:**

"""

part3_q6 = r"""
**Your answer:**

"""
# ==============
