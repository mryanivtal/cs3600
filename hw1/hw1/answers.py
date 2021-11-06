r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
1. **False** - It enables estimation of the out of sample error.   Since the model is trained on the train data and may 
   be overfitted, we need a separate dataset that the model has not seen during train to understand how it behaves on 
   data that comes from the same distribution but has not been trained on.

2. **False** - The train and test datasets should come from the same distribution, the best way to get this is by random 
    sampling from the full dataset.  Otherwise, if the original dataset is sorted by label, timestamp or otherwise, we
    might get data that represents different distributions.

3. **True** - The test set should be used only to test the model performance, and not for any decision regarding the model 
    creation and calibration, e.g. cross-validation, hyperparameter calibration etc.

4. **False** - Since the cross-validation was done based on the validation set, it would not make sense to use the same 
    data to choose.   This should be done using the test set, which is external to the cross-validation process and best
    predicts the model generalization quality.
"""

part1_q2 = r"""
**This is a wrong approach.**     
As mentioned above, teh test set should be used only to measure the generalization error of the model on unseen data and 
evaluate its performance.   Since in this case the test data was used to tune a hyperparameter, the model is already 
biased towards this data and this could lead to unrealistic lower error estimation of the model.
The right way here is tho split the data to three separate parts - Train, Validation, Test. 
Then, use the train for training, the Validation for hyperparameter tuning and the test for performance test of the final model.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
"""

part2_q2 = r"""
**Your answer:**
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

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**
"""

part4_q2 = r"""
**Your answer:**
"""

part4_q3 = r"""
**Your answer:**
"""

# ==============
