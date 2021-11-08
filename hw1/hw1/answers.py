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
**Yes** - increasing k lead to improved generalization for unseen data.
Having a very small k (e.g. 1) makes each prediction very sensitive to variations in the train data (or in other words - overfit).
This is because each prediction depends on exactly one point in the training set.  In such case, outliers will have
unfiltered impact on prediction.    On the other extreme, if k = number of points in train data, than all predictions will be the same -
The most common class.    There is no one k fits all - k must be optimized based on the specific dataset properties an tested on
a validation set to assess generalization error.  
Generally - "noisy" data will require larger k values, while very neat data will work better with smaller values. 
"""

part2_q2 = r"""
**1. Select model based on Training-set accuracy** - This is a bad idea, because it will lead to overfit.
The model that has the best Training-set accuracy is the model that managed to fit the best to the Train data.
E.g. I can build a model that predicts a person's height using only one attribute in the dataset - the person's ID number.
This model will achieve 100% train accuracy, but will obviously generalize terribly on unseen data. 
A model must be evaluated based on unseen data and not on the train data. 

**2. Select model based on Test-set accuracy** - There are two problems with this approach:
 a. Using the test data for model selection / calibration will leave us with no unseen data for generalization error assessment.
    If we use the same dataset for both, the final model assessment will not provide a good assessment for the generalization error, 
    because the model is already calibrated for this dataset specifically.    For this reason, we need to keep some unseen data for 
    the final generalization assessment.
 b. As we can see in the above plot, different train/validation datasets give different results for the same model hyperparameters (in this case k)
    Running the model in kfolds method provides some more generalization by running several (k) times per model option and 
    averaging the results to get a more general result.  
        
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
