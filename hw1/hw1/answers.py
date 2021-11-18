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
As mentioned above, the test set should be used only to measure the generalization error of the model on unseen data and 
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
The value of Delta is arbitrary because the W is scaled by the optimization mechanism, and can be scaled up and down to get closer to a minimum,
If delta is large the optimizer can compensate by adjusting the W values accordingly.
"""

part3_q2 = r"""
**1. Interpretation: The model learns geometric features in the image, and where they are on the x, y axis.**
The more differentiated (Noisy) areas look for details, while the more uniform areas can be thought of as "Masked", or less indicative.
E.g. image 2 from the left learns whether there are horizontal lines in the top, bottom, middle of the image.
Interpretation error example - In the 3rd line there is a four labeled by the model as a six, probably because it is missing the bottom 
part which has a strong impact on the score.
**2. Difference between KNN and SVM** - The SVM model is an actual learning model - it creates a model (weights) and applies the learned model to the samples in runtime
KNN works very differently - it memorizes the train samples and then compares each runtime sample to all train samples and look for the closest one in the attribute space.
Hence, the KNN does not have a "Feature extraction mechanism" like we see here.

"""

part3_q3 = r"""
**1. I'd say the learning rate is a bit higher than optimal** - We can see the loss is a bit jumpy and not smooth and stable as it could be.
Lowering it a bit woud probably give more stable behaviour.   Yet, it is stable enough to be effective and fast.
- Higher learning rate would cause the graph to jump up and down much stronger, Very high would make it not converge.
- Lower learning rate would produce smoother and more monotonic graph, but will converge slower.

**2. I'd say the model is very slightly underfitted** - a few more epochs would probably get to a better result, but not many.
This can be seen by the fact that the performance on the validation has almopst converged - accuracy is still raising but very lightly, 
loss is still reducing but very lightly as well.
The same is happening with the train data.
- Overfit would be indicated by reduction in the performance on the validation set (loss going up, accuracy going down)
- strong underfit would be indicated by strong improvement slopes in the validation set in epoch 30 (loss going down, accuracy up)
 
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**The ideal pattern to see in the residual plot is values equally and randomly spaced around the horizontal axis.
Based on the residual plot we got above we can see that generally our model fits the data well and looks similar in the train and test data except of sepcifc outliers.
We can see in the top-5 features plot that the residuals values are higher than our final plot after CV beacuse we have more errors there.**
"""

part4_q2 = r"""
  1.It is still linear model that finds linear function on the new features, but when returning back to the original features we are getting non linear function which maps them to the target.

  2. Yes, we should do approiate features engineering to represent our fetaures accordingly.
  
  3. It depends on the non linear features mapping we did, but generally speacking the bounday won't be hyperplane but a non linear decision boundary because we got non linear function.
"""

part4_q3 = r"""
1. Because we using lambda parameter in multiplications and it gives more reasonable values in the range to look for when using logspace. The advantage for CV is doing hyper paramters search without doing overfiiting on the validation set because we are changing the validatin set in each cyle. By that, we should get model that will better generalize and perform on unseen data.

2. len(degree_range) * len(lambda_range) = 60
"""

# ==============
