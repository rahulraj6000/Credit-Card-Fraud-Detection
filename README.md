# Credit-Card-Fraud-Detection



PROJECT IMPLEMENTATION:

The estimation/modeling technique(s)/approach used to arrive at the solution/equation:
Exploratory data analysis
Feature Transformation
Outlier Removal
Dimensionality Reduction
Creation of new interactive features
XGBoost, RandomForest
Model Tuning and Optimization

the strategy employed:
Replacement of missing values with mean and median but got a satisfactory result with a median.
Identified outliers with quantile and mean and removed them.
Performed random under-sampling and smote due to highly imbalanced dataset. (Smote provided better result)
Implemented Log-Transform and Box-Cox Transform to remove skewness in the variables.
Prevented overfitting by hyperparameter (max_depth, min_child_weight and gamma) tuning.
Built XGBoost and RandomForest model to train dataset. (XGBoost performed better)
Used f1-score and roc_auc score for performance evaluation.

details of each variable used in the final logic:

Ordinal Encoding of a categorical variable in acq_sub_chn (channel acquisition) feature.
Removed features like referrals, spillover, min_pay_ind etc. because of very low variance.
Merged labels of categorical variables (acq_type_grp, fee_type_grp, acq_sub_chn) with a similar response rate to reduce label counts.
As there were lots of missing values in columns like referrals, sum_tot_line_amt, min_pay_ind, sow_tot_revol_bal_amt , we applied imputer(mean/median)

Reasons for Technique(s) Used:

A dataset was highly imbalanced, we used SMOTE technique to generate new samples that are coherent with the minor class distribution.
Removed extreme outliers from features to prevent skewness of existing statistical relationship.
Due to the presence of irrelevant features, Dimensionality reduction technique was used to avoid overfitting and redundancy.
Since there were lots of missing values and a combination of categorical and numerical features, we chose to implement RandomForest and XGBoost classifier.
Finally, we opted for XGBoost as XGBoost(with smote) performed better than RandomForest after tuning the hyperparameters. It increases the roc_auc score by 2%.









![histo](https://user-images.githubusercontent.com/34141117/94197011-ccc0dc00-fed2-11ea-849b-b7dcec5ea706.png)

![distribution_of_time_feature](https://user-images.githubusercontent.com/34141117/94196699-62a83700-fed2-11ea-9c03-4216c573c157.png)

![FraudAndNonFraud](https://user-images.githubusercontent.com/34141117/94196705-650a9100-fed2-11ea-9df5-b56f78d8c70c.png)

![Uploading CorrelationMatrix.png…]()



![correlationMatrix](https://user-images.githubusercontent.com/34141117/94196804-85d2e680-fed2-11ea-8e56-5316ef934131.png)

![corrlatiommatrix](https://user-images.githubusercontent.com/34141117/94196760-781d6100-fed2-11ea-97cb-72d838c42cdf.png)
