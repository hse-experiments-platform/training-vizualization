# %%
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer, r2_score
from sklearn.metrics import make_scorer, get_scorer_names
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from cv_graphics import create_boxplot


# %%
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
    
lasso = linear_model.Lasso()

scoring = {'MSE': make_scorer(mean_squared_error),
           'MAE': make_scorer(mean_absolute_error),
           'R2': make_scorer(r2_score)}

scores = cross_validate(lasso, X, y, cv=10,
                        scoring=scoring,
                        return_train_score=True)

create_boxplot(scores['test_MSE'])
plt.show()
# %%
