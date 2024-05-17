import shap
import matplotlib.pyplot as plt

def build_feature_importance_graphics(prediction_func, X_train, X_test, feature_names):
    fig = plt.figure()
    
    explainer = shap.Explainer(prediction_func, X_train)
    
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, feature_names=feature_names, show=False)

    return fig.canvas.tostring_rgb()