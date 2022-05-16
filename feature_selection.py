import statsmodels.api as sm
import pandas as pd
from sklearn.linear_model import Lasso, LogisticRegression


class FeatureSelect:
    def __init__(self, data, y):
        self.data = data
        self.y = y
        
    def step_forward(self, keep, pred_type = 'classification', metric = "aic"):
        keep.append('const')
        self.data = sm.add_constant(self.data)
        base = self.data[keep]
        candidates = list(set(self.data.columns) - set(keep))
        if pred_type == 'classification':
            model = sm.Logit(self.y, base).fit()
            def reset_model(feats):
                return sm.Logit(self.y, feats)
        elif pred_type == 'regression':
            model = sm.OLS(self.y, base).fit()
            def reset_model(feats):
                return sm.OLS(self.y, feats)
        new_features = list(base.columns)
        if metric == "aic":
            current_ic = model.aic
        elif metric == "bic":
            current_ic = model.bic
        for i in candidates:
            new_features.append(i)
            new_model = reset_model(self.data[new_features])
            new_model = new_model.fit()
            if metric == "aic":
                new_ic = new_model.aic
            elif metric == "bic":
                new_ic = new_model.bic
            if new_ic < current_ic:
                current_ic = new_ic
            else:
                new_features.pop()
            
        self.data = self.data.drop('const', axis = 1)
        self.ic_cols = new_features
            
    def step_backward(self, keep, pred_type = 'classification', metric = "aic"):
        keep.append('const')
        self.data = sm.add_constant(self.data)
        base = self.data
        candidates = list(set(self.data.columns) - set(keep)) 
        if pred_type == 'classification':
            model = sm.Logit(self.y, base).fit()
            def reset_model(feats):
                return sm.Logit(self.y, feats)
        elif pred_type == 'regression':
            model = sm.OLS(self.y, base).fit()
            def reset_model(feats):
                return sm.OLS(self.y, feats)
        new_features = list(base.columns)
        if metric == "aic":
            current_ic = model.aic
        elif metric == "bic":
            current_ic = model.bic
        for i in candidates:
            new_features.remove(i)
            new_model = reset_model(self.data[new_features])
            new_model = new_model.fit()
            if metric == "aic":
                new_ic = new_model.aic
            elif metric == "bic":
                new_ic = new_model.bic
            if new_ic < current_ic:
                current_ic = new_ic
            else:
                new_features.append(i)
        self.data = self.data.drop('const', axis = 1)
        self.ic_cols = new_features
        
    def importance(self, model):
        model.fit(self.data, self.y)
        importances = model.feature_importances_
        impdf = pd.DataFrame({'cols': self.data.columns, 'importances': importances})
        chosen_features = list(impdf['cols'][impdf['importances'] > 0])
        self.impdf = impdf
        self.important_features = chosen_features
    
    def lasso(self, pred_type):
        if pred_type == 'regression':
            model = Lasso()
        elif pred_type == 'classification':
            model = LogisticRegression(penalty='l1', solver='liblinear')
        model.fit(self.data, self.y)
        coefficients = pd.DataFrame({'cols': self.data.columns, 'coefs': model.coef_})
        chosen_features = list(coefficients['cols'][abs(coefficients['coefs']) > 0])
        self.lasso_features = chosen_features
    
    def get_item(self, var):
        if var == "ic_cols":
            return self.ic_cols
        elif var == "data":
            return self.data
        elif var == "important_features":
            return self.important_features
        elif var == "impdf":
            return self.impdf
        elif var == "lasso_features":
            return self.lasso_features
        
