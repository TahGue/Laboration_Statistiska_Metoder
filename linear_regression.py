import numpy as np
from scipy import stats


class LinearRegression:

    def __init__(self, confidence_level=0.95, add_intercept=True, drop_first_category=True):
        
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")
        
        self.confidence_level = float(confidence_level)
        self.add_intercept = bool(add_intercept)
        self.drop_first_category = bool(drop_first_category)

        # Model parameters
        self.beta = None
        self._X = None
        self._y = None
        self._xtx_inv = None
        self._feature_names = None

        # Model dimensions
        self.n = None
        self.d = None

    @staticmethod
    def one_hot_encode(values, drop_first=True):
        
        v = np.asarray(values)
        categories = np.unique(v)
        if drop_first and categories.size > 0:
            categories = categories[1:]
        out = np.zeros((v.shape[0], categories.size), dtype=float)
        for j, cat in enumerate(categories):
            out[:, j] = (v == cat).astype(float)
        return out, categories

 
    @staticmethod
    def _as_2d(X):
        
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X    
    def _design_matrix(self, X):
       
        X = self._as_2d(X)
        if self.add_intercept:
            return np.column_stack([np.ones(X.shape[0], dtype=float), X])
        return X

    def fit(self, X, y, feature_names=None):
        
        y = np.asarray(y, dtype=float).reshape(-1)
        X = self._as_2d(X)
        
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        Xd = self._design_matrix(X)
        
        
        xtx = Xd.T @ Xd
        
        self._xtx_inv = np.linalg.pinv(xtx)
        
        self.beta = self._xtx_inv @ (Xd.T @ y)

        self._X = Xd
        self._y = y
        self._feature_names = feature_names

        self.n = int(Xd.shape[0])
        if self.add_intercept:
            self.d = int(Xd.shape[1] - 1)
        else:
            self.d = int(Xd.shape[1])
        
        return self

    def predict(self, X):
       
        if self.beta is None:
            raise ValueError("Model is not fitted. Call fit() before predict().")
        Xd = self._design_matrix(X)
        if Xd.shape[1] != self._X.shape[1]:
            raise ValueError(
                f"X has {Xd.shape[1]} features but model was fitted with {self._X.shape[1]}"
            )
        return Xd @ self.beta

    def residuals(self):
       
        if self.beta is None:
            raise ValueError("Model is not fitted")
        return self._y - (self._X @ self.beta)

    def sse(self):
        
        r = self.residuals()
        return float(r @ r)

    def variance(self):
        
        denom = self.n - self.d - 1
        if denom <= 0:
            raise ValueError(
                f"Not enough degrees of freedom to estimate variance. "
                f"Need n > d + 1, but n={self.n}, d={self.d}"
            )
        return self.sse() / denom

    def standard_deviation(self):
       
        return float(np.sqrt(self.variance()))

    def mse(self):
        
        return self.sse() / self.n

    def rmse(self):
       
        return float(np.sqrt(self.mse()))
    def syy(self):
        
        y = self._y
        mu = float(np.mean(y))
        dy = y - mu
        return float(dy @ dy)

    def ssr(self):
        
        return self.syy() - self.sse()

    def r2(self):
        
        s = self.syy()
        if s == 0.0:
            return float("nan")
        return self.ssr() / s
    def adjusted_r2(self):
        
        r2_val = self.r2()
        if np.isnan(r2_val):
            return float("nan")
        n = self.n
        d = self.d
        return 1.0 - (1.0 - r2_val) * (n - 1) / (n - d - 1)

    def f_test(self):
        
        if self.d <= 0:
            raise ValueError("F-test requires at least one feature (d > 0)")
        
        f_stat = (self.ssr() / self.d) / self.variance()
        df1 = self.d
        df2 = self.n - self.d - 1
        p_value = float(stats.f.sf(f_stat, df1, df2))
        
        return {
            "f_stat": float(f_stat),
            "df1": int(df1),
            "df2": int(df2),
            "p_value": p_value
        }

    def standard_errors(self):
       
        v = self.variance()
        diag = np.diag(self._xtx_inv)
        return np.sqrt(v * diag)

    def covariance_matrix(self):
       
        return self._xtx_inv * self.variance()
    def t_tests(self):
        
        se = self.standard_errors()
        df = self.n - self.d - 1
        t_stat = self.beta / se
        
        
        dist = stats.t(df)
        cdf = dist.cdf(t_stat)
        sf = dist.sf(t_stat)
        p_value = 2.0 * np.minimum(cdf, sf)
        
        return {
            "t_stat": t_stat,
            "df": int(df),
            "p_value": p_value
        }

    def confidence_intervals(self):
       
        se = self.standard_errors()
        df = self.n - self.d - 1
        alpha = 1.0 - self.confidence_level
        tcrit = float(stats.t(df).ppf(1.0 - alpha / 2.0))
        lower = self.beta - tcrit * se
        upper = self.beta + tcrit * se
        
        return {
            "confidence_level": float(self.confidence_level),
            "t_crit": tcrit,
            "lower": lower,
            "upper": upper
        }

    def pearson_pairs(self, X=None, include_intercept=False):
       
        if X is None:
            if self._X is None:
                raise ValueError("Model is not fitted and no X provided")
            X = self._X
        else:
            X = self._as_2d(X)
        
        # Remove intercept column if present and not requested
        if not include_intercept and X.shape[1] > 1:
            col0 = X[:, 0]
            if np.allclose(col0, 1.0):
                X = X[:, 1:]
        
        p = X.shape[1]
        r_mat = np.eye(p, dtype=float)
        p_mat = np.zeros((p, p), dtype=float)
        
        for i in range(p):
            for j in range(i + 1, p):
                r, pv = stats.pearsonr(X[:, i], X[:, j])
                r_mat[i, j] = r
                r_mat[j, i] = r
                p_mat[i, j] = pv
                p_mat[j, i] = pv
        
        return {"r": r_mat, "p_value": p_mat}

    def summary(self):
        """
        Generate a comprehensive summary of the regression results.
        
        Returns
        -------
        result : RegressionResults
            Object containing all key statistics with a pretty text representation.
        """
        if self.beta is None:
            raise ValueError("Model is not fitted")
        
        f_result = self.f_test()
        t_result = self.t_tests()
        ci_result = self.confidence_intervals()
        
        coef_names = []
        if self.add_intercept:
            coef_names.append("Intercept")
        if self._feature_names is not None:
            coef_names.extend(self._feature_names)
        else:
            for i in range(self.d):
                coef_names.append(f"X{i+1}")
        
        data = {
            "n_observations": self.n,
            "n_features": self.d,
            "coefficients": {
                "names": coef_names,
                "values": self.beta,
                "std_errors": self.standard_errors(),
                "t_stats": t_result["t_stat"],
                "p_values": t_result["p_value"],
                "ci_lower": ci_result["lower"],
                "ci_upper": ci_result["upper"]
            },
            "model_stats": {
                "r_squared": self.r2(),
                "adjusted_r_squared": self.adjusted_r2(),
                "rmse": self.rmse(),
                "residual_std_error": self.standard_deviation()
            },
            "f_test": f_result,
            "confidence_level": self.confidence_level
        }
        return RegressionResults(data)

class RegressionResults:
    """Helper class to display regression results nicely in notebooks without print()."""
    def __init__(self, data):
        self.data = data
        
    def __repr__(self):
        s = self.data
        m = s['model_stats']
        f = s['f_test']

        lines = []
        lines.append(f"{' REGRESSION RESULTS ':=^78}")
        
        lines.append(f"Observations: {s['n_observations']:<15} R-squared:      {m['r_squared']:.4f}")
        lines.append(f"Features:     {s['n_features']:<15} Adj. R-squared: {m['adjusted_r_squared']:.4f}")
        lines.append(f"RMSE:         {m['rmse']:<15.4f} F-statistic:    {f['f_stat']:.4f}")
        lines.append(f"Res. Std Err: {m['residual_std_error']:<15.4f} Prob (F-stat):  {f['p_value']:.4g}")
        lines.append("-" * 78)
        
        # Coefficients table
        conf_level_pct = s.get('confidence_level', 0.95) * 100
        ci_header = f"[{conf_level_pct:.1f}% Conf. Int.]"
        lines.append(f"{'':<28} {'Coef':>10} {'Std Err':>10} {'t':>8} {'P>|t|':>8} {ci_header:>18}")
        lines.append("-" * 78)
        
        c = s['coefficients']
        names = c['names']
        
        for i, name in enumerate(names):
            name_str = (name[:25] + '..') if len(name) > 27 else name

            pv = c['p_values'][i]
            if pv < 0.0001:
                pv_str = "0.0000"
                sig_mark = " *"
            elif pv < 0.05:
                pv_str = f"{pv:.4f}"
                sig_mark = " ."
            else:
                pv_str = f"{pv:.4f}"
                sig_mark = "  "

            display_name = name_str + sig_mark
                
            lines.append(
                f"{display_name:<28} "
                f"{c['values'][i]:10.4f} "
                f"{c['std_errors'][i]:10.4f} "
                f"{c['t_stats'][i]:8.2f} "
                f"{pv_str:>8} "
                f"{c['ci_lower'][i]:8.4f} {c['ci_upper'][i]:8.4f}"
            )
            
        lines.append("=" * 78)
        lines.append("* p<0.0001, . p<0.05")
        return "\n".join(lines)
   




