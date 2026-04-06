# This module is partially modified from scikit-learn.
# Original source: https://github.com/scikit-learn/scikit-learn
#
# License: BSD 3-Clause License
# Copyright (c) 2007-2024, scikit-learn developers.
# All rights reserved.
#
# See https://github.com/scikit-learn/scikit-learn/blob/main/COPYING
# for the full license text.


import sys
from sklearn.mixture import BayesianGaussianMixture
import numpy as np
from sklearn.utils import check_random_state
import warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.special import logsumexp
from sklearn.mixture._gaussian_mixture import _estimate_gaussian_parameters
from joblib import Parallel, delayed
import copy
import kmeans



def _one_hot(y, k):
    """
    y is a vector; k is the maximum number of classes
    """
    y_one_hot = np.zeros((y.size, k))
    y_one_hot[np.arange(y.size), y] = 1
    return y_one_hot


class BayesianGaussianMixtureMerge(BayesianGaussianMixture):
    def __init__(
        self,
        *,
        n_components=1,
        covariance_type="full",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=None,
        mean_precision_prior=None,
        mean_prior=None,
        degrees_of_freedom_prior=None,
        covariance_prior=None,
        random_state=None,
        warm_start=False,
        verbose=0,
        verbose_interval=10,
    ):
        super().__init__(
            n_components=n_components,
            covariance_type=covariance_type,
            tol=tol,
            reg_covar=reg_covar,
            max_iter=max_iter,
            n_init=n_init,
            init_params=init_params,
            weight_concentration_prior_type=weight_concentration_prior_type,
            weight_concentration_prior=weight_concentration_prior,
            mean_precision_prior=mean_precision_prior,
            mean_prior=mean_prior,
            degrees_of_freedom_prior=degrees_of_freedom_prior,
            covariance_prior=covariance_prior,
            random_state=random_state,
            warm_start=warm_start,
            verbose=verbose,
            verbose_interval=verbose_interval,
        )


    def _initialize_parameters_v2(self, X, y, random_state):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like of shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance that controls the random seed
            used for the method chosen to initialize the parameters.
        """
        X_src = X[len(y):, ]
        n_samples, _ = X_src.shape
        n_components = self.n_components - len(np.unique(y))
        _, resp = kmeans.Big_KMeans(n_clusters=self.n_components, runs=1, random_state=random_state).fit_merge_pyx(X, y)
        resp = _one_hot(resp, self.n_components)
        y_one_hot = _one_hot(y, self.n_components)
        resp_all = np.concatenate([y_one_hot, resp], axis=0)
        self._initialize(X, resp_all)
        return resp


    @staticmethod
    def _single_init_worker(base_self, X, y, seed, do_init, init):
        model = copy.deepcopy(base_self)   # independent copy
        random_state = check_random_state(seed)
        model._print_verbose_msg_init_beg(init)
        converged_ = False
        if do_init:
            init_resp = model._initialize_parameters_v2(X, y, random_state)
        else:
            init_resp = None

        lower_bound = -np.inf if do_init else model.lower_bound_

        if model.max_iter == 0:
            params = model._get_parameters()
            n_iter = 0
        else:
            for n_iter in range(1, model.max_iter + 1):
                prev_lower_bound = lower_bound

                log_prob_norm, log_resp, fake_resp, part_resp = model._e_step_v1(X, y)

                model._m_step_v2(X, fake_resp, part_resp)

                lower_bound = model._compute_lower_bound(log_resp, log_prob_norm)

                change = lower_bound - prev_lower_bound
                model._print_verbose_msg_iter_end(n_iter, change)

                if abs(change) < model.tol:
                    converged_ = True
                    break

            params = model._get_parameters()
            n_iter = n_iter

        return {
            'lower_bound': lower_bound,
            'params': params,
            'n_iter': n_iter,
            'init_resp': init_resp, 
            'converged_': converged_
        }


    def fit_merge(self, X, y):
        """Estimate model parameters using X and predict the labels for X.

        The method fits the model n_init times and sets the parameters with
        which the model has the largest likelihood or lower bound. Within each
        trial, the method iterates between E-step and M-step for `max_iter`
        times until the change of likelihood or lower bound is less than
        `tol`, otherwise, a :class:`~sklearn.exceptions.ConvergenceWarning` is
        raised. After fitting, it predicts the most probable label for the
        input data points.

        .. versionadded:: 0.20

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.
        """
        X = self._validate_data(X, dtype=[np.float64, np.float32], ensure_min_samples=2)
        if X.shape[0] < self.n_components:
            raise ValueError(
                "Expected n_samples >= n_components "
                f"but got n_components = {self.n_components}, "
                f"n_samples = {X.shape[0]}"
            )
        self._check_parameters(X)

        # if we enable warm_start, we will have a unique initialisation
        do_init = not (self.warm_start and hasattr(self, "converged_"))
        n_init = self.n_init if do_init else 1

        max_lower_bound = -np.inf
        self.converged_ = False

        random_state = check_random_state(self.random_state)
        seeds = random_state.randint(0, 1_000_000, size=n_init)  # independent seeds

        n_samples, _ = X.shape

        results = Parallel(n_jobs=-1)(
            delayed(self._single_init_worker)(self, X, y, seeds[init], do_init, init) for init in range(n_init)
        )


        best = max(results, key=lambda r: r["lower_bound"])

        max_lower_bound = best["lower_bound"]
        best_params = best["params"]
        best_n_iter = best["n_iter"]
        best_init = best["init_resp"]
        self.converged_ = best["converged_"]

        if not self.converged_ and self.max_iter > 0:
            warnings.warn(
                "Initialization did not converge. "
                "Try different init parameters, "
                "or increase max_iter, tol "
                "or check for degenerate data.",
                ConvergenceWarning,
            )

        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound


        _, log_resp, _, _ = self._e_step_v1(X, y)
        return log_resp.argmax(axis=1), np.exp(log_resp), best_init


    def _e_step_v1(self, X, y):
        """E step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns        tar_data = tar_data1
        tar_label = tar_label1

        -------
        log_prob_norm : float
            Mean of the logarithms of the probabilities of each sample in X

        log_responsibility : array, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        log_prob_norm, log_resp, fake_resp, part_resp = self._estimate_log_prob_resp_y(X, y)
        return np.mean(log_prob_norm), log_resp, fake_resp, part_resp



    def _m_step_v2(self, X, fake_resp, part_resp):
        """M step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        log_resp : array-like of shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        nk2 = part_resp.sum(axis=0) + 10*np.finfo(part_resp.dtype).eps
        nk, xk, sk = _estimate_gaussian_parameters(
            X, fake_resp, self.reg_covar, self.covariance_type
        )
        self._estimate_weights(nk2)
        self._estimate_means(nk, xk)
        self._estimate_precisions(nk, xk, sk)

    def _estimate_log_prob_resp_y(self, X, y):
        """Estimate log probabilities and responsibilities for each sample.

        Compute the log probabilities, weighted log probabilities per
        component and responsibilities for each sample in X with respect to
        the current state of the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        log_prob_norm : array, shape (n_samples,)
            log p(X)

        log_responsibilities : array, shape (n_samples, n_components)
            logarithm of the responsibilities
        """
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under="ignore"):
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]

        y_one_hot = _one_hot(y, self.n_components)
        resp = np.exp(log_resp)
        np.set_printoptions(threshold=sys.maxsize)

        fake_resp = np.concatenate([y_one_hot, resp[len(y_one_hot):, ]], axis=0)
        assert resp.shape == fake_resp.shape
        return log_prob_norm, log_resp[len(y_one_hot):, ], fake_resp, resp[len(y_one_hot):, ]
