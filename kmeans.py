# This code is modified from scikit-learn.
# Original source: https://github.com/scikit-learn/scikit-learn
#
# License: BSD 3-Clause License
# Copyright (c) 2007-2024, scikit-learn developers.
# All rights reserved.
#
# See https://github.com/scikit-learn/scikit-learn/blob/main/COPYING
# for the full license text.


import numpy as np
import scipy.sparse as sp
from sklearn.cluster import kmeans_plusplus as kmeans_plusplus_v1
from sklearn.metrics.pairwise import euclidean_distances
from _k_means_merge import lloyd_iter_chunked_dense as lloyd_iter_merge
from sklearn.cluster._k_means_common import _inertia_dense, _is_same_clustering
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms, stable_cumsum
from sklearn.utils.validation import _check_sample_weight

def _kmeans_plusplus_v2(
    X,
    n_clusters,
    given_centers,
    sample_weight=None,
    random_state=None,
    n_local_trials=None
):
    """Computational component for initialization of n_clusters by
    k-means++. Prior validation of data is assumed.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds for.

    n_clusters : int
        The number of seeds to choose.

    given_centers: ndarray of shape (n_given_centers, n_features)
        The given centers

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in `X`.

    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.

    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The initial centers for k-means.

    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    """
    sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
    random_state = check_random_state(random_state)
    x_squared_norms = row_norms(X, squared=True)


    n_samples, n_features = X.shape
    n_given_centers = len(given_centers)

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)
    #centers[:n_given_centers] = given_centers

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    indices = np.full(n_clusters, -1, dtype=int)

    given_centers = np.array(given_centers, dtype=np.float32)
    closest_dist_sq = euclidean_distances(
        given_centers, X, Y_norm_squared=x_squared_norms, squared=True
    )
    current_pot = closest_dist_sq @ sample_weight # get a c array
    start_center = np.argmin(current_pot)
    closest_dist_sq = closest_dist_sq[start_center]
    current_pot = current_pot[start_center]

    # Pick the remaining points
    for c in range(0, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.uniform(size=n_local_trials) * current_pot
        candidate_ids = np.searchsorted(
            stable_cumsum(sample_weight * closest_dist_sq), rand_vals
        )
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True
        )

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates @ sample_weight.reshape(-1, 1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers, indices


def _kmeans_single_lloyd_merge(
    X,
    given_centers,
    given_weights_in_clusters ,
    sample_weight,
    centers_init,
    max_iter=300,
    verbose=False,
    tol=1e-2,
    n_threads=1,
):
    """A single run of k-means lloyd, assumes preparation completed prior.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. If sparse matrix, must be in CSR format.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    centers_init : ndarray of shape (n_clusters, n_features)
        The initial centers.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    label : ndarray of shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
    n_clusters = centers_init.shape[0]

    # Buffers to avoid new allocations at each iteration.
    centers = centers_init.astype(dtype=np.float32)
    centers_new = np.zeros_like(centers,dtype=np.float32)
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    labels_old = labels.copy()
    weight_in_clusters = np.zeros(n_clusters, dtype=np.float32)
    center_shift = np.zeros(n_clusters, dtype=np.float32)

    lloyd_iter = lloyd_iter_merge
    _inertia = _inertia_dense
    # strict_convergence = False



    for i in range(max_iter):
        lloyd_iter(
            X,
            given_centers,
            given_weights_in_clusters,
            sample_weight,
            centers,
            centers_new,
            weight_in_clusters,
            labels,
            center_shift,
            n_threads,
        )

        if verbose:
            inertia = _inertia(X, sample_weight, centers, labels, n_threads)
            #print(f"Iteration {i}, inertia {inertia}.")

        centers, centers_new = centers_new, centers

        if np.array_equal(labels, labels_old):
            # First check the labels for strict convergence.
            if verbose:
                print(f"Converged at iteration {i}: strict convergence.")
            strict_convergence = True
            break
        else:
            # No strict convergence, check for tol based convergence.
            center_shift_tot = (center_shift**2).sum()
            clusters_not_changed = np.abs(centers_new - centers) < tol
            if np.all(clusters_not_changed) != False:
                if verbose:
                    print(
                        f"Converged at iteration {i}: center shift "
                        f"{center_shift_tot} within tolerance {tol}."
                    )
                break
        labels_old[:] = labels

    inertia = _inertia(X, sample_weight, centers, labels, n_threads)

    return labels, inertia, centers, i + 1

class Big_KMeans:
    def __init__(self, n_clusters=3, tolerance=0.01, max_iter=100, runs=1, random_state=None):
        self.n_clusters = n_clusters
        self.tolerance = tolerance
        self.cluster_means = np.zeros(n_clusters)
        self.max_iter = max_iter
        self.runs = runs
        self.random_state = random_state
        self._n_threads = _openmp_effective_n_threads()

    def fit_merge_pyx(self, X, y):
        y_cls = np.unique(y)
        n_clusters_unknown = self.n_clusters - len(np.unique(y))
        X_src = X[:len(y), ]
        X = X[len(y):, ]
        row_count, col_count = X.shape
        centers_init  = np.zeros((self.n_clusters, col_count))
        given_centers = []
        given_centers_iter = np.zeros_like(centers_init, dtype=np.float32) # given centers for kmeans lloyd iteration
        given_weights_in_clusters = np.zeros(self.n_clusters, dtype=np.float32)

        for c in y_cls:
            centers_init [c,:] = np.mean(X_src[y==c,:], axis=0)
            given_centers.append(centers_init [c,:])
            given_centers_iter[c] = X_src[y == c].sum(axis=0)
            given_weights_in_clusters[c] = np.sum(y == c).astype(np.float32)

        diff_cls = np.array([c for c in range(self.n_clusters) if c not in y_cls])
        given_centers = np.stack(given_centers, axis=0) # given centers for kmeans++
        given_centers_iter = given_centers_iter.astype(np.float32)
        given_weights_in_clusters = given_weights_in_clusters.astype(np.float32)
    
        best_inertia, best_labels = None, None
        sample_weight = np.ones(X.shape[0], dtype=X.dtype)


        for i in range(self.runs):
            centers_init [diff_cls, :] = self.__initialize_means(X, n_clusters_unknown, given_centers, 200,
                                                                self.random_state)

            n_threads=1
            labels, inertia, centers, n_iter_ = _kmeans_single_lloyd_merge(
                X,
                given_centers_iter,
                given_weights_in_clusters,
                sample_weight,
                centers_init,
                max_iter=self.max_iter,
                verbose=True,
                tol=self.tolerance,
                n_threads=self._n_threads,
            )

            if best_inertia is None or (
                inertia < best_inertia
                and not _is_same_clustering(labels, best_labels, self.n_clusters)
            ):
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
                best_n_iter = n_iter_

        return best_centers, best_labels

    def __post_initialize(self, centers, given_centers, k, indices):
        dist = euclidean_distances(centers, given_centers)
        dist_min = np.min(dist, axis=1)
        dist_argsort = np.argsort(dist_min)[::-1]
        index = dist_argsort[:k]
        return centers[index], indices[index]

    def __initialize_means(self, X, k, given_centers=None, max_k=None, random_state=None):
        if given_centers is not None:
            if max_k is not None:
                centers, indices = _kmeans_plusplus_v2(X, n_clusters=max_k, given_centers=given_centers, random_state=random_state)
                centers, indices = self.__post_initialize(centers, given_centers, k, indices)
            else:
                centers, indices = _kmeans_plusplus_v2(X, n_clusters=k, given_centers=given_centers, random_state=random_state)

        else:
            centers, indices = kmeans_plusplus_v1(X, n_clusters=k, random_state=random_state)
        return np.array(centers)


