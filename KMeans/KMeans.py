from tensorflow.contrib.factorization.python.ops.clustering_ops import *
from tensorflow.contrib.factorization.python.ops.clustering_ops import _InitializeClustersOpFactory
from tensorflow.contrib.factorization import KMeans


class KMeansExperimental(KMeans):
    def __init__(self,
                 inputs,
                 num_clusters,
                 initial_clusters=RANDOM_INIT,
                 distance_metric=SQUARED_EUCLIDEAN_DISTANCE,
                 use_mini_batch=False,
                 mini_batch_steps_per_iteration=1,
                 random_seed=0,
                 kmeans_plus_plus_num_retries=2,
                 kmc2_chain_length=200,
                 experimental_score=False):
        super(KMeansExperimental, self).__init__(inputs,
                                                 num_clusters,
                                                 initial_clusters=initial_clusters,
                                                 distance_metric=distance_metric,
                                                 use_mini_batch=use_mini_batch,
                                                 mini_batch_steps_per_iteration=mini_batch_steps_per_iteration,
                                                 random_seed=random_seed,
                                                 kmeans_plus_plus_num_retries=kmeans_plus_plus_num_retries,
                                                 kmc2_chain_length=kmc2_chain_length)
        self.experimental_score = experimental_score

    def training_graph(self):
        """Generate a training graph for kmeans algorithm.

        This returns, among other things, an op that chooses initial centers
        (init_op), a boolean variable that is set to True when the initial centers
        are chosen (cluster_centers_initialized), and an op to perform either an
        entire Lloyd iteration or a mini-batch of a Lloyd iteration (training_op).
        The caller should use these components as follows. A single worker should
        execute init_op multiple times until cluster_centers_initialized becomes
        True. Then multiple workers may execute training_op any number of times.

        Returns:
          A tuple consisting of:
          all_scores: A matrix (or list of matrices) of dimensions (num_input,
            num_clusters) where the value is the distance of an input vector and a
            cluster center.
          cluster_idx: A vector (or list of vectors). Each element in the vector
            corresponds to an input row in 'inp' and specifies the cluster id
            corresponding to the input.
          scores: Similar to cluster_idx but specifies the distance to the
            assigned cluster instead.
          cluster_centers_initialized: scalar indicating whether clusters have been
            initialized.
          init_op: an op to initialize the clusters.
          training_op: an op that runs an iteration of training.
        """
        # Implementation of kmeans.
        if (isinstance(self._initial_clusters, str) or
                callable(self._initial_clusters)):
            initial_clusters = self._initial_clusters
            num_clusters = ops.convert_to_tensor(self._num_clusters)
        else:
            initial_clusters = ops.convert_to_tensor(self._initial_clusters)
            num_clusters = array_ops.shape(initial_clusters)[0]

        inputs = self._inputs
        (cluster_centers_var, cluster_centers_initialized, total_counts,
         cluster_centers_updated,
         update_in_steps) = self._create_variables(num_clusters)
        init_op = _InitializeClustersOpFactory(
            self._inputs, num_clusters, initial_clusters, self._distance_metric,
            self._random_seed, self._kmeans_plus_plus_num_retries,
            self._kmc2_chain_length, cluster_centers_var, cluster_centers_updated,
            cluster_centers_initialized).op()
        cluster_centers = cluster_centers_var

        if self._distance_metric == COSINE_DISTANCE:
            inputs = self._l2_normalize_data(inputs)
            if not self._clusters_l2_normalized():
                cluster_centers = nn_impl.l2_normalize(cluster_centers, dim=1)

        all_scores, scores, cluster_idx = self._infer_graph(inputs, cluster_centers)
        if self._use_mini_batch:
            sync_updates_op = self._mini_batch_sync_updates_op(
                update_in_steps, cluster_centers_var, cluster_centers_updated,
                total_counts)
            assert sync_updates_op is not None
            with ops.control_dependencies([sync_updates_op]):
                training_op = self._mini_batch_training_op(
                    inputs, cluster_idx, cluster_centers_updated, total_counts)
        else:
            assert cluster_centers == cluster_centers_var
            if self.experimental_score:
                training_op = self._full_batch_training_op_ex(
                    inputs, num_clusters, cluster_idx, cluster_centers_var, scores)
            else:
                training_op = self._full_batch_training_op(
                    inputs, num_clusters, cluster_idx, cluster_centers_var)
        return (all_scores, cluster_idx, scores, cluster_centers_initialized,
                init_op, training_op)

    def _full_batch_training_op_ex(self, inputs, num_clusters, cluster_idx_list,
                                   cluster_centers, scores):
        """Creates an op for training for full batch case.

        Args:
          inputs: list of input Tensors.
          num_clusters: an integer Tensor providing the number of clusters.
          cluster_idx_list: A vector (or list of vectors). Each element in the
            vector corresponds to an input row in 'inp' and specifies the cluster id
            corresponding to the input.
          cluster_centers: Tensor Ref of cluster centers.

        Returns:
          An op for doing an update of mini-batch k-means.
        """
        epsilon = constant_op.constant(1e-6, dtype=inputs[0].dtype)
        cluster_sums = []
        weights = []
        for inp, cluster_idx, score in zip(inputs, cluster_idx_list, scores):
            with ops.colocate_with(cluster_centers, ignore_existing=True):

                mean_distance = math_ops.unsorted_segment_sum(score, cluster_idx, num_clusters)\
                    / math_ops.unsorted_segment_sum(array_ops.ones(array_ops.shape(inp)[0]), cluster_idx, num_clusters)
                mean_map = array_ops.gather(mean_distance, cluster_idx)
                weight = array_ops.reshape(math_ops.exp(-score / mean_map),[-1,1])
                cluster_sums.append(math_ops.unsorted_segment_sum(
                    inp*weight, cluster_idx, num_clusters))
                weights.append(math_ops.unsorted_segment_sum(weight, cluster_idx, num_clusters))

        with ops.colocate_with(cluster_centers, ignore_existing=True):
            new_clusters_centers = math_ops.add_n(cluster_sums) / (math_ops.add_n(weights)+epsilon)

            if self._clusters_l2_normalized():
                new_clusters_centers = nn_impl.l2_normalize(new_clusters_centers, dim=1)
        return state_ops.assign(cluster_centers, new_clusters_centers)
