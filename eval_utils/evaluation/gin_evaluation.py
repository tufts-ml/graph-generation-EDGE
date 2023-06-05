import os
import torch
import numpy as np
import time
from scipy import linalg
import sklearn
import dgl
from sklearn.preprocessing import StandardScaler

from eval_utils.evaluation.models.gin.gin import GIN


def load_feature_extractor(
    device, num_layers=3, hidden_dim=35, neighbor_pooling_type='sum',
        graph_pooling_type='sum', input_dim=1, edge_feat_dim=0,
        dont_concat=False, num_mlp_layers=2, output_dim=1,
        node_feat_loc='attr', edge_feat_loc='attr', init='orthogonal',
        **kwargs):

    model = GIN(
        num_layers=num_layers, hidden_dim=hidden_dim,
        neighbor_pooling_type=neighbor_pooling_type,
        graph_pooling_type=graph_pooling_type, input_dim=input_dim,
        edge_feat_dim=edge_feat_dim, num_mlp_layers=num_mlp_layers,
        output_dim=output_dim, init=init)

    model.node_feat_loc = node_feat_loc
    model.edge_feat_loc = edge_feat_loc

    use_pretrained = kwargs.get('use_pretrained', False)
    if use_pretrained:
        model_path = kwargs.get('model_path')
        assert model_path is not None, 'Please pass model_path if use_pretrained=True'
        print('loaded', model_path)
        saved_model = torch.load(model_path)
        model.load_state_dict(saved_model['model_state_dict'])

    model.eval()

    if dont_concat:
        model.forward = model.get_graph_embed_no_cat
    else:
        model.forward = model.get_graph_embed

    model.device = device
    return model.to(device)

def time_function(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        end = time.time()
        return results, end - start
    return wrapper

# class OneHotEncoder():
#     def __init__(self, dataset, graphs=None):
#         if graphs is not None:
#             graphs = dgl.batch(graphs)
#             self.max_degree = torch.max(graphs.in_degrees())
#         elif dataset == 'grid':
#             self.max_degree = 4
#         elif dataset == 'zinc':
#             self.max_degree = 4
#         elif dataset == 'lobster':
#             self.max_degree = 18
#         elif dataset == 'community':
#             self.max_degree = 40
#         elif dataset == 'ego':
#             self.max_degree = 99
#         elif dataset == 'proteins':
#             self.max_degree = 15
#         else:
#             raise Exception(f'Unsupported dataset {dataset}, pls pass graphs')
#
#         self.onehot = torch.eye(self.max_degree + 1)
#
#     def encode(self, degrees):
#         onehot = self.onehot.to(degrees.device)
#         degrees = torch.clamp(degrees, max=self.max_degree + 1)
#         return onehot[degrees]


class GINMetric():
    def __init__(self, model):
        self.feat_extractor = model
        self.get_activations = self.get_activations_gin


    @time_function
    def get_activations_gin(self, generated_dataset, reference_dataset):
        return self._get_activations(generated_dataset, reference_dataset)

    def _get_activations(self, generated_dataset, reference_dataset):
        gen_activations = self.__get_activations_single_dataset(generated_dataset)
        ref_activations = self.__get_activations_single_dataset(reference_dataset)

        scaler = StandardScaler()
        scaler.fit(ref_activations)
        ref_activations = scaler.transform(ref_activations)
        gen_activations = scaler.transform(gen_activations)

        return gen_activations, ref_activations

    def __get_activations_single_dataset(self, dataset):
        node_feat_loc = self.feat_extractor.node_feat_loc
        edge_feat_loc = self.feat_extractor.edge_feat_loc

        ndata = [node_feat_loc] if node_feat_loc in dataset[0].ndata\
            else '__ALL__'
        edata = [edge_feat_loc] if edge_feat_loc in dataset[0].edata\
            else '__ALL__'
        graphs = dgl.batch(
            dataset, ndata=ndata, edata=edata).to(self.feat_extractor.device)

        if node_feat_loc not in graphs.ndata:  # Use degree as features
            feats = graphs.in_degrees() + graphs.out_degrees()
            feats = feats.unsqueeze(1).type(torch.float32)
        else:
            feats = graphs.ndata[node_feat_loc]
        feats = feats.to(self.feat_extractor.device)

        graph_embeds = self.feat_extractor(graphs, feats)
        return graph_embeds.cpu().detach().numpy()

    def evaluate(self, *args, **kwargs):
        raise Exception('Must be implemented by child class')


class MMDEvaluation(GINMetric):
    def __init__(self, model, kernel='rbf', sigma='range', multiplier='mean'):
        super().__init__(model)

        if multiplier == 'mean':
            self.__get_sigma_mult_factor = self.__mean_pairwise_distance
        elif multiplier == 'median':
            self.__get_sigma_mult_factor = self.__median_pairwise_distance
        elif multiplier is None:
            self.__get_sigma_mult_factor = lambda *args, **kwargs: 1
        else:
            raise Exception(multiplier)

        if 'rbf' in kernel:
            if sigma == 'range':
                self.base_sigmas = np.array([
                    0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0])

                if multiplier == 'mean':
                    self.name = 'mmd_rbf'
                elif multiplier == 'median':
                    self.name = 'mmd_rbf_adaptive_median'
                else:
                    self.name = 'mmd_rbf_adaptive'

            elif sigma == 'one':
                self.base_sigmas = np.array([1])

                if multiplier == 'mean':
                    self.name = 'mmd_rbf_single_mean'
                elif multiplier == 'median':
                    self.name = 'mmd_rbf_single_median'
                else:
                    self.name = 'mmd_rbf_single'

            else:
                raise Exception(sigma)

            self.evaluate = self.calculate_MMD_rbf_quadratic

        elif 'linear' in kernel:
            self.evaluate = self.calculate_MMD_linear_kernel

        else:
            raise Exception()

    def __get_pairwise_distances(self, generated_dataset, reference_dataset):
        return sklearn.metrics.pairwise_distances(
            reference_dataset, generated_dataset,
            metric='euclidean', n_jobs=8) ** 2

    def __mean_pairwise_distance(self, dists_GR):
        return np.sqrt(dists_GR.mean())

    def __median_pairwise_distance(self, dists_GR):
        return np.sqrt(np.median(dists_GR))

    def get_sigmas(self, dists_GR):
        mult_factor = self.__get_sigma_mult_factor(dists_GR)
        return self.base_sigmas * mult_factor

    @time_function
    def calculate_MMD_rbf_quadratic(self, generated_dataset=None, reference_dataset=None):
        # https://github.com/djsutherland/opt-mmd/blob/master/two_sample/mmd.py
        if not isinstance(generated_dataset, torch.Tensor) and not isinstance(generated_dataset, np.ndarray):
            (generated_dataset, reference_dataset), _ = self.get_activations(generated_dataset, reference_dataset)

        GG = self.__get_pairwise_distances(generated_dataset, generated_dataset)
        GR = self.__get_pairwise_distances(generated_dataset, reference_dataset)
        RR = self.__get_pairwise_distances(reference_dataset, reference_dataset)

        max_mmd = 0
        sigmas = self.get_sigmas(GR)
        for sigma in sigmas:
            gamma = 1 / (2 * sigma**2)

            K_GR = np.exp(-gamma * GR)
            K_GG = np.exp(-gamma * GG)
            K_RR = np.exp(-gamma * RR)

            mmd = K_GG.mean() + K_RR.mean() - 2 * K_GR.mean()
            max_mmd = mmd if mmd > max_mmd else max_mmd

        return {self.name: max_mmd}

    @time_function
    def calculate_MMD_linear_kernel(self, generated_dataset=None, reference_dataset=None):
        # https://github.com/djsutherland/opt-mmd/blob/master/two_sample/mmd.py
        if not isinstance(generated_dataset, torch.Tensor) and not isinstance(generated_dataset, np.ndarray):
            generated_dataset, reference_dataset, _ = self.get_activations(generated_dataset, reference_dataset)

        G_bar = generated_dataset.mean(axis=0)
        R_bar = reference_dataset.mean(axis=0)
        Z_bar = G_bar - R_bar
        mmd = Z_bar.dot(Z_bar)
        mmd = mmd if mmd >= 0 else 0
        return {'mmd_linear': mmd}


class KIDEvaluation(GINMetric):
    @time_function
    def evaluate(self, generated_dataset=None, reference_dataset=None):
        import tensorflow as tf
        import tensorflow_gan as tfgan

        if not isinstance(generated_dataset, torch.Tensor) and not isinstance(generated_dataset, np.ndarray):
            generated_dataset, reference_dataset, _ = self.get_activations(generated_dataset, reference_dataset)

        gen_activations = tf.convert_to_tensor(generated_dataset, dtype=tf.float32)
        ref_activations = tf.convert_to_tensor(reference_dataset, dtype=tf.float32)
        kid = tfgan.eval.kernel_classifier_distance_and_std_from_activations(ref_activations, gen_activations)[0].numpy()
        return {'kid': kid}

class FIDEvaluation(GINMetric):
    # https://github.com/mseitzer/pytorch-fid
    @time_function
    def evaluate(self, generated_dataset=None, reference_dataset=None):
        if not isinstance(generated_dataset, torch.Tensor) and not isinstance(generated_dataset, np.ndarray):
            generated_dataset, reference_dataset, _ = self.get_activations(generated_dataset, reference_dataset)

        mu_ref, cov_ref = self.__calculate_dataset_stats(reference_dataset)
        mu_generated, cov_generated = self.__calculate_dataset_stats(generated_dataset)
        # print(np.max(mu_generated), np.max(cov_generated), 'mu, cov fid')
        fid = self.compute_FID(mu_ref, mu_generated, cov_ref, cov_generated)
        return {'fid': fid}

    def __calculate_dataset_stats(self, activations):
        # print('activation mean -----------------------------------------', activations.mean())
        mu = np.mean(activations, axis = 0)
        cov = np.cov(activations, rowvar = False)

        return mu, cov

    def compute_FID(self, mu1, mu2, cov1, cov2, eps = 1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """
        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert cov1.shape == cov2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
        # print(np.max(covmean), 'covmean')
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(cov1.shape[0]) * eps
            covmean = linalg.sqrtm((cov1 + offset).dot(cov2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                #raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        # print(tr_covmean, 'tr_covmean')

        return (diff.dot(diff) + np.trace(cov1) +
                np.trace(cov2) - 2 * tr_covmean)


class prdcEvaluation(GINMetric):
    # From PRDC github: https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py#L54
    def __init__(self, *args, use_pr=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_pr = use_pr

    @time_function
    def evaluate(self, generated_dataset=None, reference_dataset=None, nearest_k=5):
        """
        Computes precision, recall, density, and coverage given two manifolds.
        Args:
            real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            nearest_k: int.
        Returns:
            dict of precision, recall, density, and coverage.
        """
        # start = time.time()
        if not isinstance(generated_dataset, torch.Tensor) and not isinstance(generated_dataset, np.ndarray):
            generated_dataset, reference_dataset, _ = self.get_activations(generated_dataset, reference_dataset)

        real_nearest_neighbour_distances = self.__compute_nearest_neighbour_distances(
            reference_dataset, nearest_k)
        distance_real_fake = self.__compute_pairwise_distance(
            reference_dataset, generated_dataset)

        if self.use_pr:
            fake_nearest_neighbour_distances = self.__compute_nearest_neighbour_distances(
                generated_dataset, nearest_k)
            precision = (
                    distance_real_fake <=
                    np.expand_dims(real_nearest_neighbour_distances, axis=1)
            ).any(axis=0).mean()

            recall = (
                    distance_real_fake <=
                    np.expand_dims(fake_nearest_neighbour_distances, axis=0)
            ).any(axis=1).mean()

            f1_pr = 2 / ((1 / (precision + 1e-5)) + (1 / (recall + 1e-5)))
            result = dict(precision=precision, recall=recall, f1_pr=f1_pr)

        else:
            density = (1. / float(nearest_k)) * (
                    distance_real_fake <=
                    np.expand_dims(real_nearest_neighbour_distances, axis=1)
            ).sum(axis=0).mean()

            coverage = (
                    distance_real_fake.min(axis=1) <=
                    real_nearest_neighbour_distances
            ).mean()

            f1_dc = 2 / ((1 / (density + 1e-5)) + (1 / (coverage + 1e-5)))
            result = dict(density=density, coverage=coverage, f1_dc=f1_dc)
        # end = time.time()
        # print('prdc', start - end)
        return result

    def __compute_pairwise_distance(self, data_x, data_y=None):
        """
        Args:
            data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
            data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
        Returns:
            numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
        """
        if data_y is None:
            data_y = data_x
        dists = sklearn.metrics.pairwise_distances(
            data_x, data_y, metric='euclidean', n_jobs=8)
        return dists


    def __get_kth_value(self, unsorted, k, axis=-1):
        """
        Args:
            unsorted: numpy.ndarray of any dimensionality.
            k: int
        Returns:
            kth values along the designated axis.
        """
        indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
        k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
        kth_values = k_smallests.max(axis=axis)
        return kth_values


    def __compute_nearest_neighbour_distances(self, input_features, nearest_k):
        """
        Args:
            input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            nearest_k: int
        Returns:
            Distances to kth nearest neighbours.
        """
        distances = self.__compute_pairwise_distance(input_features)
        radii = self.__get_kth_value(distances, k=nearest_k + 1, axis=-1)
        return radii
