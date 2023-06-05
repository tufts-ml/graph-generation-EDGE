import torch
import torch.nn.functional as F
import numpy as np
from inspect import isfunction
from torch_scatter import scatter
import torch_geometric as pyg
from diffusion.diffusion_base import *
"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8


class BinomialDiffusionVanilla(DiffusionBase):
    def __init__(self, num_node_classes, num_edge_classes, initial_graph_sampler, denoise_fn, timesteps=1000,
                 loss_type='vb_kl', parametrization='x0', final_prob_node=None, final_prob_edge=None, sample_time_method='importance', 
                 noise_schedule=cosine_beta_schedule, device='cuda'):
        super(BinomialDiffusionVanilla, self).__init__(num_node_classes, num_edge_classes, initial_graph_sampler, denoise_fn,
                                                    timesteps, sample_time_method, device)

        log_final_prob_node = torch.tensor(final_prob_node)[None, :].log()
        log_final_prob_edge = torch.tensor(final_prob_edge)[None, :].log()
        
        self.loss_type = loss_type
        self.parametrization = parametrization
        alphas = noise_schedule(timesteps)
        alphas = torch.tensor(alphas.astype('float64'))
        log_alpha = np.log(alphas)
        log_cumprod_alpha = np.cumsum(log_alpha)

        log_1_min_alpha = log_1_min_a(log_alpha)
        log_1_min_cumprod_alpha = log_1_min_a(log_cumprod_alpha)
        assert log_add_exp(log_alpha, log_1_min_alpha).abs().sum().item() < 1.e-5
        assert log_add_exp(log_cumprod_alpha, log_1_min_cumprod_alpha).abs().sum().item() < 1e-5
        assert (np.cumsum(log_alpha) - log_cumprod_alpha).abs().sum().item() < 1.e-5

        self.register_buffer('log_alpha', log_alpha.float())
        self.register_buffer('log_1_min_alpha', log_1_min_alpha.float())
        self.register_buffer('log_cumprod_alpha', log_cumprod_alpha.float())
        self.register_buffer('log_1_min_cumprod_alpha', log_1_min_cumprod_alpha.float())
        self.register_buffer('log_final_prob_node', log_final_prob_node.float())
        self.register_buffer('log_final_prob_edge', log_final_prob_edge.float())

        self.register_buffer('Lt_history', torch.zeros(timesteps))
        self.register_buffer('Lt_count', torch.zeros(timesteps))

    def _q_posterior(self, log_x_start, log_x_t, t, log_final_prob):
        assert log_x_start.shape[1] == 2, f'num_class > 2 not supported'

        tmin1 = t - 1
        # Remove negative values, will not be used anyway for final decoder
        tmin1 = torch.where(tmin1 < 0, torch.zeros_like(tmin1), tmin1)
        log_p1 = log_final_prob[:,1]
        log_1_min_p1 = log_final_prob[:,0]

        log_x_start_real = log_x_start[:,1]
        log_1_min_x_start_real = log_x_start[:,0]

        log_x_t_real = log_x_t[:,1]
        log_1_min_x_t_real = log_x_t[:,0]

        log_alpha_t = extract(self.log_alpha, t, log_x_start_real.shape)
        log_beta_t = extract(self.log_1_min_alpha, t, log_x_start_real.shape)
        
        log_cumprod_alpha_tmin1 = extract(self.log_cumprod_alpha, tmin1, log_x_start_real.shape)

        log_1_min_cumprod_alpha_tmin1 = extract(self.log_1_min_cumprod_alpha, tmin1, log_x_start_real.shape)


        log_xtmin1_eq_0_given_x_t = log_add_exp(log_beta_t+log_p1+log_x_t_real, log_1_min_a(log_beta_t+log_p1)+log_1_min_x_t_real)
        log_xtmin1_eq_1_given_x_t = log_add_exp(log_add_exp(log_alpha_t, log_beta_t+log_p1) + log_x_t_real,
                    log_beta_t + log_1_min_p1 + log_1_min_x_t_real)

        log_xtmin1_eq_0_given_x_start = log_add_exp(log_1_min_cumprod_alpha_tmin1+log_1_min_p1, log_cumprod_alpha_tmin1 + log_1_min_x_start_real)
        log_xtmin1_eq_1_given_x_start = log_add_exp(log_cumprod_alpha_tmin1 + log_x_start_real, log_1_min_cumprod_alpha_tmin1+log_p1)

        log_xt_eq_0_given_xt_x_start = log_xtmin1_eq_0_given_x_t + log_xtmin1_eq_0_given_x_start
        log_xt_eq_1_given_xt_x_start = log_xtmin1_eq_1_given_x_t + log_xtmin1_eq_1_given_x_start

        unnorm_log_probs = torch.stack([log_xt_eq_0_given_xt_x_start, log_xt_eq_1_given_xt_x_start], dim=1)
        log_EV_xtmin_given_xt_given_xstart = unnorm_log_probs - unnorm_log_probs.logsumexp(1, keepdim=True)
        return log_EV_xtmin_given_xt_given_xstart


    def _predict_x0_or_xtmin1(self, batched_graph, t_node, t_edge):
        out_node, out_edge = self._denoise_fn(batched_graph, t_node, t_edge)

        assert out_node.size(1) == self.num_node_classes
        assert out_edge.size(1) == self.num_edge_classes

        log_pred_node = F.log_softmax(out_node, dim=1)
        log_pred_edge = F.log_softmax(out_edge, dim=1)
        return log_pred_node, log_pred_edge

    def _ce_prior(self, batched_graph):
        ones_node = torch.ones(batched_graph.nodes_per_graph.sum(), device=self.device).long()
        ones_edge = torch.ones(batched_graph.edges_per_graph.sum(), device=self.device).long()

        log_qxT_prob_node, log_qxT_prob_edge = self._q_pred(batched_graph, t_node=(self.num_timesteps - 1) * ones_node, 
                                                t_edge=(self.num_timesteps - 1) * ones_edge)

        log_final_prob_node = self.log_final_prob_node * torch.ones_like(log_qxT_prob_node)
        log_final_prob_edge = self.log_final_prob_edge * torch.ones_like(log_qxT_prob_edge)


        ce_prior_node = -log_categorical(log_qxT_prob_node, log_final_prob_node)
        ce_prior_node = scatter(ce_prior_node, batched_graph.batch, dim=-1, reduce='sum')

        ce_prior_edge = -log_categorical(log_qxT_prob_edge, log_final_prob_edge)
        ce_prior_edge = scatter(ce_prior_edge, batched_graph.batch[batched_graph.full_edge_index[0]], dim=-1, reduce='sum')

        ce_prior = ce_prior_node + ce_prior_edge

        return ce_prior

    def _kl_prior(self, batched_graph):

        ones_node = torch.ones(batched_graph.nodes_per_graph.sum(), device=self.device).long()
        ones_edge = torch.ones(batched_graph.edges_per_graph.sum(), device=self.device).long()

        log_qxT_prob_node, log_qxT_prob_edge = self._q_pred(batched_graph, t_node=(self.num_timesteps - 1) * ones_node, 
                                                t_edge=(self.num_timesteps - 1) * ones_edge)

        log_final_prob_node = self.log_final_prob_node * torch.ones_like(log_qxT_prob_node)
        log_final_prob_edge = self.log_final_prob_edge * torch.ones_like(log_qxT_prob_edge)


        kl_prior_node = self.multinomial_kl(log_qxT_prob_node, log_final_prob_node)
        kl_prior_node = scatter(kl_prior_node, batched_graph.batch, dim=-1, reduce='sum')

        kl_prior_edge = self.multinomial_kl(log_qxT_prob_edge, log_final_prob_edge)
        kl_prior_edge = scatter(kl_prior_edge, batched_graph.batch[batched_graph.full_edge_index[0]], dim=-1, reduce='sum')

        kl_prior = kl_prior_node + kl_prior_edge

        return kl_prior
   
    def _compute_MC_KL(self, batched_graph, t_edge, t_node):
        log_model_prob_node, log_model_prob_edge = self._p_pred(batched_graph=batched_graph, t_node=t_node, t_edge=t_edge)
        log_true_prob_node, log_true_prob_edge = self._q_pred_one_timestep(batched_graph=batched_graph, t_node=t_node, t_edge=t_edge)

        cross_ent_node = -log_categorical(batched_graph.log_node_attr_tmin1, log_model_prob_node)
        cross_ent_edge = -log_categorical(batched_graph.log_full_edge_attr_tmin1, log_model_prob_edge)


        ent_node = log_categorical(batched_graph.log_node_attr_t, log_true_prob_node).detach()
        ent_edge = log_categorical(batched_graph.log_full_edge_attr_t, log_true_prob_edge).detach()

        loss_node = cross_ent_node + ent_node
        loss_edge = cross_ent_edge + ent_edge

        loss_node = scatter(loss_node, batched_graph.batch, dim=-1, reduce='sum')
        loss_edge = scatter(loss_edge, batched_graph.batch[batched_graph.full_edge_index[0]], dim=-1, reduce='sum') 
        loss = loss_node + loss_edge

        return loss

    def _compute_RB_KL(self, batched_graph, t, t_edge, t_node):
        log_true_prob_node = self._q_posterior(log_x_start=batched_graph.log_node_attr, 
                                            log_x_t=batched_graph.log_node_attr_t, t=t_node, log_final_prob=self.log_final_prob_node)
        log_true_prob_edge = self._q_posterior(log_x_start=batched_graph.log_full_edge_attr, 
                                            log_x_t=batched_graph.log_full_edge_attr_t, t=t_edge, log_final_prob=self.log_final_prob_edge)
        log_model_prob_node, log_model_prob_edge = self._p_pred(batched_graph=batched_graph, t_node=t_node, t_edge=t_edge) 

        kl_node = self.multinomial_kl(log_true_prob_node, log_model_prob_node)
        kl_node = scatter(kl_node, batched_graph.batch, dim=-1, reduce='sum')
        kl_edge = self.multinomial_kl(log_true_prob_edge, log_model_prob_edge)
        kl_edge = scatter(kl_edge, batched_graph.batch[batched_graph.full_edge_index[0]], dim=-1, reduce='sum')
        kl = kl_node + kl_edge

        decoder_nll_node = -log_categorical(batched_graph.log_node_attr, log_model_prob_node)
        decoder_nll_node = scatter(decoder_nll_node, batched_graph.batch, dim=-1, reduce='sum')
        decoder_nll_edge = -log_categorical(batched_graph.log_full_edge_attr, log_model_prob_edge)
        decoder_nll_edge = scatter(decoder_nll_edge, batched_graph.batch[batched_graph.full_edge_index[0]], dim=-1, reduce='sum')
        decoder_nll = decoder_nll_node + decoder_nll_edge

        mask = (t == torch.zeros_like(t)).float()
        loss = mask * decoder_nll + (1. - mask) * kl

        return loss

    def _q_sample_and_set_xt_given_x0(self, batched_graph, t_node, t_edge):
        batched_graph.log_node_attr = index_to_log_onehot(batched_graph.node_attr, self.num_node_classes)
        batched_graph.log_full_edge_attr = index_to_log_onehot(batched_graph.full_edge_attr, self.num_edge_classes)
        
        log_node_attr_t, log_full_edge_attr_t = self.q_sample(batched_graph, t_node, t_edge)
        
        batched_graph.log_node_attr_t = log_node_attr_t
        batched_graph.log_full_edge_attr_t = log_full_edge_attr_t 
        

    def _q_sample_and_set_xtmin1_xt_given_x0(self, batched_graph, t_node, t_edge):
        batched_graph.log_node_attr = index_to_log_onehot(batched_graph.node_attr, self.num_node_classes)
        batched_graph.log_full_edge_attr = index_to_log_onehot(batched_graph.full_edge_attr, self.num_edge_classes)
        
        # sample xt-1
        tmin1_node = t_node - 1
        tmin1_edge = t_edge - 1
        tmin1_node_clamped = torch.where(tmin1_node < 0, torch.zeros_like(tmin1_node), tmin1_node)
        tmin1_edge_clamped = torch.where(tmin1_edge < 0, torch.zeros_like(tmin1_edge), tmin1_edge)
        
        log_node_attr_tmin1, log_full_edge_attr_tmin1 = self.q_sample(batched_graph, tmin1_node_clamped, tmin1_edge_clamped)
        batched_graph.log_node_attr_tmin1 = log_node_attr_tmin1
        batched_graph.log_full_edge_attr_tmin1 = log_full_edge_attr_tmin1

        batched_graph.log_node_attr_tmin1[tmin1_node<0] = batched_graph.log_node_attr[tmin1_node<0]
        batched_graph.log_full_edge_attr_tmin1[tmin1_edge<0] = batched_graph.log_full_edge_attr[tmin1_edge<0]

        # sample xt given xt-1
        log_node_attr_t, log_full_edge_attr_t = self._q_sample_one_timestep(batched_graph, t_node, t_edge)
        batched_graph.log_node_attr_t = log_node_attr_t
        batched_graph.log_full_edge_attr_t = log_full_edge_attr_t


    def _q_sample_one_timestep(self, batched_graph, t_node, t_edge):
        log_prob_node, log_prob_edge = self._q_pred_one_timestep(batched_graph, t_node, t_edge)

        log_out_node = self.log_sample_categorical(log_prob_node, self.num_node_classes)

        log_out_edge = self.log_sample_categorical(log_prob_edge, self.num_edge_classes)

        return log_out_node, log_out_edge  

        # xt ~ q(xt|xtmin1)

    def _q_pred_one_timestep(self, batched_graph, t_node, t_edge):
        log_alpha_t_node = extract(self.log_alpha, t_node, batched_graph.log_node_attr.shape)
        log_1_min_alpha_t_node = extract(self.log_1_min_alpha, t_node, batched_graph.log_node_attr.shape)

        log_prob_nodes = log_add_exp(
            batched_graph.log_node_attr_tmin1 + log_alpha_t_node,
            log_1_min_alpha_t_node + self.log_final_prob_node
        )

        log_alpha_t_edge = extract(self.log_alpha, t_edge, batched_graph.log_full_edge_attr.shape)
        log_1_min_alpha_t_edge = extract(self.log_1_min_alpha, t_edge, batched_graph.log_full_edge_attr.shape)

        log_prob_edges = log_add_exp(
            batched_graph.log_full_edge_attr_tmin1 + log_alpha_t_edge,
            log_1_min_alpha_t_edge + self.log_final_prob_edge
        )

        return log_prob_nodes, log_prob_edges 

    def _calc_num_entries(self, batched_graph):
        return batched_graph.full_edge_attr.shape[0] + batched_graph.node_attr.shape[0]

    def _sample_time(self, b, device, method='uniform'):
        if method == 'importance':
            if not (self.Lt_count > 10).all():
                return self._sample_time(b, device, method='uniform')

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # Overwrite decoder term with L1.
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)

            pt = pt_all.gather(dim=0, index=t)
            return t, pt

        elif method == 'uniform':
            length = self.Lt_count.shape[0]
            t = torch.randint(0, length, (b,), device=device).long()

            pt = torch.ones_like(t).float() / length
            return t, pt
        else:
            raise ValueError 
    

    def _q_pred(self, batched_graph, t_node, t_edge):
        # nodes prob
        log_cumprod_alpha_t_node = extract(self.log_cumprod_alpha, t_node, batched_graph.log_node_attr.shape)
        log_1_min_cumprod_alpha_node = extract(self.log_1_min_cumprod_alpha, t_node, batched_graph.log_node_attr.shape)
        log_prob_nodes = log_add_exp(
            batched_graph.log_node_attr + log_cumprod_alpha_t_node, 
            log_1_min_cumprod_alpha_node + self.log_final_prob_node 
        ) 

        # edges prob
        log_cumprod_alpha_t_edge = extract(self.log_cumprod_alpha, t_edge, batched_graph.log_full_edge_attr.shape)
        log_1_min_cumprod_alpha_edge = extract(self.log_1_min_cumprod_alpha, t_edge, batched_graph.log_full_edge_attr.shape)
        log_prob_edges = log_add_exp(
            batched_graph.log_full_edge_attr + log_cumprod_alpha_t_edge,
            log_1_min_cumprod_alpha_edge + self.log_final_prob_edge
        )
        return log_prob_nodes, log_prob_edges 


    def _p_pred(self, batched_graph, t_node, t_edge):
        if self.parametrization == 'x0':
            log_node_recon, log_full_edge_recon = self._predict_x0_or_xtmin1(batched_graph, t_node=t_node, t_edge=t_edge)
            log_model_pred_node = self._q_posterior(
                log_x_start=log_node_recon, log_x_t=batched_graph.log_node_attr_t, t=t_node, log_final_prob=self.log_final_prob_node)
            log_model_pred_edge = self._q_posterior(
                log_x_start=log_full_edge_recon, log_x_t=batched_graph.log_full_edge_attr_t, t=t_edge, log_final_prob=self.log_final_prob_edge)
        elif self.parametrization == 'xt':
            log_model_pred_node, log_model_pred_edge = self._predict_x0_or_xtmin1(batched_graph, t_node=t_node, t_edge=t_edge) 
        return log_model_pred_node, log_model_pred_edge


    def _prepare_data_for_sampling(self, batched_graph):
        batched_graph.log_node_attr = index_to_log_onehot(batched_graph.node_attr, self.num_node_classes)
        log_prob_node = torch.ones_like(batched_graph.log_node_attr, device=self.device) * self.log_final_prob_node
        batched_graph.log_node_attr_t = self.log_sample_categorical(log_prob_node, self.num_node_classes)
        

        batched_graph.log_full_edge_attr = index_to_log_onehot(batched_graph.full_edge_attr, self.num_edge_classes)

        log_prob_edge = torch.ones_like(batched_graph.log_full_edge_attr, device=self.device) * self.log_final_prob_edge

      
        batched_graph.log_full_edge_attr_t = self.log_sample_categorical(log_prob_edge, self.num_edge_classes)

        return batched_graph

    def _eval_loss(self, batched_graph):
        b = batched_graph.num_graphs
        batched_graph.num_entries = self._calc_num_entries(batched_graph)
        if self.loss_type == 'vb_kl':
            t, pt = self._sample_time(b, self.device, self.sample_time_method)
            
            t_node = t.repeat_interleave(batched_graph.nodes_per_graph)
            t_edge = t.repeat_interleave(batched_graph.edges_per_graph)
            
            self._q_sample_and_set_xt_given_x0(batched_graph, t_node, t_edge)

            kl = self._compute_RB_KL(batched_graph, t, t_edge, t_node)
            kl_prior = self._kl_prior(batched_graph=batched_graph)
            # Upweigh loss term of the kl
            loss = kl / pt + kl_prior
            return -loss
        
        elif self.loss_type == 'vb_ce_x0':
            assert self.parametrization == 'x0'
            pass #TODO not in the scope of the current submission

        elif self.loss_type == 'vb_ce_xt':
            assert self.parametrization == 'xt'

            t, pt =  self._sample_time(b, self.device, self.sample_time_method)

            t_node = t.repeat_interleave(batched_graph.nodes_per_graph)
            t_edge = t.repeat_interleave(batched_graph.edges_per_graph)

            self._q_sample_and_set_xtmin1_xt_given_x0(batched_graph, t_node, t_edge)

            kl = self._compute_MC_KL(batched_graph, t_edge, t_node)

            ce_prior = self._ce_prior(batched_graph=batched_graph)
            # Upweigh loss term of the kl
            vb_loss = kl / pt + ce_prior
            return -vb_loss 

       
        else:
            raise ValueError()


    def _train_loss(self, batched_graph):
        b = batched_graph.num_graphs
        batched_graph.num_entries = self._calc_num_entries(batched_graph) 
        if self.loss_type == 'vb_kl':             
            # not sure it is ok to allow the parameterization to be xt, which is also sensible in math, for now it must be x0
            assert self.parametrization == 'x0'
            # sample t for each graph
            t, pt = self._sample_time(b, self.device, self.sample_time_method)

            t_node = t.repeat_interleave(batched_graph.nodes_per_graph)
            t_edge = t.repeat_interleave(batched_graph.edges_per_graph)
            
            self._q_sample_and_set_xt_given_x0(batched_graph, t_node, t_edge)

            kl = self._compute_RB_KL(batched_graph, t, t_edge, t_node)

            Lt2 = kl.pow(2)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

            kl_prior = self._kl_prior(batched_graph=batched_graph)

            # Upweigh loss term of the kl
            vb_loss = kl / pt + kl_prior
            return -vb_loss
        
        
        elif self.loss_type == 'vb_ce_x0':
            assert self.parametrization == 'x0'
            pass # TODO 


        elif self.loss_type == 'vb_ce_xt':
            assert self.parametrization == 'xt'

            t, pt =  self._sample_time(b, self.device, self.sample_time_method)

            t_node = t.repeat_interleave(batched_graph.nodes_per_graph)
            t_edge = t.repeat_interleave(batched_graph.edges_per_graph)
            
            self._q_sample_and_set_xtmin1_xt_given_x0(batched_graph, t_node, t_edge)


            kl = self._compute_MC_KL(batched_graph, t_edge, t_node)

            Lt2 = kl.pow(2)
            Lt2_prev = self.Lt_history.gather(dim=0, index=t)
            new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
            self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
            self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

            ce_prior = self._ce_prior(batched_graph=batched_graph)
            # Upweigh loss term of the kl
            vb_loss = kl / pt + ce_prior

            return -vb_loss 


        else:
            raise ValueError()
    