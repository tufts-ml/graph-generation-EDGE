import torch
import torch.nn.functional as F
import numpy as np
from inspect import isfunction
from torch_scatter import scatter
import torch_geometric as pyg
from diffusion.diffusion_base import cosine_beta_schedule, log_1_min_a, log_add_exp, log_categorical, index_to_log_onehot, extract
from diffusion.diffusion_binomial_vanilla import BinomialDiffusionVanilla
"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8


class BinomialDiffusionActive(BinomialDiffusionVanilla):
    def __init__(self, num_node_classes, num_edge_classes, initial_graph_sampler, denoise_fn, timesteps=1000,
                 loss_type='vb_kl', parametrization='x0', final_prob_node=None, final_prob_edge=None, sample_time_method='importance', 
                 noise_schedule=cosine_beta_schedule, device='cuda'):
        super(BinomialDiffusionActive, self).__init__(num_node_classes, num_edge_classes, initial_graph_sampler, denoise_fn, timesteps,
                 loss_type, parametrization, final_prob_node, final_prob_edge, sample_time_method, noise_schedule, device)

    def sample_increment(self, num_samples):
        # some bugs are in here, do not use for now.
        raise NotImplementedError
        batched_graph = self.initial_graph_sampler.sample(num_samples)
        batched_graph.to(self.device)

        num_nodes = batched_graph.nodes_per_graph.sum()
        batched_graph = self._prepare_data_for_sampling(batched_graph)

        edge_attr_t = batched_graph.log_full_edge_attr_t.argmax(-1)
        is_edge_indices_t = edge_attr_t.nonzero(as_tuple=True)[0]
        batched_graph.edge_index_t = batched_graph.full_edge_index[:, is_edge_indices_t]
        print()
        for t in reversed(range(0, self.num_timesteps)):
            print(f'Sample timestep {t:4d}', end='\r')
            t_node = torch.full((num_nodes,), t, device=self.device, dtype=torch.long)
            t_edge = None
            # p_sample variants
            degree_t = self._compute_degree(torch.ones_like(batched_graph.edge_index_t[0]), batched_graph.edge_index_t, batched_graph.num_nodes)
            log_model_prob_active = self._q_posterior_actives(batched_graph.degree, degree_t, t_node)
            active_node_masks = self.log_sample_categorical(log_model_prob_active, num_classes=2).argmax(1).bool()
            batched_graph.active_node_indices = active_node_masks.nonzero(as_tuple=True)[0]
            batched_graph.active_edge_indices = active_node_masks[batched_graph.full_edge_index[0]].logical_and(
            active_node_masks[batched_graph.full_edge_index[1]]).nonzero(as_tuple=True)[0] 
            
            if batched_graph.active_edge_indices.size(0) == 0:
                continue

            _, log_model_prob_edge = self._p_pred(batched_graph, t_node, t_edge)

            assert log_model_prob_edge.size(0) == batched_graph.active_edge_indices.size(0)

            log_out_edge_active = self.log_sample_categorical(log_model_prob_edge, self.num_edge_classes)

            row = batched_graph.full_edge_index[0].index_select(0, batched_graph.active_edge_indices[log_out_edge_active.argmax(-1).bool()])
            col = batched_graph.full_edge_index[1].index_select(0, batched_graph.active_edge_indices[log_out_edge_active.argmax(-1).bool()])
            sampled_edge_indices_tmin1 = torch.stack((row,col))
            edge_index_t = torch.cat((batched_graph.edge_index_t, sampled_edge_indices_tmin1), dim=-1)
            batched_graph.edge_index_t = pyg.utils.coalesce(edge_index_t, reduce='max') 
        print()

        batched_graph.edge_index = batched_graph.edge_index_t 
        edge_slice = batched_graph.batch[batched_graph.edge_index[0]]
        edge_slice = scatter(torch.ones_like(edge_slice), edge_slice, dim_size=batched_graph.num_graphs)
        edge_slice = torch.nn.functional.pad(edge_slice, (1,0), 'constant', 0)
        edge_slice = torch.cumsum(edge_slice, 0)
        batched_graph._slice_dict['edge_index'] = edge_slice
        batched_graph._inc_dict['edge_index'] = batched_graph._inc_dict['full_edge_index']
        return batched_graph
         
    @torch.no_grad()
    def p_sample(self, batched_graph, t_node, t_edge):
        self._p_sample_and_set_actives(batched_graph, t_node)
        assert hasattr(batched_graph, 'active_node_indices')
        assert hasattr(batched_graph, 'active_edge_indices')
        if batched_graph.active_edge_indices.size(0) == 0:
            return batched_graph.log_node_attr_t, batched_graph.log_full_edge_attr_t
        log_model_prob_node, log_model_prob_edge = self._p_pred(batched_graph, t_node, t_edge)

        assert log_model_prob_edge.size(0) == batched_graph.active_edge_indices.size(0)

        log_out_node = self.log_sample_categorical(log_model_prob_node, self.num_node_classes)
        log_out_edge_active = self.log_sample_categorical(log_model_prob_edge, self.num_edge_classes)

        log_out_edge = batched_graph.log_full_edge_attr_t

        log_out_edge[batched_graph.active_edge_indices] = log_out_edge_active

        return log_out_node, log_out_edge
    
    def _compute_degree(self, full_edge_attr, full_edge_index, num_nodes):
        degree = scatter(full_edge_attr, full_edge_index[0], dim=0, dim_size=num_nodes) +\
                scatter(full_edge_attr, full_edge_index[1], dim=0, dim_size=num_nodes) 
        return degree

    def _q_posterior_actives(self, degree_start, degree_t, t_node):
        tmin1 = t_node - 1
        tmin1 = torch.where(tmin1 < 0, torch.zeros_like(tmin1), tmin1)

        log_beta_t = extract(self.log_1_min_alpha, t_node, degree_start.shape)
        
        log_cumprod_alpha_t_min_1 = extract(self.log_cumprod_alpha, tmin1, degree_start.shape)

        log_1_min_cumprod_alpha_t = extract(self.log_1_min_cumprod_alpha, t_node, degree_start.shape)

        logprob_edge_t = log_beta_t + log_cumprod_alpha_t_min_1 - log_1_min_cumprod_alpha_t

        logit_edge_t = logprob_edge_t - log_1_min_a(logprob_edge_t)

        n_trials = torch.max(degree_start-degree_t, torch.zeros_like(degree_start))
        
        logprob_node_nochange_t = torch.distributions.Binomial(total_count=n_trials, logits=logit_edge_t).log_prob(torch.zeros_like(degree_start))
        logprob_node_change_t = log_1_min_a(logprob_node_nochange_t)

        unnorm_log_probs = torch.stack([logprob_node_nochange_t, logprob_node_change_t], dim=1)
        log_node_change_given_dt_given_dstart = unnorm_log_probs - unnorm_log_probs.logsumexp(1, keepdim=True) 

        return log_node_change_given_dt_given_dstart

    def _p_sample_and_set_actives(self, batched_graph, t_node):
        if self.parametrization == 'xt_prescribed_st':
            degree_t = self._compute_degree(batched_graph.log_full_edge_attr_t.argmax(1), batched_graph.full_edge_index, batched_graph.num_nodes)
            log_model_prob_active = self._q_posterior_actives(batched_graph.degree, degree_t, t_node)
            active_node_masks = self.log_sample_categorical(log_model_prob_active, num_classes=2).argmax(1).bool()
            batched_graph.active_node_indices = active_node_masks.nonzero(as_tuple=True)[0]
            batched_graph.active_edge_indices = active_node_masks[batched_graph.full_edge_index[0]].logical_and(
            active_node_masks[batched_graph.full_edge_index[1]]).nonzero(as_tuple=True)[0] 
        elif self.parametrization == 'xt_st': 
            pass #TODO
        else:
            raise NotImplementedError

    def _q_set_actives(self, batched_graph):
        degree_tmin1 = self._compute_degree(batched_graph.log_full_edge_attr_tmin1.argmax(1), batched_graph.full_edge_index, batched_graph.num_nodes)
        degree_t = self._compute_degree(batched_graph.log_full_edge_attr_t.argmax(1), batched_graph.full_edge_index, batched_graph.num_nodes)
       
        # set up active node indices, if K nodes are active, the length of active_nodes_indices is K
        active_node_masks = degree_tmin1 > degree_t
        batched_graph.active_node_indices = active_node_masks.nonzero(as_tuple=True)[0]
        # set up active edge indices, if K nodes are active, the length of active_edges_indices is K * (K-1) // 2
        batched_graph.active_edge_indices = active_node_masks[batched_graph.full_edge_index[0]].logical_and(
            active_node_masks[batched_graph.full_edge_index[1]]).nonzero(as_tuple=True)[0]
        batched_graph.edge_predict_masks = active_node_masks[batched_graph.full_edge_index[0]].logical_and(
            active_node_masks[batched_graph.full_edge_index[1]])

    def _predict_xtmin1_given_xt_st(self, batched_graph, t_node, t_edge):
        out_node, out_edge = self._denoise_fn(batched_graph, t_node, t_edge)

        assert out_node.size(1) == self.num_node_classes
        assert out_edge.size(1) == self.num_edge_classes

        log_pred_node = F.log_softmax(out_node, dim=1)
        log_pred_edge = F.log_softmax(out_edge, dim=1)
        return log_pred_node, log_pred_edge

    def _compute_MC_KL_joint(self, batched_graph, t, t_node, t_edge):
        log_model_prob_node, log_model_prob_edge = self._p_pred(batched_graph=batched_graph, t_node=t_node, t_edge=t_edge)

        active_edge_attr_tmin1 = batched_graph.log_full_edge_attr_tmin1.index_select(0, batched_graph.active_edge_indices)
        
        loss_node = 0#scatter(loss_node, batched_graph.batch, dim=-1, reduce='sum')


        cross_ent_edge = -log_categorical(active_edge_attr_tmin1, log_model_prob_edge)
       


        cross_ent_edge = scatter(cross_ent_edge, batched_graph.batch[batched_graph.full_edge_index[0].index_select(0, batched_graph.active_edge_indices)], dim=-1, reduce='sum', dim_size=batched_graph.num_graphs)

        # recover constant term
        num_actives_edge_per_graphs = scatter(torch.ones_like(batched_graph.active_edge_indices), batched_graph.batch[batched_graph.full_edge_index[0].index_select(0, batched_graph.active_edge_indices)], dim=-1, reduce='sum', dim_size=batched_graph.num_graphs)
        num_nodes_per_graphs = scatter(torch.ones(batched_graph.num_nodes, device=self.device), batched_graph.batch)
        num_nodes_per_graphs*(num_nodes_per_graphs-1)//2
        num_inactive_edges_per_graph = num_nodes_per_graphs*(num_nodes_per_graphs-1)//2 - num_actives_edge_per_graphs
        cross_ent_edge += 6.9078e-29 * num_inactive_edges_per_graph
        ent_edge = 6.9078e-29 * batched_graph.edges_per_graph
        loss_edge = cross_ent_edge + ent_edge

        loss = loss_node + loss_edge
        return loss    

    def _p_pred(self, batched_graph, t_node, t_edge):
        if self.parametrization in ['x0', 'xt']:
            return super(BinomialDiffusionActive, self)._p_pred(batched_graph, t_node, t_edge)
        elif self.parametrization == 'xt_prescribed_st':
            log_model_pred_node, log_model_pred_edge = self._predict_xtmin1_given_xt_st(batched_graph, t_node=t_node, t_edge=t_edge) 
            return log_model_pred_node, log_model_pred_edge
        elif self.parametrization == 'xt_st':
            pass # TODO

    def _calc_num_entries(self, batched_graph):
        return batched_graph.full_edge_attr.shape[0]# + batched_graph.node_attr.shape[0]

    def _eval_loss(self, batched_graph):
        if self.loss_type in ['vb_kl', 'vb_ce_xt']:
            # this is the same as vanilla since variable st is not introduced
            return super(BinomialDiffusionActive, self)._eval_loss(batched_graph)
        else:
            b = batched_graph.num_graphs
            if self.loss_type == 'vb_ce_xt_kl_st':
                pass

            elif self.loss_type == 'vb_ce_xt_ce_st':
                pass
            
            elif self.loss_type == 'vb_ce_xt_prescribred_st':
                t, pt =  self._sample_time(b, self.device, self.sample_time_method)

                t_node = t.repeat_interleave(batched_graph.nodes_per_graph)
                t_edge = t.repeat_interleave(batched_graph.edges_per_graph)

                self._q_sample_and_set_xtmin1_xt_given_x0(batched_graph, t_node, t_edge)
                self._q_set_actives(batched_graph)

                kl = self._compute_MC_KL_joint(batched_graph, t, t_node, t_edge)

                ce_prior = self._kl_prior(batched_graph=batched_graph)
                # Upweigh loss term of the kl
                vb_loss = kl / pt + ce_prior
                batched_graph.num_entries = self._calc_num_entries(batched_graph)

                return -vb_loss 

            else:
                raise ValueError()

    def _train_loss(self, batched_graph):
        if self.loss_type in ['vb_kl', 'vb_ce_xt']:
            return super(BinomialDiffusionActive, self)._train_loss(batched_graph)
        else:
            b = batched_graph.num_graphs
            if self.loss_type == 'vb_ce_xt_kl_st':
                pass # TODO

            elif self.loss_type == 'vb_ce_xt_ce_st':
                pass # TODO
            
            elif self.loss_type == 'vb_ce_xt_prescribred_st':
                t, pt =  self._sample_time(b, self.device, self.sample_time_method)

                t_node = t.repeat_interleave(batched_graph.nodes_per_graph)
                t_edge = t.repeat_interleave(batched_graph.edges_per_graph)
                self._q_sample_and_set_xtmin1_xt_given_x0(batched_graph, t_node, t_edge)

                self._q_set_actives(batched_graph)

                kl = self._compute_MC_KL_joint(batched_graph, t, t_node, t_edge)

                Lt2 = kl.pow(2)
                Lt2_prev = self.Lt_history.gather(dim=0, index=t)
                new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
                self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
                self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))

                ce_prior = self._kl_prior(batched_graph=batched_graph)# TODO replaced it back to _ce_prior
                # Upweigh loss term of the kl
                vb_loss = kl / pt + ce_prior

                batched_graph.num_entries = self._calc_num_entries(batched_graph)

                return -vb_loss
            else:
                raise ValueError()
    