import torch
import os
import pickle
from diffusion.loss import elbo_bpd
from diffusion.utils import add_parent_path
from scipy import sparse as sp
import torch_geometric as pyg
import networkx as nx
import matplotlib.pyplot as plt

add_parent_path(level=2)
from diffusion.experiment import DiffusionExperiment
from diffusion.experiment import add_exp_args as add_exp_args_parent


def add_exp_args(parser):
    add_exp_args_parent(parser)
    parser.add_argument('--clip_value', type=float, default=None)
    parser.add_argument('--clip_norm', type=float, default=None)
    parser.add_argument('--num_generation', type=int, default=64)

class GraphExperiment(DiffusionExperiment):
    def train_fn(self, epoch):
        self.model.train()
        loss_sum = 0.0
        loss_count = 0
        data_count = 0
        for pyg_data in self.train_loader:
            self.optimizer.zero_grad()
            pyg_data = pyg_data.to(self.args.device)
            # pyg_data.num_entries = self.model._calc_num_entries(pyg_data)
            loss = elbo_bpd(self.model, pyg_data)
            loss.backward()
            
            if self.args.clip_value: torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip_value)
            if self.args.clip_norm: torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_norm)
            self.optimizer.step()

            if self.scheduler_iter: self.scheduler_iter.step()
            loss_sum += loss.detach().cpu().item() * pyg_data.num_graphs
            loss_count += pyg_data.num_graphs
            data_count += pyg_data.num_graphs#pyg_data.num_graphs
            print('Training. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, self.args.epochs, data_count, len(self.train_loader.dataset), loss_sum/loss_count), end='\r')
            # self.model.complex_data = None
        if self.scheduler_epoch: self.scheduler_epoch.step()
        return {'bpd': loss_sum / loss_count, 'lr': self.optimizer.param_groups[0]['lr']}

    def eval_fn(self, epoch):
        self.model.eval()
        eval_dict = {}
        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            data_count = 0

            for pyg_data in self.eval_loader:
                pyg_data = pyg_data.to(self.args.device)
                # pyg_data.num_entries = self.model._calc_num_entries(pyg_data)
                loss = elbo_bpd(self.model, pyg_data) 
                loss_sum += loss.detach().cpu().item() * pyg_data.num_graphs#len(x)
                loss_count += pyg_data.num_graphs #len(x)
                data_count += pyg_data.num_graphs #pyg_data.num_graphs

            print('Train evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, self.args.epochs, data_count, len(self.eval_loader.dataset), loss_sum/loss_count), end='\r')            
            eval_dict['bpd'] = loss_sum/loss_count

            generated_pyg_datas = self.model.sample(self.args.num_generation)
            generated_graphs = []
            
            pyg_data_list = generated_pyg_datas.to_data_list()
            for pyg_data in pyg_data_list:
                # assert pyg_data.edge_index.shape[1]%2==0
                # assert pyg_data.edge_index.shape[0]%2==0
                g_gen = pyg.utils.to_networkx(pyg_data, to_undirected=True)
                generated_graphs.append(g_gen)

            w = 8 if self.args.num_generation >= 64 else 2
            fig, axes = plt.subplots(w, w, figsize=(17,17))
            for i, g_gen in enumerate(generated_graphs[:w**2]):
                nx.draw(g_gen, ax=axes[i%w][i//w], node_size=30)

            plt.savefig(os.path.join(self.log_path, f"eval/sample{epoch}.png"))
            plt.close()

            # statistics evaluation
            metrics = self.eval_evaluator.evaluate(generated_graphs)
            eval_dict.update(metrics)

        return eval_dict

    def test_fn(self, epoch):
        self.model.eval()
        test_dict = {}
        with torch.no_grad():
            loss_sum = 0.0
            loss_count = 0
            data_count = 0

            for pyg_data in self.test_loader:
                pyg_data = pyg_data.to(self.args.device)
                # pyg_data.num_entries = self.model._calc_num_entries(pyg_data)
                loss = elbo_bpd(self.model, pyg_data) 
                loss_sum += loss.detach().cpu().item() * pyg_data.num_graphs#len(x)
                loss_count += pyg_data.num_graphs #len(x)
                data_count += pyg_data.num_graphs #pyg_data.num_graphs
            print('Train evaluating. Epoch: {}/{}, Datapoint: {}/{}, Bits/dim: {:.3f}'.format(epoch+1, self.args.epochs, data_count, len(self.eval_loader.dataset), loss_sum/loss_count), end='\r')            
            test_dict['bpd'] = loss_sum/loss_count
            generated_pyg_datas = self.model.sample(self.args.num_generation)
            generated_graphs = []
            
            pyg_data_list = generated_pyg_datas.to_data_list()
            for pyg_data in pyg_data_list:
                g_gen = pyg.utils.to_networkx(pyg_data, to_undirected=True)
                generated_graphs.append(g_gen)

            w = 8 if self.args.num_generation >= 64 else 2
            fig, axes = plt.subplots(w, w, figsize=(17,17))
            for i, g_gen in enumerate(generated_graphs[:w**2]):
                nx.draw(g_gen, ax=axes[i%w][i//w], node_size=30)

            plt.savefig(os.path.join(self.log_path, f"test/sample{epoch}.png"))
            plt.close()

            # statistics evaluation
            metrics = self.test_evaluator.evaluate(generated_graphs)
            test_dict.update(metrics)

        return test_dict 