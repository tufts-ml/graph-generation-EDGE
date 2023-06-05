import dgl
import numpy as np
import networkx as nx
from eval_utils.graph_statistics import compute_graph_statistics
from eval_utils.evaluation.evaluator import Evaluator
from eval_utils.evaluation.graph_structure_evaluation import MMDEval, NSPDKEvaluation

class GenericGraphEvaluator:
    def __init__(self, references_nx_graphs, device):
        self.references = [dgl.from_networkx(g) for g in references_nx_graphs]
        self.nerual_evaluator = Evaluator(device=device)
        self.struc_evaluators = {'degree': MMDEval(statistic='degree'), 
                                 'spectral': MMDEval(statistic='spectral'),
                                 'clustering': MMDEval(statistic='clustering'),
                                 'orbits': MMDEval(statistic='orbits'),
                                 'nspdk': NSPDKEvaluation()}
    def evaluate(self, target_nx_graphs):
        target_graphs = [dgl.from_networkx(g) for g in target_nx_graphs]
        metrics = self.nerual_evaluator.evaluate_all(self.references, target_graphs)
        metrics['degree_mmd'] = self.struc_evaluators['degree'].evaluate(self.references, target_graphs)[0]['degree_mmd']
        metrics['spectral_mmd'] = self.struc_evaluators['spectral'].evaluate(self.references, target_graphs)[0]['spectral_mmd']
        metrics['clustering_mmd'] = self.struc_evaluators['clustering'].evaluate(self.references, target_graphs)[0]['clustering_mmd']
        metrics['orbits_mmd'] = self.struc_evaluators['orbits'].evaluate(self.references, target_graphs)[0]['orbits_mmd']
        metrics['nspdk_mmd'] = self.struc_evaluators['nspdk'].evaluate(self.references, target_graphs)[0]['nspdk_mmd']
        return metrics

class NetworkEvaluator:
    def __init__(self, reference_nx_graph):
        self.reference = reference_nx_graph
        self.reference_stats = compute_graph_statistics(nx.to_scipy_sparse_array(self.reference))

    def evaluate(self, target_nx_graphs):
        metric_per_graphs = [compute_graph_statistics(nx.to_scipy_sparse_array(target_nx_graph)) for target_nx_graph in target_nx_graphs]
        # merge metrics and compute mean
        metrics = {f'nmae/{k}': abs((np.mean([m[k] for m in metric_per_graphs]) - self.reference_stats[k])/self.reference_stats[k]) for k in metric_per_graphs[0].keys()}
        metrics.update({f'value/{k}': np.mean([m[k] for m in metric_per_graphs]) for k in metric_per_graphs[0].keys()})
        return metrics
