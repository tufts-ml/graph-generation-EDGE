from eval_utils.evaluation import graph_structure_evaluation
from eval_utils.evaluation import gin_evaluation


class Evaluator():
    def __init__(self, feature_extractor='gin', **kwargs):
        if feature_extractor != 'mmd-structure':

            model = gin_evaluation.load_feature_extractor(**kwargs)

            # Create individual evaluators for each GNN-based metric
            self.evaluators = []
            self.evaluators.append(gin_evaluation.FIDEvaluation(
                model=model))
            self.evaluators.append(gin_evaluation.KIDEvaluation(
                model=model))
            self.evaluators.append(gin_evaluation.prdcEvaluation(
                model=model, use_pr=True))
            self.evaluators.append(gin_evaluation.prdcEvaluation(
                model=model, use_pr=False))
            self.evaluators.append(gin_evaluation.MMDEvaluation(
                model=model, kernel='rbf', sigma='range', multiplier='mean'))
            self.evaluators.append(gin_evaluation.MMDEvaluation(
                model=model, kernel='linear'))

            try:
                num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
                self.logger.info(f'number of parameters: {num_parameters}')
            except:
                pass

        elif feature_extractor == 'mmd-structure' and kwargs.get('statistic') != 'WL' and kwargs.get('statistic') != 'nspdk':
            self.evaluators = [graph_structure_evaluation.MMDEval(**kwargs)]

        elif feature_extractor == 'mmd-structure' and kwargs.get('statistic') == 'WL':
            self.evaluators = [graph_structure_evaluation.WLMMDEvaluation()]
        elif feature_extractor == 'mmd-structure' and kwargs.get('statistic') == 'nspdk':
            self.evaluators = [graph_structure_evaluation.NSPDKEvaluation()]
        else:
            raise Exception('Unsupported feature extractor {} or statistic {}'.format(kwargs.get('feature_extractor'), kwargs.get('statistic')))

    def evaluate_all(
        self, generated_dataset=None, reference_dataset=None, **kwargs):
        metrics = {}
        if len(self.evaluators) > 2:
            (generated_dataset, reference_dataset), _ = self.evaluators[0].get_activations(generated_dataset, reference_dataset)
            # metrics['activations_time'] = time

        for evaluator in self.evaluators:
            try:
                res, time = evaluator.evaluate(
                    generated_dataset=generated_dataset,
                    reference_dataset=reference_dataset)
                metrics.update(res)
            except:
                pass

        del generated_dataset
        del reference_dataset
        return metrics

    @property
    def feature_extractor(self):
        return self.evaluators[0].feat_extractor
