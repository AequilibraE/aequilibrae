import numpy as np
from typing import List
from aequilibrae.paths.assignment_class import AssignmentClass
from aequilibrae.paths.all_or_nothing import allOrNothing
from aequilibrae.paths.results import AssignmentResults
from aequilibrae import Parameters
from aequilibrae.paths.vdf import VDF
from aequilibrae import logger

class MSA:
    def __init__(self, traffic_classes: List[AssignmentClass]):
        parameters = Parameters().parameters['assignment']['equilibrium']
        self.rgap_target = parameters['rgap']
        self.max_iter = parameters['maximum_iterations']

        # A single class for now
        self.graph = traffic_classes[0].graph
        self.matrix = traffic_classes[0].matrix
        self.final_results = traffic_classes[0].results

        self.aon_results = AssignmentResults()
        self.aon_results.prepare(self.graph, self.matrix)
        self.iter = 0
        self.rgap = np.inf
        self.vdf = VDF()

    def execute(self):
        logger.info('MSA Assignment STATS')
        logger.info('Iteration,RelativeGap')
        for self.iter in range(1, self.max_iter + 1):
            aon = allOrNothing(self.matrix, self.graph, self.aon_results)
            aon.execute()

            self.final_results.link_loads[:, :] = self.final_results.link_loads[:, :] * ((float(self.iter) - 1.0) / float(self.iter))
            self.final_results.link_loads[:, :] += self.aon_results.link_loads[:, :] * (1.0 / float(self.iter))

            self.msa_class_flow = np.sum(self.final_results.link_loads, axis=1)

            self.congested_time = self.vdf.apply_vdf("BPR",link_flows=self.msa_class_flow, capacity=self.graph.capacity,
                                                     fftime=self.graph.free_flow_time)
            self.graph.cost = self.congested_time

            # Check convergence
            if self.check_convergence() and self.iter > 1:
                break
            self.aon_results.reset()
            logger.info('{},{}'.format(self.iter, self.rgap))

        if self.rgap > self.rgap_target:
            logger.error('Desired RGap of {} was NOT reached'.format(self.rgap_target))
        logger.info('MSA Assignment finished. {} iterations and {} final gap'.format(self.iter, self.rgap))

    def check_convergence(self):
        aon_class_flow = np.sum(self.aon_results.link_loads, axis=1)

        aon_cost = np.sum(self.congested_time * aon_class_flow)
        msa_cost = np.sum(self.congested_time * self.msa_class_flow)
        self.rgap = abs(msa_cost - aon_cost) / msa_cost
        print(self.iter, self.rgap)
        if self.rgap_target >= self.rgap:
            return True
        return False

