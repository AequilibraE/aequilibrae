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

    def execute(self):
        for i in range(1, self.max_iter + 1):
            allOrNothing(self.matrix, self.graph, self.aon_results)
            allOrNothing.execute()

            self.aon_class_flow = np.sum(self.aon_results.link_loads, axis=1)

            self.congested_time = VDF(link_flows=self.aon_class_flow, capacity=self.graph.capacity,
                                      fftime=self.graph.free_flow_time)
            self.graph.cost = self.congested_time
            self.final_results.link_loads[:, :] = self.final_results.link_loads[:, :] * ((i - 1.0) / i)
            self.final_results.link_loads[:, :] += self.aon_results.link_loads[:, :] * (1.0 / i)

            # Check convergence
            if self.check_convergence() and self.iter > 1:
                break

        if self.rgap > self.rgap_target:
            logger.error('Desired RGap of {} was NOT reached'.format(self.rgap_target))
        logger.info('MSA Assignment finished. {} iterations and {} final gap'.format(self.iter, self.rgap))

    def check_convergence(self):
        class_flow = np.sum(self.final_results.link_loads, axis=1)
        msa_cost = np.sum(self.congested_time * class_flow)
        aon_cost = np.sum(self.congested_time * self.aon_class_flow)
        self.rgap = abs(msa_cost - aon_cost) / msa_cost
        if self.rgap_target >= self.rgap:
            return True
        return False

