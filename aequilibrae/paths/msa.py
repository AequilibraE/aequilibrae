import numpy as np
from typing import List
from aequilibrae.paths.traffic_class import TrafficClass
from aequilibrae.paths.all_or_nothing import allOrNothing
from aequilibrae.paths.results import AssignmentResults
from aequilibrae import Parameters
from aequilibrae.paths.vdf import VDF
from aequilibrae import logger

if False:
    from aequilibrae.paths.traffic_assignment import TrafficAssignment


class MSA:
    def __init__(self, assig_spec) -> None:
        parameters = Parameters().parameters['assignment']['equilibrium']
        self.rgap_target = parameters['rgap']
        self.max_iter = parameters['maximum_iterations']
        self.assig = assig_spec  # type: TrafficAssignment

        if None in [assig_spec.classes, assig_spec.vdf, assig_spec.capacity_field, assig_spec.time_field,
                    assig_spec.vdf_parameters]:
            raise Exception("Parameters missing. Setting the algorithm is the last thing to do when assigning")

        self.traffic_classes = assig_spec.classes  # type: List[TrafficAssignment]
        self.num_classes = len(assig_spec.classes)

        self.cap_field = assig_spec.capacity_field
        self.time_field = assig_spec.time_field
        self.vdf = assig_spec.vdf

        self.vdf_parameters = {}
        for k, v in assig_spec.vdf_parameters.items():
            if isinstance(v, str):
                self.vdf_parameters[k] = assig_spec.classes[0].graph.graph[k]
            else:
                self.vdf_parameters[k] = v

        self.iter = 0
        self.rgap = np.inf

    def execute(self):
        logger.info('MSA Assignment STATS for {} classes'.format(len(self.traffic_classes)))

        logger.info('Iteration,RelativeGap,BlendingFraction')
        for self.iter in range(1, self.max_iter + 1):
            flows = []
            aon_flows = []
            for c in self.traffic_classes:
                aon = allOrNothing(c.matrix, c.graph, c._aon_results)
                aon.execute()

                stepsize = 1.0 / float(self.iter)

                c.results.link_loads[:, :] = c.results.link_loads[:, :] * (1 - stepsize)
                c.results.link_loads[:, :] += c._aon_results.link_loads[:, :] * stepsize

                # We already get the total traffic class, in PCEs, corresponding to the total for the user classes
                flows.append(np.sum(c.results.link_loads, axis=1) * c.pce)
                aon_flows.append(np.sum(c._aon_results.link_loads, axis=1) * c.pce)

            self.msa_total_flow = np.sum(flows, axis=0)
            self.aon_total_flow = np.sum(aon_flows, axis=0)

            pars = {'link_flows': self.msa_total_flow, 'capacity': c.graph.graph[self.cap_field],
                    'fftime': c.graph.graph[self.time_field]}

            # Check convergence
            if self.iter > 1:
                if self.check_convergence():
                    break

            self.congested_time = self.vdf.apply_vdf(**{**pars, **self.vdf_parameters})

            for c in self.traffic_classes:
                c.graph.cost = self.congested_time
                c._aon_results.reset()

            logger.info('{},{},{}'.format(self.iter, self.rgap, stepsize))

        if self.rgap > self.rgap_target:
            logger.error('Desired RGap of {} was NOT reached'.format(self.rgap_target))
        logger.info('MSA Assignment finished. {} iterations and {} final gap'.format(self.iter, self.rgap))

    def check_convergence(self):

        aon_cost = np.sum(self.congested_time * self.aon_total_flow)
        msa_cost = np.sum(self.congested_time * self.msa_total_flow)
        self.rgap = abs(msa_cost - aon_cost) / msa_cost
        if self.rgap_target >= self.rgap:
            return True
        return False
