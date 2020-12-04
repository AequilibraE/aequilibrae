from __future__ import print_function
import os

from .node import Node
from .link import Link


class OpenBenchmark:
    link_fields = {"from": 1, "to": 2, "capacity": 3, "length": 4, "t0": 5, "B": 6, "power": 7, "V": 8}

    def __init__(self, folder, link_file, trip_file):
        self.folder = folder
        self.link_file = link_file
        self.trip_file = trip_file

        self.nodes = None
        self.links = None

    def build_datastructure(self):
        links, nodes = self.open_link_file()
        ods = self.open_trip_file()

        destinations = []
        origins = []

        for (origin, destination) in ods:
            if destination not in destinations:
                destinations.append(destination)

            if origin not in origins:
                origins.append(origin)

        self.links = links
        self.nodes = nodes

        return links, nodes, ods, destinations, origins

    def get_data_structure_with_by_directional_links(self):
        links, nodes, ods, destination, origins = self.build_datastructure()
        nodes = {}

        for link in links:
            a = link.node_id_from
            b = link.node_id_to

            if a < b:
                pair = (a, b)
            else:
                pair = (b, a)

            if pair not in nodes:
                nodes[pair] = []

            nodes[pair].append(link)

        bi_directional_links = []
        for pair in nodes:
            if len(nodes[pair]) == 1:
                bi_directional_links.append(nodes[pair][0])
            else:
                if nodes[pair][0].node_id_from < nodes[pair][0].node_id_to:
                    lab = nodes[pair][0]
                    lba = nodes[pair][1]
                else:
                    lba = nodes[pair][1]
                    lab = nodes[pair][0]

                lab.direction = 0
                lab.t0_ba = lba.t0
                lab.capacity_ba = lba.capacity
                bi_directional_links.append(lab)

        return bi_directional_links, nodes, ods, destination, origins

    def compute_objective_function(self, flow_file):
        flows = self.open_flow_file(flow_file)
        cost = 0
        for i, link in enumerate(self.links):
            cost += link.get_cost(flows[i])
        return cost

    def open_link_file(self):
        f = open(os.path.join(self.folder, self.link_file))
        lines = f.readlines()
        f.close()

        links_info = []
        header_found = False
        for line in lines:
            if not header_found and line.startswith("~"):
                header_found = True
            elif header_found:
                links_info.append(line)

        nodes = {}
        links = []
        for line in links_info:
            data = line.split("\t")

            try:
                origin_node = int(data[self.link_fields["from"]]) - 1
            except IndexError:
                continue
            to_node = int(data[self.link_fields["to"]]) - 1
            capacity = float(data[self.link_fields["capacity"]])
            t0 = float(data[self.link_fields["t0"]])
            alfa = float(data[self.link_fields["B"]])
            power = float(data[self.link_fields["power"]])
            # power=1.0
            #
            # print origin_node, to_node, capacity, length, t0, alfa, power

            if origin_node not in nodes:
                n = Node(node_id=origin_node)
                nodes[origin_node] = n

            if to_node not in nodes:
                n = Node(node_id=to_node)
                nodes[to_node] = n

            l_ = Link(
                link_id=len(links),
                t0=t0,
                capacity=capacity,
                alfa=alfa,
                beta=power,
                node_id_from=origin_node,
                node_id_to=to_node,
            )
            links.append(l_)

        return links, nodes.values()

    def open_trip_file(self):
        f = open(os.path.join(self.folder, self.trip_file))
        lines = f.readlines()
        f.close()

        ods = {}
        current_origin = None
        for line in lines:
            if current_origin is None and line.startswith("Origin"):
                origin = int(line.split("Origin")[1])
                current_origin = origin

            elif current_origin is not None and len(line) < 3:
                # print "blank",line,
                current_origin = None

            elif current_origin is not None:
                to_process = line[0:-2]
                # print "process", current_origin, to_process.split(";")
                # print
                for el in to_process.split(";"):
                    # print el.split(":")
                    try:
                        dest = int(el.split(":")[0])
                        demand = float(el.split(":")[1])
                        # print current_origin, dest, demand
                        if current_origin != dest:
                            ods[current_origin - 1, dest - 1] = demand
                    except:  # noqa: E722
                        continue
        return ods

    def open_flow_file(self, filename):
        f = open(self.folder + filename)
        lines = f.readlines()
        f.close()

        links_info = []
        flows = []

        header_found = False
        for line in lines:
            if not header_found and line.startswith("~"):
                header_found = True
            elif header_found:
                links_info.append(line)

        for line in links_info:
            data = line.split("\t")
            flow = float(data[4])
            flows.append(flow)
        return flows


def convert_to_sparse(cvx_opt_matrix):
    pass
