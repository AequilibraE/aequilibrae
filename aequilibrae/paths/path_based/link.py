import math


class Link:
    def __init__(self, node_id_from, node_id_to, link_id, t0, capacity, alfa, beta, direction=1, t0_ba=None, c_ba=None):
        self.link_id = link_id
        self.node_id_from = node_id_from
        self.node_id_to = node_id_to
        self.link_index = None
        self.t0 = t0
        self.capacity = capacity
        self.alfa = alfa
        self.beta = beta
        self.direction = direction
        self.t0_ba = t0_ba
        self.capacity_ba = c_ba

    def get_cost(self, flow):
        # cost of the integral
        flow = float(flow)
        # return self.t0*flow*(self.alfa*math.pow((flow/self.capacity),self.beta))/(self.beta+1) + self.t0*flow
        return self.t0 * flow * (self.alfa * ((flow / self.capacity) ** self.beta)) / (self.beta + 1) + self.t0 * flow

    def get_time(self, flow):
        # return self.t0*(1+self.alfa*math.pow((flow/self.capacity),self.beta))
        return self.t0 * (1 + self.alfa * ((flow / self.capacity) ** self.beta))

    def get_dtime(self, flow):
        try:
            # p = math.pow(flow, self.beta-1)
            p = flow ** (self.beta - 1)
        except:  # noqa: E722
            return 0

        den = math.pow(self.capacity, self.beta)

        return p * self.alfa * self.t0 * self.beta / den

    def get_quadratic_approximation(self, flow):
        c_0 = self.get_cost(flow)
        c_1 = self.get_time(flow)
        c_2 = self.get_dtime(flow)

        alfa_1 = c_2 / 2
        alfa_2 = c_1 - flow * c_2
        # alphas_2[link_id]=weights[link_id]-flow*dtime;
        alfa_2 = c_1 - flow * c_2

        alfa_3 = c_0 - c_1 * flow + (c_2 / 2.0) * flow * flow

        return alfa_1, alfa_2, alfa_3


if __name__ == "__main__":

    l13 = Link(link_id=13, t0=1, capacity=500.0, alfa=0.15, beta=4, node_id_from=0, node_id_to=2)

    flows = [5.0 * i for i in xrange(1000)]  # noqa: F821
    costs = []

    for flow in flows:
        cost = l13.get_cost(flow)
        costs.append(cost)

    import pylab

    pylab.plot(flows, costs)

    a_1, a_2, a_3 = l13.get_quadratic_approximation(1300 / 2)
    print(a_1, a_2)

    costs_0 = []

    for flow in flows:
        cost = a_1 * flow * flow + a_2 * flow + a_3
        costs_0.append(cost)
    pylab.plot(flows, costs_0)

    pylab.show()
