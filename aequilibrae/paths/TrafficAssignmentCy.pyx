# distutils: language=c++
# distutils: sources = TrafficAssignment.cpp

from cpython cimport array
#import array
import ctypes
import numpy
cimport numpy
import cvxopt
from libcpp.vector cimport vector
from libcpp.pair cimport pair

cdef extern from "TrafficAssignment.h":
    ctypedef struct Link:
        int link_id
        float flow
        float t0
        float alfa
        int beta
        int from_node
        int to_node


    cdef cppclass TrafficAssignment:
        TrafficAssignment(unsigned long num_links, unsigned long num_nodes, unsigned long num_centroids) except +
        void insert_od(unsigned long fromnode, unsigned long tonode, float demand)
        void add_link(int link_id, float t0, float alfa, int beta, float capacity, unsigned long from_node,unsigned long to_node)
        # void set_edges()
        float get_objective_function()
        float objective_derivative_stepsize(double stepsize)
        void get_subproblem_data(unsigned long origin, float *Q, float *c, float *A, float *b, float *G, float *h)
        unsigned int get_total_paths(int origin)
        unsigned int get_total_paths(int origin, int destination)
        # void compute_shortest_paths(int origin)
        # void perform_initial_solution()
        void get_link_flows(float *ptr_flows)
        void update_path_flows(unsigned long origin, float *flows)
        void get_odpath_times(int origin, int destination, float *buffer, float *path_times)

        void update_current_iteration_flows_by_origin(unsigned long origin, float *flows)
        void update_link_flows_stepsize(double stepsize)
        void update_path_flows_stepsize(double stepsize)
        # void get_precedence(int *prec)
        void set_precedence(int *prec)
        void compute_path_link_sequence_external_precedence(int origin)
        void set_initial_path_flows(unsigned int origin)
        void update_link_flows(unsigned int from_node)
        void get_congested_times(float *travel_time)
        float compute_gap()

        pair[uint, uint] get_total_paths_for_partition(unsigned int origin, unsigned int num_partitions, unsigned int partition_index);
        void get_problem_data_partitions(unsigned int origin, float *Q, float *c, float *A, float *b, float *G, float *h,
                                            int num_partitions, int partition_index)
        void update_path_flows_partition(unsigned long centroid, float *flows,int num_partitions, int partition_index)



cdef class TrafficAssignmentCy:
    cdef TrafficAssignment *thisptr
    cdef object num_nodes
    cdef object num_centroids
    cdef object links

    def __init__(self, links, int num_links, int num_nodes, int num_centroids):
        self.links=links
        self.num_nodes=num_nodes
        self.num_centroids = num_centroids


    def __cinit__(self, links, int num_links, int num_nodes, int num_centroids):
        cdef vector[Link] link_vector
        cdef Link l

        self.thisptr = new TrafficAssignment(num_links, num_nodes, num_centroids)

        for link in links:
            self.thisptr.add_link(link.link_id, link.t0, link.alfa, link.beta, link.capacity, link.node_id_from,
                            link.node_id_to)

        # self.thisptr.set_edges()


    def __dealloc__(self):
        del self.thisptr


    def insert_od(self, origin, destination, flow):
        self.thisptr.insert_od(origin, destination, flow)


#    def compute_shortest_paths(self, origin):
#        self.thisptr.compute_shortest_paths(origin)


    def set_precedence(self, precedence):
        cdef array.array prec = array.array('i', precedence)
        self.thisptr.set_precedence(prec.data.as_ints)

    def compute_path_link_sequence_external_precedence(self, origin):
        self.thisptr.compute_path_link_sequence_external_precedence(origin)

#    def perform_initial_solution(self):
#        self.thisptr.perform_initial_solution()

    def set_initial_path_flows(self, origin):
        self.thisptr.set_initial_path_flows(origin)

    def get_link_flows(self):
        zeros = [0.0 for i in range(len(self.links))]
        # need double here for integration with aequilibrae
        cdef array.array link_f = array.array('f', zeros)
        self.thisptr.get_link_flows(link_f.data.as_floats)
        return link_f

    def get_congested_times(self):
        zeros = [0.0 for i in range(len(self.links))]
        # need double here for integration with aequilibrae
        cdef array.array link_times = array.array('f', zeros)
        self.thisptr.get_congested_times(link_times.data.as_floats)
        return link_times


#    def get_precedence(self):
#        zeros = [0 for i in range(self.num_nodes)]
#        cdef array.array prec = array.array('i', zeros)
#        self.thisptr.get_precedence(prec.data.as_ints)
#        return prec


    def get_objective_function(self):
        return self.thisptr.get_objective_function()


    def get_problem_data(self, origin, num_destinations_from_origin):
        num_paths = self.get_total_paths(origin)

        cdef array.array Q= array.array('f', [0.0 for i in range(num_paths*num_paths)])
        #array.resize(Q,num_paths*num_paths)

        cdef array.array q= array.array('f', [0.0 for i in range(num_paths)])
        #array.resize(q,num_paths)

        cdef array.array A= array.array('f',  [0.0 for i in range(num_destinations_from_origin*num_paths)])
        #array.resize(A, num_destinations_from_origin*num_paths)

        cdef array.array b= array.array('f', [0.0 for i in range(num_destinations_from_origin)])
        #array.resize(b,num_destinations_from_origin)

        cdef array.array G= array.array('f', [0.0 for i in range(num_paths*num_paths)])
        #array.resize(G, num_paths*num_paths)

        cdef array.array h= array.array('f', [0.0 for i in range(num_paths)])
        #array.resize(h,num_paths)

        self.thisptr.get_subproblem_data(origin,Q.data.as_floats,q.data.as_floats,
                                         A.data.as_floats, b.data.as_floats,
                                         G.data.as_floats, h.data.as_floats)

        return Q,q,A,b,G,h


    def get_total_paths(self, origin):
        return self.thisptr.get_total_paths(origin)


    def update_path_flows(self, origin, flows):
        cdef array.array path_flows= array.array('f', flows)
        self.thisptr.update_path_flows(origin, path_flows.data.as_floats)

    def update_link_flows(self, from_node):
        self.thisptr.update_link_flows(from_node)


    def get_path_times(self, origin, destination):
        num_paths = self.thisptr.get_total_paths(origin, destination)

        cdef array.array path_times= array.array('f', [0.0 for i in range(num_paths)])
        cdef array.array path_flows= array.array('f', [0.0 for i in range(num_paths)])

        self.thisptr.get_odpath_times(origin, destination, path_times.data.as_floats, path_flows.data.as_floats)
        return path_times, path_flows

    def get_problem_data_partition(self, origin, num_partitions, partition):
        pair_ret_val = self.thisptr.get_total_paths_for_partition(origin,num_partitions, partition);


        cdef int total_destinations = pair_ret_val.first;
        cdef int num_paths = pair_ret_val.second;

        cdef array.array Q= array.array('f', [0.0 for i in range(num_paths*num_paths)])

        cdef array.array q= array.array('f', [0.0 for i in range(num_paths)])

        cdef array.array A= array.array('f',  [0.0 for i in range(total_destinations*num_paths)])

        cdef array.array b= array.array('f', [0.0 for i in range(total_destinations)])

        cdef array.array G= array.array('f', [0.0 for i in range(num_paths*num_paths)])
        #array.resize(G, num_paths*num_paths)

        cdef array.array h= array.array('f', [0.0 for i in range(num_paths)])
        #array.resize(h,num_paths)

        self.thisptr.get_problem_data_partitions(origin,Q.data.as_floats,q.data.as_floats,
                                         A.data.as_floats, b.data.as_floats,
                                         G.data.as_floats, h.data.as_floats,
                                         num_partitions, partition)

        A_diff = numpy.array(A, 'd')
        A_diff.reshape((num_paths, total_destinations))

        b_diff = numpy.array(b, 'd')

        Q_diff = numpy.array(Q, 'd')
        Q_diff.reshape((num_paths,num_paths))

        q_diff = numpy.array(q, 'd')

        G_diff = numpy.array(G,'d')
        G_diff.reshape((num_paths,num_paths))

        h_diff = numpy.array(h, 'd')

        # print destinations_per_origin[0], t_paths, len(A)
        Am = cvxopt.matrix(A_diff.astype(numpy.double), ((num_paths, total_destinations)),"d" )
        bm = cvxopt.matrix(b_diff.astype(numpy.double), (total_destinations,1),"d" )

        Qm = cvxopt.matrix(Q_diff.astype(numpy.double), (num_paths,num_paths),"d" )
        qm = cvxopt.matrix(q_diff.astype(numpy.double), (num_paths,1),"d" )

        Gm = cvxopt.matrix(G_diff.astype(numpy.double), (num_paths,num_paths),"d" )
        hm = cvxopt.matrix(h_diff, (num_paths,1),"d" )


        return Qm,qm,Am,bm,Gm,hm

    def update_path_flows_for_partition(self, origin, flows, num_partitions, partition_index):
        cdef array.array path_flows= array.array('f', flows)
        self.thisptr.update_path_flows_partition(origin, path_flows.data.as_floats, num_partitions, partition_index)


    def compute_gap(self):
        return self.thisptr.compute_gap()

    ### new for parallel implementation

    def update_current_iteration_flows_by_origin(self, origin, flows):
        cdef array.array path_flows = array.array('f', flows)
        self.thisptr.update_current_iteration_flows_by_origin(origin, path_flows.data.as_floats)

    def update_link_flows_stepsize(self, stepsize):
        self.thisptr.update_link_flows_stepsize(stepsize)

    def update_path_flows_stepsize(self, stepsize):
        self.thisptr.update_path_flows_stepsize(stepsize)

    def objective_derivative_stepsize(self, stepsize):
        """Calculate the derivative of the objective function with respect to the stepsize given the
        update scheme described in the notes"""
        return self.thisptr.objective_derivative_stepsize(stepsize)



