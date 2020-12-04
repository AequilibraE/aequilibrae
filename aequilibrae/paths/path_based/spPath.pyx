# distutils: language=c++
# distutils: sources = ShortestPathComputation.cpp

from cpython cimport array
import array
import ctypes

cdef extern from "ShortestPathComputation.h":
    cdef cppclass ShortestPathComputation:
        ShortestPathComputation(int num_nodes, int num_arcs) except +
        void compute_shortest_paths(float *weights, int from_node,int *precedence, float *costs)
        void set_edges(int *from_nodes, int *to_nodes)

cdef class PyShortestPath:
    cdef ShortestPathComputation *thisptr
    cdef array.array precedence
    cdef array.array costs

    def __cinit__(self, int num_nodes, int num_arcs):
        self.thisptr = new ShortestPathComputation(num_nodes, num_arcs)
        #self.cost = new float[num_nodes]

        self.costs= array.array('f',[0.0 for i in xrange(num_nodes)])

        empty_list = [0 for i in xrange(num_nodes)]

        self.precedence= array.array('i', empty_list)

    def __dealloc__(self):
        del self.thisptr


    def copy_precedence(self, p_i):
        for i in xrange(len(p_i)):
            p_i[i]=self.precedence[i]

    def copy_costs(self, c_i):
        for i in xrange(len(c_i)):
            c_i[i]=self.costs[i]

    def compute_shortest_paths(self,from_node,weights):
        #cdef int precedence[2]
        #cdef float costs[2]


        cdef array.array w= array.array('f', weights)

        #cdef array.array costs= array.array('f',[0.0 for i in xrange(num_nodes)])

        #empty_list = [0 for i in xrange(num_nodes)]

        #cdef array.array precedence= array.array('i', empty_list)


        self.thisptr.compute_shortest_paths(w.data.as_floats, from_node, self.precedence.data.as_ints,
                    self.costs.data.as_floats)

        #print self.precedence.data_as_ints[0]

        #return precedence, costs

    def set_edges(self, from_nodes, to_nodes):
        cdef array.array a= array.array('i', from_nodes)
        cdef array.array b= array.array('i', to_nodes)



        self.thisptr.set_edges(a.data.as_ints, b.data.as_ints)