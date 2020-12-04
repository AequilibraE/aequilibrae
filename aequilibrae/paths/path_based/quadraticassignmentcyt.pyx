
import spPath

import cvxopt
import cvxopt.solvers
import cvxopt.lapack
import time
import random
import copy

cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['maxiters'] = 5
cvxopt.solvers.options['abstol'] =1e-10
cvxopt.solvers.options['reltol'] =1e-10
cvxopt.solvers.options['feastol'] =1e-9

import numpy
from cpython cimport array

from libc.math cimport pow

MAX_PATH = 10

import cyPath
#from pyPath import PyPath
#import pyPath



cdef class QuadraticAssignmentCyt:
    cdef public array.array link_flows
    cdef public array.array link_flows_origin
    cdef public array.array link_times
    cdef public array.array precedence

    cdef int num_links
    cdef public object links
    cdef public object nodes
    cdef public object ods
    cdef public object paths
    cdef public object path_flows
    cdef public object num_paths_by_origin
    cdef public object num_paths_by_od
    cdef public object destinations_by_origin
    cdef public object by_nodes_links
    cdef public object origins
    cdef public object destination
    cdef public object shortest_path

    cdef public object buffer_path
    cdef public object destinations

    def __cinit__(self, links, nodes, ods):


    #def __init__(self, links,nodes,ods):
        self.links = links
        self.nodes = nodes
        self.ods = ods
        self.paths = {}
        self.path_flows = {}
        self.num_paths_by_origin = {}
        self.num_paths_by_od = {}

        self.destinations_by_origin = {}


        edges_from = []
        edges_to = []
        self.by_nodes_links = {}

        for i in range(len(links)):
            self.links[i].link_index = i
            edges_from.append(self.links[i].node_id_from)
            edges_to.append(self.links[i].node_id_to)
            self.by_nodes_links[(self.links[i].node_id_from,self.links[i].node_id_to)] = i


        n_nodes = len(self.nodes)

        for o,d in self.ods.keys():
            if o==d or self.ods[o,d]==0:
                continue

            if o not in self.destinations_by_origin:
                self.destinations_by_origin[o] = []

            self.destinations_by_origin[o].append(d)
            self.num_paths_by_od[o,d] = 0
            self.paths[o,d] = []

            for i in range(MAX_PATH):
                self.paths[o,d] = []

        self.buffer_path = array.array('i',[0 for i in range(len(self.links))])
        origins = []
        destinations = []

        for od in self.ods:
            o,d = od
            if o not in origins:
                origins.append(o)

            if d not in destinations:
                destinations.append(d)

            self.paths[od] = []
            self.path_flows[od] = []

        self.origins = origins
        for origin in self.origins:
            self.num_paths_by_origin[origin] = 0


        self.destinations = destinations
        self.shortest_path = spPath.PyShortestPath(len(nodes), len(links))

        self.shortest_path.set_edges(edges_from,edges_to)
        self.num_links = len(links)


        #self.precedence = numpy.zeros(n_nodes, numpy.int32)
        self.precedence = array.array('i',[0 for i in range(n_nodes)])

        self.link_flows = array.array('f',[0.0 for i in range(len(links))])
        self.link_times = array.array('f',[0.0 for i in range(len(links))])
        self.link_flows_origin = array.array('f',[0.0 for i in range(len(links))])


    def perform_initial_solution(self):
        weights = []

        for i,link in enumerate(self.links):
            #weights.append(link.get_time(0.0))
            self.link_times[i] = link.get_time(0.0)
#        print weights



        for origin in self.origins:
            t0 = time.time()
            self.shortest_path.compute_shortest_paths(origin, self.link_times)
            self.shortest_path.copy_precedence(self.precedence)

            #print origin, self.precedence
            for destination in self.destinations_by_origin[origin]:
                #print origin, destination

                # print origin, destination, self.precedence
                #pyPath = self.compute_path(origin, destination,self.precedence)
                # print destination

                pyp = self.compute_path(origin, destination,self.precedence)
                #pyp = PyPath(len(path), path)

                #print origin, destination, path

                #path.append(origin)
                if pyp not in self.paths[origin,destination]:
                    self.paths[(origin, destination)].append(pyp)
                    self.path_flows[origin,destination].append(self.ods[origin,destination])
                #self.path_flows[(origin,destination)].append()

        self.update_link_flows()
        # print self.path_flows
        # print self.paths


    def compute_iteration(self):
        import copy
        # a = copy.deepcopy(self.origins)
        # shuffle_oriign = random.shuffle(a)

        # o = random.randint(0,len(self.origins)-1)

        complete_loop_times = [0.0 for i in range(len(self.origins))]
        time_paths = [0.0 for i in range(len(self.origins))]
        problem_building_times = [0.0 for i in range(len(self.origins))]
        optimization_time = [0.0 for i in range(len(self.origins))]
        variable_update_time = [0.0 for i in range(len(self.origins))]


        for origin in self.origins:
            t_start = time.time()

            self.update_link_flows()
            self.shortest_path.compute_shortest_paths(origin, self.link_times)
            self.shortest_path.copy_precedence(self.precedence)



            for destination in self.destinations_by_origin[origin]:


                p = self.compute_path(origin, destination,self.precedence)

                #p = pyPath.PyPath(len(path), path)

                matched = False
                for py_path in self.paths[origin,destination]:
                    if p.crc == py_path.crc:
                        matched = True
                        break

                # if path not in self.paths[origin,destination]:
                if not matched:
                    self.paths[(origin, destination)].append(p)
                    self.path_flows[origin,destination].append(0)
                    # print origin, destination, 'added path', path

            link_set = {}

            num_paths = 0
            map_path_index = {}
            for destination in self.destinations_by_origin[origin]:
                # if origin==destination:
                #     continue
                #
                # if (origin,destination) not in self.ods:
                #     continue

                for i,py_path in enumerate(self.paths[origin,destination]):
                    path = py_path.link_ids
                    map_path_index[destination,i] = num_paths

                    num_paths += 1
                    for j in range(py_path.num_links):
                        l = py_path.links_array[j]
                        if l not in link_set:
                            link_set[l] = []
                        link_set[l].append(map_path_index[destination,i])

                    #for l in py_path.get_as_list():
                    #    if l not in link_set:
                    #        link_set[l] = []
                    #    link_set[l].append(map_path_index[destination,i])


            # del link_set[0]
            # del link_set[5]
            #link_set.sort()
            ordered_set = sorted(link_set.keys())
            # print ordered_set
            #link_flow_origin = self.compute_link_flow_from_origin(origin)
            self.compute_link_flow_from_origin(origin)

            num_links_o = len(link_set)



            t_end_path = time.time()

            Q = cvxopt.matrix(0.0, (num_paths, num_paths),'d')
            c = cvxopt.matrix(0.0, (num_paths, 1),'d')
            A = cvxopt.matrix(0.0, (len(self.destinations_by_origin[origin]), num_paths),'d')
            b = cvxopt.matrix(0.0, (len(self.destinations_by_origin[origin]), 1),'d')
            primal_start = cvxopt.matrix(0.0, (num_paths,1),'d')
            G = cvxopt.matrix(0.0,(num_paths, num_paths),'d')
            h = cvxopt.matrix(0.0,(num_paths, 1),'d')

            # print  "dijkstra overhead", time.time()-t_origin


            for link in ordered_set:
                alfa_1, alfa_2,alfa_3 = self.links[link].get_quadratic_approximation(self.link_flows[link])
                x0 = self.link_flows[link]-self.link_flows_origin[link]

                for i_a in link_set[link]:
                    for i_b in link_set[link]:
                        Q[i_a,i_b] += 2*alfa_1


                # combinations = []

                # print link, 'a_1', alfa_1,'a_2', alfa_2,'time', self.links[link].get_time(l_flows[link]),x0

                # for path_id in link_set[link]:
                #     for path_idb in link_set[link]:
                #         i_a = path_id
                #         i_b = path_idb
                #
                #         combinations.append((i_a,i_b))

                # for i_a,i_b in combinations:
                #     Q[i_a,i_b] += 2*alfa_1

                for l_ia in link_set[link]:
                    c[l_ia] += 2*x0*alfa_1

                for l_ia in link_set[link]:
                    c[l_ia] += alfa_2

            # print map_path_index

            for destination,i_destination in map_path_index:
                # if destination not in valid_destinations:
                #     continue

                #d_index = valid_destinations.index(destination)
                d_index = self.destinations_by_origin[origin].index(destination)
                path_index = map_path_index[destination,i_destination]
                primal_start[path_index,0] = self.path_flows[origin,destination][i_destination]

                # print origin, destination, i_destination,self.ods[origin,destination]

                # print self.paths[origin,destination][i_destination]

                A[d_index,path_index] = 1.0

            for destination in  self.destinations_by_origin[origin]:
                d_index = self.destinations_by_origin[origin].index(destination)

                if origin!=destination:
                    b[d_index] = self.ods[origin,destination]
                else:
                    print list(A[d_index,:])



            for i in range(num_paths):
                G[i,i] = -1.0

            # print Q
            # print c

            # big_matrix = cvxopt.matrix(0.0, (num_paths+len(valid_destinations),num_paths+len(valid_destinations)),
            #                            'd')
            #
            # big_vector = cvxopt.matrix(0.0, (num_paths+len(valid_destinations),1),'d')

            # big_matrix[0:num_paths,0:num_paths] = Q
            # big_matrix[num_paths:,0:num_paths] = A
            # big_matrix[0:num_paths,num_paths:] = A.trans()
            #
            # big_vector[0:num_paths] = -c
            # big_vector[num_paths:] = b
            #
            # #print Q
            # #print A
            # #print big_matrix
            # cvxopt.lapack.posv(big_matrix, big_vector)
            # try:
            #     cvxopt.lapack.gesv(big_matrix, big_vector)
            #
            #     sol = big_vector[0:num_paths]
            #
            #     feasible = True
            #     for i in xrange(num_paths):
            #         if sol[i] < 0:
            #             feasible = False
            # except:
            #     feasible = False



            t_problem_build = time.time()
            solution = cvxopt.solvers.qp(Q,c, G,h, A,b, primalstart=primal_start)
            sol = solution['x']
            t_optimization_end = time.time()

            # print time.time()-t0
            # print big_vector
            # print solution['x']
            # print 'solution', solution['x']
            # print 0.5*solution['x'].trans()*Q*solution['x']+c.trans()*solution['x']

            for destination,i_destination in map_path_index:
                d_index = self.destinations.index(destination)
                path_index = map_path_index[destination,i_destination]
                flow = sol[path_index]

                self.path_flows[origin,destination][i_destination]=flow


            # print 'origin', origin
            # print 'link set', link_set
            # print 'math path index', map_path_index
            t_variable_update = time.time()

            complete_loop_times[origin] = t_variable_update-t_start
            time_paths[origin] = t_end_path-t_start
            problem_building_times[origin] = t_problem_build-t_end_path
            optimization_time[origin] = t_optimization_end-t_problem_build
            variable_update_time[origin] = t_variable_update-t_optimization_end



            # print time.time()-t_origin
            # print origin, link_set, map_path_index

        avg_time = sum(complete_loop_times)/len(complete_loop_times)
        avg_path = sum(time_paths)/len(time_paths)
        avg_build = sum(problem_building_times)/len(problem_building_times)
        avg_opt = sum(optimization_time)/len(optimization_time)
        avg_var = sum(variable_update_time)/len(variable_update_time)

        return avg_time, avg_path,avg_build,avg_opt, avg_var



    def update_link_flows(self):
        cdef int i
        cdef int j

        #change by a memset?
        #t0=time.time()
        for i in range(self.num_links):
            self.link_flows[i] = 0.0

        for (origin, destination) in self.paths:
            #print self.path_flows[origin,destination]
            for i in range(len(self.paths[origin,destination])):
                pyPath = self.paths[origin,destination][i]

                #path = pyPath.get_as_list()
                for j in range(pyPath.num_links):
                    l_id = pyPath.links_array[j]
                    self.link_flows[l_id] += self.path_flows[origin,destination][i]
                #path = pyPath.link_ids
                #for l_id in pyPath.get_as_list():

                    #self.link_flows[l_id] += self.path_flows[origin,destination][i]

                #print len(pyPath.get_as_list())

        for i in range(self.num_links):

            #self.link_times[i] = self.links[i].get_time(self.link_flows[i])
            self.link_times[i] = self.links[i].t0*(1+self.links[i].alfa*pow((self.link_flows[i]/self.links[i].capacity),self.links[i].beta))
        #t1=time.time()

    def compute_link_flows(self):
        cdef int i
        cdef int j
        self.update_link_flows()
        return self.link_flows

    def compute_link_flow_from_origin(self, o):
        cdef int i
        cdef int j

        for i in range(self.num_links):
            self.link_flows_origin[i] = 0.0

        for (origin, destination) in self.paths:
            if o!= origin:
                continue


            for i in range(len(self.paths[origin,destination])):
                pyPath = self.paths[origin,destination][i]

                for j in range(pyPath.num_links):
                    l_id = pyPath.links_array[j]
                    self.link_flows_origin[l_id] += self.path_flows[origin,destination][i]

                #for l_id in path:
                #    link_flows[l_id] += self.path_flows[origin,destination][i]




    def compute_path(self, origin,destination, precedence):
        #path = []

        num_links = 0

        n=destination


        while precedence[n]!=origin:
            b = precedence[n]
            #path.append(self.by_nodes_links[b,n])
            self.buffer_path[num_links] = self.by_nodes_links[b,n]
            #path.append(self.precedence[n])
            num_links += 1
            n=b

        #path.append(self.by_nodes_links[origin,n])
        self.buffer_path[num_links] = self.by_nodes_links[origin,n]
        num_links += 1
        #path.reverse()

        #new_p = array.copy(self.buffer_path)
        new_p = array.array('i',self.buffer_path)

        p = cyPath.PyPath(num_links, new_p)

        #pyP = pyPath.PyPath(num_links,path)
        #return path
        return p


    def compute_gap(self):
        total_gap = 0
        tts = 0
        for origin in self.origins:
            t_start = time.time()

            for destination in self.destinations_by_origin[origin]:

                #p = pyPath.PyPath(len(path), path)

                matched = False
                path_times = []
                for py_path in self.paths[origin,destination]:
                    path_times.append(self.compute_time(py_path,self.link_flows))


                min_v = min(path_times)

                for i,p_time in enumerate(path_times):
                    tts += p_time*self.path_flows[origin,destination][i]
                    total_gap = total_gap + (p_time-min_v)*self.path_flows[origin,destination][i]

        return total_gap/tts


    def compute_time(self, path, flows):

        t = 0
        for l_id in path.get_as_list():
            t += self.links[l_id].get_time(flows[l_id])

        return t






