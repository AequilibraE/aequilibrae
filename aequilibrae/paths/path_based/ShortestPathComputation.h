/*
 * ShortestPathComputation.h
 *
 *  Created on: Nov 29, 2016
 *      Author: fas
 */

#ifndef SHORTESTPATHCOMPUTATION_H_
#define SHORTESTPATHCOMPUTATION_H_

#include <boost/config.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/property_map/property_map.hpp>

#include <iostream>
#include <fstream>

using namespace boost;

typedef adjacency_list < listS, vecS, directedS,
no_property, property < edge_weight_t, float > > graph_t;
typedef graph_traits < graph_t >::vertex_descriptor vertex_descriptor;
typedef graph_traits < graph_t >::edge_descriptor edge_descriptor;
typedef std::pair<int, int> Edge;


class ShortestPathComputation {
	int n_nodes;
	int n_arcs;

	std::vector<Edge> edges;

public:
	ShortestPathComputation(int num_nodes, int num_arcs);
	virtual ~ShortestPathComputation();

	void set_edges(int *from_nodes, int *to_nodes);

	void compute_shortest_paths(float *weights, int from_node,int *precedence, float *costs);
};

#endif /* SHORTESTPATHCOMPUTATION_H_ */
