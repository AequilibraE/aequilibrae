/*
 * ShortestPathComputation.cpp
 *
 *  Created on: Nov 29, 2016
 *      Author: fas
 */



#include "ShortestPathComputation.h"

ShortestPathComputation::ShortestPathComputation(int num_nodes, int num_arcs) {
	// TODO Auto-generated constructor stub
	this->n_arcs = num_arcs;
	this->n_nodes = num_nodes;

	//edges.resize(num_arcs);

	/*for (int i=0; i<n_arcs;i++) {
		edges[i] = Edge(from_nodes[i], to_nodes[i]);
	}*/
}

ShortestPathComputation::~ShortestPathComputation() {
	// TODO Auto-generated destructor stub
}


void ShortestPathComputation::set_edges(int *from_nodes, int *to_nodes) {
	edges.resize(n_arcs);

	for (int i=0; i<n_arcs;i++) {
		edges[i] = Edge(from_nodes[i], to_nodes[i]);
	}
}

void ShortestPathComputation::compute_shortest_paths(float  *weights, int from_node,
		int *precedence, float *costs) {


	  Edge *edge_array = &edges[0];

	  int num_arcs = n_arcs;

	  graph_t g(edge_array, edge_array + num_arcs, weights, n_nodes);


	  //std::vector<vertex_descriptor> p(num_vertices(g));
	  //std::vector<float> d(num_vertices(g));
	  std::vector<float> d(boost::num_vertices(g));
	  vertex_descriptor s = vertex(from_node, g);

	  //dijkstra_shortest_paths(g, s, predecessor_map(&p[0]).distance_map(&d[0]));
	  dijkstra_shortest_paths(g, s, predecessor_map(precedence).distance_map(costs));
	  //dijkstra_shortest_paths(g, s,distance_map(&d[0]).predecessor_map(&p[0]));

/*	  std::cout << "distances and parents:" << std::endl;
	  graph_traits < graph_t >::vertex_iterator vi, vend;


	//  boost::distance_map(distanceMap).predecessor_map(predecessorMap));

	   for (tie(vi, vend) = vertices(g); vi != vend; ++vi) {

	        //vertex_descriptor d = (vertex_descriptor)*vi;
	        //float f = (float)d;
	        //std::cout << f << std::endl;
	        std::cout << costs[*vi] << "  " << precedence[*vi] << std::endl;
	        //std::cout << "parent of "<< name[*vi]  <<" = "  <<  name[p[*vi]] << std::endl;
	    }

*/

}
