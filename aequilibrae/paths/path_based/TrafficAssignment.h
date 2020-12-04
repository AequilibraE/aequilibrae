/*
 * TrafficAssignment.h
 *
 *  Created on: Dec 20, 2016
 *      Author: fas
 */

#ifndef TRAFFICASSIGNMENT_H_
#define TRAFFICASSIGNMENT_H_
#include <vector>
#include <map>
#include "ShortestPathComputation.h"

#define PATHS_PER_OD 10

struct Link {
	unsigned long link_id;
	float flow;
	float t0;
	float alfa;
	float capacity;
	int beta;
	int from_node;
	int to_node;
};



struct Path{
	int path_index;
	std::vector<int> link_sequence;
	float flow;
};

struct DestinationDescriptor {
	int destination;
	float demand;
	std::vector<unsigned int> path_indices;

};


struct Centroid {
	int node;
	unsigned int num_paths;
	std::map<unsigned int, unsigned int> crcs;
	std::map<unsigned long, DestinationDescriptor> destinationDescriptors;
	std::map<int, std::vector<unsigned int> > path_link_incidence;
	int *paths;
	std::vector<float> path_flows;
};



class TrafficAssignment {
public:
	std::vector<Link> links;
	std::map<unsigned long, int> node_to_link;
	std::vector<Centroid> centroidsDescriptors;
	int num_nodes;

    int *precedence;
    int *buffer_path;

    float *weights;
    float *costs;
    float *link_flows;
    float *alphas_1;
    float *alphas_2;

    float *link_flows_origin;
    float *link_flows_origin_current_iter_diff;

    unsigned int n_cent;
    unsigned int n_links;
	ShortestPathComputation *spComputation;



	TrafficAssignment(int num_links, int num_nodes, int num_centroids);

	void add_link(int link_id, float t0, float alfa, int beta, float capacity, int from_node,
	              int to_node);

	void set_edges();

	void perform_initial_solution();

	void update_link_flows(unsigned int from_node);
	void update_link_flows_stepsize(unsigned int from_node, float alpha);
	void update_link_flows_by_origin(unsigned int from_node);

	void insert_od(unsigned long from, unsigned long to, float demand);

	void compute_shortest_paths(int from_node);

	void get_subproblem_data(unsigned int origin, float *Q, float *c, float *A, float *b, float *G, float *h);

	unsigned int get_total_paths(unsigned long origin);
	unsigned int get_total_paths(unsigned long origin, unsigned long destination);

	void update_path_flows(unsigned long centroid, float *flows);
	void update_path_flows_without_link_flows(unsigned long centroid, float *flows);

	void compute_path_link_sequence(int origin, int destination);

	void get_link_flows(float *ptr_flows);

	void get_odpath_times(unsigned long origin, unsigned long destination, float *path_times,  float *path_flows);

	void update_link_derivatives(int link_id);
	void update_all_link_derivatives();

    float get_objective_function();

    void get_objective_data(unsigned int origin, float *Q, float *c);
    void get_equality_data(unsigned int origin, float *A, float *b);
    void get_inequality_data(unsigned int origin, float *G, float *h);

	virtual ~TrafficAssignment();
};




#endif /* TRAFFICASSIGNMENT_H_ */
