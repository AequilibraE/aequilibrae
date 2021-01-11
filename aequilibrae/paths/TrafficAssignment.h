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
#include <iostream>
#include <algorithm>
//#include "ShortestPathComputation.h"


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
	std::vector<unsigned long> destinations;
	std::map<int, std::vector<unsigned int> > path_link_incidence;
	std::vector<std::vector<int> > paths;
	//int *paths;
	std::vector<float> path_flows;
	std::vector<float> path_flows_current_iter;
};


class TrafficAssignment {
public:
	std::vector<Link> links;
	std::map<unsigned long, int> node_to_link;
	std::vector<Centroid> centroidsDescriptors;
	int num_nodes;

    int *precedence;
    // for parallel shortest path:
    //std::map<int, std::vector<int>> precedence_by_origin;

    int *buffer_path;

    std::vector<float> weights;
    std::vector<float> costs;
    std::vector<float> link_flows;
    std::vector<float> alphas_1;
    std::vector<float> alphas_2;
    std::vector<float> link_flows_out_of_partition;
    std::vector<std::vector<float> > link_flows_origin;

    //float *link_flows_origin;
    //float *link_flows_origin_current_iter_diff;
    std::vector<std::vector<float> > link_flows_origin_current_iter_diff;

    unsigned int n_cent;
    unsigned int n_links;
	//ShortestPathComputation *spComputation;



	TrafficAssignment(int num_links, int num_nodes, int num_centroids);

	void add_link(int link_id, float t0, float alfa, int beta, float capacity, int from_node,
	              int to_node);

	//void set_edges();
	//void perform_initial_solution();
	//void compute_shortest_paths(int from_node);

	void update_link_flows(unsigned int from_node);
	void update_link_flows_stepsize(double stepsize);
	void update_link_flows_by_origin(unsigned int from_node);
	void update_link_flows_by_origin_for_all();
    void update_path_flows_stepsize(double stepsize);

	void insert_od(unsigned long from, unsigned long to, float demand);

	void get_subproblem_data(unsigned int origin, float *Q, float *c, float *A, float *b, float *G, float *h);

	unsigned int get_total_paths(unsigned long origin);
	unsigned int get_total_paths(unsigned long origin, unsigned long destination);

	void update_path_flows(unsigned long centroid, float *flows);
	void update_current_iteration_flows_by_origin(unsigned long centroid, float *flows);
	float objective_derivative_stepsize(double stepsize);

    float compute_gap();

	void compute_path_link_sequence(int origin, int destination);

	void get_link_flows(float *ptr_flows);
	// test for shortest path replacement:
	//void get_precedence(int *prec);
	void set_precedence(int *prec);
	void compute_path_link_sequence_external_precedence(int from_node);
	void set_initial_path_flows(unsigned int origin);
	void get_congested_times(float *travel_time);

	void get_odpath_times(unsigned long origin, unsigned long destination, float *path_times,  float *path_flows);

	void update_link_derivatives(int link_id);

    float get_objective_function();

    void get_objective_data(unsigned int origin, float *Q, float *c);
    void get_equality_data(unsigned int origin, float *A, float *b);
    void get_inequality_data(unsigned int origin, float *G, float *h);

	virtual ~TrafficAssignment();
};




#endif /* TRAFFICASSIGNMENT_H_ */
