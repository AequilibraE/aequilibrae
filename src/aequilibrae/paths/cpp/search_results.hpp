#pragma once

#include <cstddef>
#include <limits>

namespace aequilibrae::paths::cpp::mvp {
// Purpose
// -------
//
// The SearchResults object aims to store the results of a path finding
// operation in such a manner that whether turn penalties were enabled or not
// for the operation is irrelevant. It accomplishes this by replacing the normal
// "node predecessors" with "state predecessors" and "terminal states", it also
// moves the main distance "skimming" to be the responsibility of the path
// finding function. This allows consumes to use that output rather than attempt
// to re-work the path, then have to worry about include the turn penalties
// properly.
//
// The splitting of "node predecessors" into "state predecessors" and "terminal
// states" is required because the optimal path to a node is not necessarily the
// same as an optimal path that goes through a node when turn penalties are
// enabled. The simplest case to think about is when path finding to the "via
// node" of turn penalty vs the "to node". By continuing past the via node, an
// additional cost is incurred. In order to be able to correctly reconstruct the
// path we must differentiate between a path that ends at a node vs when that
// node is used somewhere else in the path. However, this difference between the
// paths only occurs once, right at the end of the path, if at all. Thus we only
// need to differentiate between them for the first "predecessor" look up.
//
// More concisely, without turn penalties, the invariant hypothesis of a
// shortest path on a non-negative DAG is:
//
//   For a shortest path between two nodes P and Q, if R is a node on that
//   minimal path, then P to R is minimal.
//
// When turn penalties are enabled, the above is not true. The path P to R may
// not be minimal.
//
//
// Terms
// -----
//
// We generalise the "nodes" and "arcs" of each method into "states".
//
// A state is what the search assigns a cost and a parent to. In node routing,
// a state is a node. In turn routing, a state is an incoming link. The root is
// the state where the search starts. A state's parent is the previous state
// on its path. Its connector is the link used to reach it from that parent.
//
// A state is settled when the search removes it from the heap with its final
// shortest-path cost. A node's terminal is the first settled state that ends
// at that node. An invalid index means there is no such entry. Its value is
// numeric_limits<size_t>::max(), written as X in the example below.
//
// Node routing has node_count states. Turn routing has one state per link and
// one extra state for the source. Its root is at index link_count, so it has
// link_count + 1 states in total. Two links into the same node give two states
// for that node. There is no extra state for each path. Each state stores one
// best cost and one parent.
//
// Fields
// ------
// Here N is node_count, S is state_count, s is a state, d is a physical node,
// and i is a position in settlement order.
//
// State arrays (S entries):
//
// - predecessors[s]: parent state. This is the previous node in node routing
//   and the previous incoming link (or root) in turn routing.
// - connectors[s]: link used to reach s. This is the incoming link in node
//   routing and s itself in turn routing.
// - reached_first[i]: states in settlement order. Entries are node indices in
//   node routing and incoming-link state indices in turn routing.
// - distances[s]: total path cost. This includes links and turn penalties.
// - turn_costs[s]: cumulative turn cost. This is zero for settled node-routing
//   states and the sum of penalties for settled turn-routing states.
//
// Physical-node arrays (N entries):
//
// - destination_mask[d]: nonzero when d is a requested target.
// - terminal_states[d]: selected arrival state. This is d in node routing and
//   an incoming link (or root) in turn routing.
//
// Scalars:
//
// - root: origin in node routing and link_count in turn routing.
// - origin: physical source node.
// - destination_count: number of nonzero destination-mask entries.
// - reached_destination_count: number of requested physical nodes settled.
// - settled_count: number of settled states, including root.
//
// The root has no parent or incoming link, so its predecessor and connector
// are invalid. Its costs are zero. The origin's terminal is the root.
// Unsettled states have invalid predecessors and connectors. Their costs are
// infinite. A node has an invalid terminal if no arrival at it was settled.
// This can happen because the node is unreachable or because the search
// stopped early.
//
// The first settled_count entries of reached_first are valid. Search stops
// after every requested destination has a terminal, or after the reachable
// state space is exhausted. An empty mask disables early exit. Before the
// first search, settled_count and both destination counts are zero, the mask
// is empty, and origin and root are invalid.
//
// Paths and invariants
// --------------------
// Every prefix of a shortest path is a shortest path to the state it ends at.
// In node routing, it is the cheapest path to that node. In turn routing, it
// is the cheapest path that ends with that incoming link. It need not be the
// cheapest path to the link's head node.
//
// For each settled state s other than the root, let p be predecessors[s].
// The costs satisfy these equations:
//
//   distances[s] = distances[p] + cost(connectors[s]) + penalty(p, s)
//   turn_costs[s] = turn_costs[p] + penalty(p, s)
//
// The penalty is zero in node routing. It is also zero for the first link
// out of the source in turn routing. All costs and penalties must be
// nonnegative. A parent settles before its children. Following parents must
// therefore reach the root without a cycle, even when costs are zero.
//
// The first settled arrival at a node gives its minimum total cost. If two
// arrivals have the same cost, the search keeps the one that settles first.
// It does not use the turn cost to break ties.
//
// To read the path to node d, start at terminal_states[d]. Record that state's
// connector, then move to its predecessor. Repeat until reaching the root.
// Reverse the recorded links to get the path from the origin. This works for
// both routing modes.
//
// Worked example
// --------------
// Consider a search from A to D. The node indices are A=0, B=1, C=2 and D=3.
// The labels on the edges are link indices. Every link costs 1. The turn from
// link 0 to link 2 has a penalty of 10. All other turns have no penalty.
//
//       A --0--> B --2--> D
//       |       ^
//       1       |
//       v       3
//       C ------+
//
// The predecessor tree is:
//
//       4: source at A, cost 0
//       +-- 0: A->B, cost 1
//       +-- 1: A->C, cost 1
//           +-- 3: C->B, cost 2
//               +-- 2: B->D, cost 3
//
//       State  Meaning  predecessors  connectors  distances  turn_costs
//       -----  -------  ------------  ----------  ---------  ----------
//       0      A->B     4             0           1          0
//       1      A->C     4             1           1          0
//       2      B->D     3             2           3          0
//       3      C->B     1             3           2          0
//       4      Source   X             X           0          0
//
//       terminal_states[A, B, C, D] = [4, 0, 1, 2]
//       reached_first = [4, 0, 1, 3, 2]  (states 0 and 1 may swap on the tie)
//       root = 4, origin = 0, settled_count = 5
//       destination_mask = [0, 0, 0, 1]
//       destination_count = reached_destination_count = 1
//
// State 0 is the cheapest arrival at B. It costs 1, but continuing to D costs
// 1 + 1 + 10 = 12. State 3 reaches B at a cost of 2. Continuing from there to
// D costs 2 + 1 = 3. So terminal_states[B] is 0, but predecessors[2] is 3.
// The path to B and the path to D use different arrivals at B.
//
// Starting at terminal_states[D] and following parents gives 2 -> 3 -> 1 -> 4.
// The connectors, in reverse order, are [1, 3, 2]. These links give the path
// A->C->B->D. Its total cost is 3 and its turn cost is 0.
//
// Node routing ignores the turn penalty and has four states:
//
//       0: A, cost 0
//       +-- 1: B, cost 1
//       |   +-- 3: D, cost 2
//       +-- 2: C, cost 1
//
//       State  Node  predecessors  connectors  distances  turn_costs
//       -----  ----  ------------  ----------  ---------  ----------
//       0      A     X             X           0          0
//       1      B     0             0           1          0
//       2      C     0             1           1          0
//       3      D     1             2           2          0
//
//       terminal_states[A, B, C, D] = [0, 1, 2, 3]
//       reached_first = [0, 1, 2, 3]  (states 1 and 2 may swap on the tie)
//       root = 0, origin = 0, settled_count = 4
//       destination_mask = [0, 0, 0, 1]
//       destination_count = reached_destination_count = 1
//
// Following parents gives states 3 -> 1 -> 0. Reading their connectors in
// reverse order gives links [0, 2] and the path A->B->D. Its total cost is 2
// and its turn cost is 0.
//
// Constructing the paths
// ----------------------
// This pseudocode builds either path above. The target d is a node index.
// heads[link] gives the head node of a local directed link.
//
//   function path_to(d):
//       if settled_count == 0:
//           return (nodes=[], links=[], cost=infinity, turn_cost=infinity)
//
//       terminal = terminal_states[d]
//       if terminal == invalid:
//           return (nodes=[], links=[], cost=infinity, turn_cost=infinity)
//
//       links = []
//       state = terminal
//       while state != root:
//           links.append(connectors[state])
//           state = predecessors[state]
//       links.reverse()
//
//       nodes = [origin]
//       for link in links:
//           nodes.append(heads[link])
//
//       return (nodes=nodes, links=links,
//               cost=distances[terminal], turn_cost=turn_costs[terminal])
//
// In the turn example, the loop collects [2, 3, 1], then reverses it to
// [1, 3, 2]. The nodes are [A, C, B, D]. In the node example, it collects
// [2, 0], then reverses it to [0, 2]. The nodes are [A, B, D].
//
// If d is the origin, its terminal is root and the loop takes no steps.
// The path has one node, no links and zero costs. An invalid terminal means
// there is no settled path to return. The total cost already includes the
// turn cost, so the two returned costs must not be added together.
//
struct SearchResults {
  // This array has state_count entries. Each entry is the parent state.
  // In turn routing, the first links have root as their parent. The entries
  // for root and unsettled states are invalid.
  std::size_t *predecessors = nullptr;

  // This array has state_count entries. Each entry is the local link index
  // used to reach the state from its parent. This identifies the chosen link
  // when parallel links have the same end nodes. In turn routing, it equals
  // the link state's index. Root and unsettled states have invalid entries.
  std::size_t *connectors = nullptr;

  // This array has room for state_count entries. The first settled_count
  // entries list states in settlement order, including root. Parents appear
  // before children. In turn routing, several entries can end at the same node.
  std::size_t *reached_first = nullptr;

  // This array has node_count entries. A nonzero entry requests that node as a
  // destination. Both routing modes stop when every requested node has its
  // first settled arrival. An all-zero mask disables early exit. The mask
  // remains unchanged during the search, so it can be inspected afterwards
  // and safely reused by the caller.
  const bool *destination_mask = nullptr;

  // This array has state_count entries. Each entry is the total path cost,
  // including turn penalties. The cost is zero at root and infinite for
  // unsettled states. If node d has a valid terminal, its cheapest path costs
  // distances[terminal_states[d]].
  double *distances = nullptr;

  // This array has state_count entries. Each entry is the sum of penalties
  // along the path, not just the last turn's cost. It is zero at root and for
  // settled states in node routing. It is infinite for unsettled states.
  // This has the same meaning as the existing arc_turn_penalties array.
  // turn_costs[terminal_states[d]] replaces node_turn_penalties[d].
  double *turn_costs = nullptr;

  // This array has node_count entries. Each entry selects the first settled
  // state ending at that node. In node routing, it is the node itself. In
  // turn routing, it is an incoming link. The origin's entry is always root.
  // An entry is invalid if no arrival was settled. Use this array to find the
  // start of a predecessor chain, not to choose parents within the chain.
  std::size_t *terminal_states = nullptr;

  // The root is origin in node routing and link_count in turn routing.
  // It has zero cost and no parent or connector. First links pay no turn cost.
  // Both modes stop at root when following parents. The path from the origin
  // to itself contains only root and has no links. This field is invalid
  // before the first search.
  std::size_t root = std::numeric_limits<std::size_t>::max();

  // This is the source node of the last search and the first node in its path.
  // It may differ from the root state index. It is invalid before a search.
  std::size_t origin = std::numeric_limits<std::size_t>::max();

  // The caller supplies destination_count together with the immutable borrowed
  // mask, whose allocation must outlive the search. Searches preserve both and
  // reset only reached_destination_count. Comparing the counts reports whether
  // all destinations were reached without rescanning the mask.
  std::size_t destination_count = 0;
  std::size_t reached_destination_count = 0;

  // This counts settled states, including root, rather than physical nodes.
  // It is zero before a search and at least one afterwards. The existing code
  // returns found - 1, which is the last
  // valid position in reached_first. This field gives the number of entries.
  std::size_t settled_count = 0;
};

} // namespace aequilibrae::paths::cpp::mvp
