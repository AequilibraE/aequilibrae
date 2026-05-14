#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "pq_4ary_heap.hpp"
#include "pq_pairing_heap.hpp"
#include "pq_std_priority_queue_adapter.hpp"

using aequilibrae::paths::cpp::ElementState;
using aequilibrae::paths::cpp::FourAryHeap;
using aequilibrae::paths::cpp::PairingHeap;
using aequilibrae::paths::cpp::StdPriorityQueueAdapter;

struct Update
{
    std::size_t idx;
    double new_key;
};

struct RunResult
{
    bool ok = true;
    double ms = 0.0;
    std::string message;
};

struct CliOptions
{
    std::size_t n = 100000;
    bool csv = false;
    bool csv_header_only = false;
};

bool parse_positive_size(const char *text, std::size_t &value)
{
    const unsigned long parsed = std::strtoul(text, nullptr, 10);
    if (parsed == 0UL)
    {
        return false;
    }
    value = static_cast<std::size_t>(parsed);
    return true;
}

void print_usage(const char *program)
{
    std::cerr << "Usage: " << program << " [num_elements > 0] [--csv] [--csv-header-only]" << '\n';
}

bool parse_args(int argc, char **argv, CliOptions &options)
{
    bool saw_n = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--csv")
        {
            options.csv = true;
            continue;
        }
        if (arg == "--csv-header-only")
        {
            options.csv_header_only = true;
            continue;
        }
        if (!saw_n)
        {
            if (!parse_positive_size(argv[i], options.n))
            {
                return false;
            }
            saw_n = true;
            continue;
        }
        return false;
    }

    return true;
}

void print_csv_header()
{
    std::cout << "elements,decrease_key_ops,std_ms,adapter_ms,fourary_ms,pairing_ms,adapter_vs_std,fourary_vs_std,pairing_vs_std,status" << '\n';
}

std::vector<Update> build_updates(std::vector<double> &keys, std::size_t updates_count, std::mt19937_64 &rng)
{
    std::vector<Update> updates;
    updates.reserve(updates_count);

    if (keys.empty())
    {
        return updates;
    }

    std::uniform_int_distribution<std::size_t> index_dist(0, keys.size() - 1);
    std::uniform_real_distribution<double> drop_dist(0.001, 50.0);

    for (std::size_t i = 0; i < updates_count; ++i)
    {
        const std::size_t idx = index_dist(rng);
        const double new_key = keys[idx] - drop_dist(rng);
        keys[idx] = new_key;
        updates.push_back({idx, new_key});
    }

    return updates;
}

RunResult run_std_reference(
    const std::vector<double> &initial_keys,
    const std::vector<Update> &updates,
    std::vector<std::size_t> &order,
    std::vector<double> &keys)
{
    RunResult result;

    using Entry = std::pair<double, std::size_t>;
    std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> pq;

    std::vector<double> current = initial_keys;
    std::vector<bool> scanned(initial_keys.size(), false);

    const auto start = std::chrono::steady_clock::now();

    for (std::size_t i = 0; i < current.size(); ++i)
    {
        pq.push({current[i], i});
    }

    for (const Update &update : updates)
    {
        current[update.idx] = update.new_key;
        pq.push({update.new_key, update.idx});
    }

    order.clear();
    keys.clear();
    order.reserve(current.size());
    keys.reserve(current.size());

    while (order.size() < current.size())
    {
        if (pq.empty())
        {
            result.ok = false;
            result.message = "stdlib heap exhausted early";
            return result;
        }

        const Entry top = pq.top();
        pq.pop();

        const std::size_t idx = top.second;
        const double key = top.first;
        if (scanned[idx])
        {
            continue;
        }
        if (key != current[idx])
        {
            continue;
        }

        scanned[idx] = true;
        order.push_back(idx);
        keys.push_back(key);
    }

    const auto stop = std::chrono::steady_clock::now();
    result.ms = std::chrono::duration<double, std::milli>(stop - start).count();
    return result;
}

template <class Queue>
RunResult run_queue(
    const std::vector<double> &initial_keys,
    const std::vector<Update> &updates,
    std::vector<std::size_t> &order,
    std::vector<double> &keys)
{
    RunResult result;
    Queue queue;

    const auto start = std::chrono::steady_clock::now();
    queue.init_heap(initial_keys.size());

    for (std::size_t i = 0; i < initial_keys.size(); ++i)
    {
        queue.insert(i, initial_keys[i]);
    }

    for (const Update &update : updates)
    {
        queue.decrease_key(update.idx, update.new_key);
    }

    order.clear();
    keys.clear();
    order.reserve(initial_keys.size());
    keys.reserve(initial_keys.size());

    for (std::size_t i = 0; i < initial_keys.size(); ++i)
    {
        if (queue.is_empty())
        {
            result.ok = false;
            result.message = "queue exhausted early";
            return result;
        }

        const std::size_t idx = queue.extract_min();
        const double key = queue.element_key(idx);

        if (queue.effective_state(idx) != ElementState::SCANNED)
        {
            result.ok = false;
            result.message = "state mismatch (expected SCANNED) at step " + std::to_string(i);
            return result;
        }

        order.push_back(idx);
        keys.push_back(key);
    }

    if (!queue.is_empty())
    {
        result.ok = false;
        result.message = "queue not empty after full extraction";
        return result;
    }

    const auto stop = std::chrono::steady_clock::now();
    result.ms = std::chrono::duration<double, std::milli>(stop - start).count();
    return result;
}

RunResult validate_against_reference(
    const std::vector<std::size_t> &order,
    const std::vector<double> &keys,
    const std::vector<std::size_t> &expected_order,
    const std::vector<double> &expected_keys)
{
    RunResult result;
    if (order.size() != expected_order.size() || keys.size() != expected_keys.size())
    {
        result.ok = false;
        result.message = "result vector size mismatch";
        return result;
    }

    for (std::size_t i = 0; i < order.size(); ++i)
    {
        if (order[i] != expected_order[i])
        {
            result.ok = false;
            result.message = "extraction order mismatch at step " + std::to_string(i);
            return result;
        }
        if (keys[i] != expected_keys[i])
        {
            result.ok = false;
            result.message = "extracted key mismatch at step " + std::to_string(i);
            return result;
        }
    }
    return result;
}

int main(int argc, char **argv)
{
    CliOptions options;
    if (!parse_args(argc, argv, options))
    {
        print_usage(argv[0]);
        return 2;
    }

    if (options.csv_header_only)
    {
        print_csv_header();
        return 0;
    }

    const std::size_t n = options.n;

    std::mt19937_64 rng(1337);
    std::uniform_real_distribution<double> key_dist(0.0, 1'000'000.0);

    std::vector<double> initial_keys;
    initial_keys.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        initial_keys.push_back(key_dist(rng) + static_cast<double>(i) * 1e-9);
    }

    std::vector<double> mutable_keys = initial_keys;
    const std::vector<Update> updates = build_updates(mutable_keys, n / 2, rng);

    std::vector<std::size_t> expected_order;
    std::vector<double> expected_keys;

    const RunResult std_result = run_std_reference(initial_keys, updates, expected_order, expected_keys);
    if (!std_result.ok)
    {
        std::cerr << "Reference run failed: " << std_result.message << '\n';
        return 1;
    }

    std::vector<std::size_t> order;
    std::vector<double> keys;

    const RunResult four_ary_result = run_queue<FourAryHeap>(initial_keys, updates, order, keys);
    if (!four_ary_result.ok)
    {
        std::cerr << "FourAryHeap failed: " << four_ary_result.message << '\n';
        return 1;
    }
    const RunResult four_ary_validation = validate_against_reference(order, keys, expected_order, expected_keys);
    if (!four_ary_validation.ok)
    {
        std::cerr << "FourAryHeap validation failed: " << four_ary_validation.message << '\n';
        return 1;
    }

    const RunResult pairing_result = run_queue<PairingHeap>(initial_keys, updates, order, keys);
    if (!pairing_result.ok)
    {
        std::cerr << "PairingHeap failed: " << pairing_result.message << '\n';
        return 1;
    }
    const RunResult pairing_validation = validate_against_reference(order, keys, expected_order, expected_keys);
    if (!pairing_validation.ok)
    {
        std::cerr << "PairingHeap validation failed: " << pairing_validation.message << '\n';
        return 1;
    }

    const RunResult adapter_result = run_queue<StdPriorityQueueAdapter>(initial_keys, updates, order, keys);
    if (!adapter_result.ok)
    {
        std::cerr << "StdPriorityQueueAdapter failed: " << adapter_result.message << '\n';
        return 1;
    }
    const RunResult adapter_validation = validate_against_reference(order, keys, expected_order, expected_keys);
    if (!adapter_validation.ok)
    {
        std::cerr << "StdPriorityQueueAdapter validation failed: " << adapter_validation.message << '\n';
        return 1;
    }

    const double adapter_vs_std = std_result.ms > 0.0 ? adapter_result.ms / std_result.ms : 0.0;
    const double four_ary_vs_std = std_result.ms > 0.0 ? four_ary_result.ms / std_result.ms : 0.0;
    const double pairing_vs_std = std_result.ms > 0.0 ? pairing_result.ms / std_result.ms : 0.0;

    std::cout << std::fixed << std::setprecision(3);
    if (options.csv)
    {
        std::cout << n << ','
                  << updates.size() << ','
                  << std_result.ms << ','
                  << adapter_result.ms << ','
                  << four_ary_result.ms << ','
                  << pairing_result.ms << ','
                  << adapter_vs_std << ','
                  << four_ary_vs_std << ','
                  << pairing_vs_std << ','
                  << "ok" << '\n';
    }
    else
    {
        std::cout << "Basic benchmark + validation" << '\n';
        std::cout << "  elements: " << n << '\n';
        std::cout << "  decrease-key ops: " << updates.size() << '\n';
        std::cout << "  std::priority_queue reference: " << std_result.ms << " ms" << '\n';
        std::cout << "  StdPriorityQueueAdapter: " << adapter_result.ms << " ms"
                  << " (x" << adapter_vs_std << " vs std)" << '\n';
        std::cout << "  FourAryHeap: " << four_ary_result.ms << " ms"
                  << " (x" << four_ary_vs_std << " vs std)" << '\n';
        std::cout << "  PairingHeap: " << pairing_result.ms << " ms"
                  << " (x" << pairing_vs_std << " vs std)" << '\n';
        std::cout << "Validation passed" << '\n';
    }
    return 0;
}
