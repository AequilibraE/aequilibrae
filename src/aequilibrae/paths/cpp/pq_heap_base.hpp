#pragma once

#include <cstddef>
#include <limits>

namespace aequilibrae::paths::cpp
{

    enum ElementState
    {
        SCANNED,
        NOT_IN_HEAP,
        IN_HEAP,
    };

    inline constexpr double kInfinity = std::numeric_limits<double>::infinity();

    template <class Derived>
    class PriorityQueueBase
    {
    public:
        PriorityQueueBase(const PriorityQueueBase &) = delete;
        PriorityQueueBase &operator=(const PriorityQueueBase &) = delete;
        PriorityQueueBase(PriorityQueueBase &&) = delete;
        PriorityQueueBase &operator=(PriorityQueueBase &&) = delete;

        void init_heap(std::size_t length) noexcept
        {
            derived().init_heap_impl(length);
        }

        void alloc_heap(std::size_t length) noexcept
        {
            derived().alloc_heap_impl(length);
        }

        void reset_heap() noexcept
        {
            derived().reset_heap_impl();
        }

        void free_heap() noexcept
        {
            derived().free_heap_impl();
        }

        void insert(std::size_t element_idx, double key) noexcept
        {
            derived().insert_impl(element_idx, key);
        }

        void decrease_key(std::size_t element_idx, double key_new) noexcept
        {
            derived().decrease_key_impl(element_idx, key_new);
        }

        [[nodiscard]] double peek() const noexcept
        {
            return derived().peek_impl();
        }

        [[nodiscard]] bool is_empty() const noexcept
        {
            return derived().is_empty_impl();
        }

        [[nodiscard]] std::size_t extract_min() noexcept
        {
            return derived().extract_min_impl();
        }

        [[nodiscard]] ElementState effective_state(std::size_t element_idx) const noexcept
        {
            return derived().effective_state_impl(element_idx);
        }

        [[nodiscard]] double element_key(std::size_t element_idx) const noexcept
        {
            return derived().element_key_impl(element_idx);
        }

    protected:
        PriorityQueueBase() = default;
        ~PriorityQueueBase() = default;

    private:
        Derived &derived() noexcept
        {
            return *static_cast<Derived *>(this);
        }

        const Derived &derived() const noexcept
        {
            return *static_cast<const Derived *>(this);
        }
    };

} // namespace aequilibrae::paths::cpp