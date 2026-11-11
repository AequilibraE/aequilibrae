#pragma once

#include <cstddef>
#include <unordered_map>

namespace aequilibrae::utils::cpp {

// Copy parallel arrays into immutable storage; the last pair wins for
// duplicates.
template <typename Key, typename Value> class ArrayHashMap {
public:
  ArrayHashMap(const Key *keys, const Value *values, std::size_t count) {
    entries_.reserve(count);
    for (std::size_t i = 0; i < count; ++i)
      entries_.insert_or_assign(keys[i], values[i]);
  }

  // Write into caller-owned storage. Return the first missing key's position,
  // or count if every key was found. Lookups never insert missing keys.
  std::size_t lookup(const Key *keys, Value *values, std::size_t count) const {
    for (std::size_t i = 0; i < count; ++i) {
      const auto entry = entries_.find(keys[i]);
      if (entry == entries_.end())
        return i;
      values[i] = entry->second;
    }
    return count;
  }

private:
  std::unordered_map<Key, Value> entries_;
};

} // namespace aequilibrae::utils::cpp
