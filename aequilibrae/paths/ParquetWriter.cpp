#include <iostream>
#include <vector>
#include <string>

#include <arrow/python/pyarrow.h>
#include <arrow/api.h>
#include <parquet/arrow/writer.h>
#include <arrow/io/file.h>

#include "ParquetWriter.h"

using arrow::Int64Builder;

ParquetWriter::ParquetWriter() {}
ParquetWriter::~ParquetWriter() {}


#define EXIT_ON_FAILURE(expr)                      \
  do {                                             \
    arrow::Status status_ = (expr);                \
    if (!status_.ok()) {                           \
      std::cerr << status_.message() << std::endl; \
      return EXIT_FAILURE;                         \
    }                                              \
  } while (0);


int ParquetWriter::write_parquet(std::vector<int64_t> vec, std::string filename)
{
    // TODO: sort out types
    // std::vector<int64_t> vec_ = static_cast<int64_t>(vec);
    std::shared_ptr<arrow::Table> table;
    EXIT_ON_FAILURE(VectorToColumnarTable(vec, &table));

    std::shared_ptr<arrow::io::FileOutputStream> outfile;
    PARQUET_ASSIGN_OR_THROW(
        outfile,
        arrow::io::FileOutputStream::Open(filename));

    PARQUET_THROW_NOT_OK(
        parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, 1));

    return 1;
}


// Transforming a vector of structs into a columnar Table.
//
// The final representation should be an `arrow::Table` which in turn
// is made up of an `arrow::Schema` and a list of
// `arrow::ChunkedArray` instances. As the first step, we will iterate
// over the data and build up the arrays incrementally.  For this
// task, we provide `arrow::ArrayBuilder` classes that help in the
// construction of the final `arrow::Array` instances.
//
// For each type, Arrow has a specially typed builder class. For the primitive
// values `id` and `cost` we can use the respective `arrow::Int64Builder` and
// `arrow::DoubleBuilder`. For the `cost_components` vector, we need to have two
// builders, a top-level `arrow::ListBuilder` that builds the array of offsets and
// a nested `arrow::DoubleBuilder` that constructs the underlying values array that
// is referenced by the offsets in the former array.
arrow::Status ParquetWriter::VectorToColumnarTable(const std::vector<int64_t>& rows,
                                    std::shared_ptr<arrow::Table>* table) {
    // The builders are more efficient using
    // arrow::jemalloc::MemoryPool::default_pool() as this can increase the size of
    // the underlying memory regions in-place. At the moment, arrow::jemalloc is only
    // supported on Unix systems, not Windows.
    arrow::MemoryPool* pool = arrow::default_memory_pool();

    Int64Builder id_builder(pool);

//    // Now we can loop over our existing data and insert it into the builders. The
//    // `Append` calls here may fail (e.g. we cannot allocate enough additional memory).
//    // Thus we need to check their return values. For more information on these values,
//    // check the documentation about `arrow::Status`.
//    for (const data_row& row : rows) {
//    ARROW_RETURN_NOT_OK(id_builder.Append(row.id));
//    }

//    // At the end, we finalise the arrays, declare the (type) schema and combine them
//    // into a single `arrow::Table`:
//    std::shared_ptr<arrow::Array> id_array;
//    ARROW_RETURN_NOT_OK(id_builder.Finish(&id_array));

    // Make place for 8 values in total
    id_builder.Resize(rows.size());
    id_builder.AppendValues(rows);

    std::shared_ptr<arrow::Array> id_array;
    arrow::Status st = id_builder.Finish(&id_array);



    std::vector<std::shared_ptr<arrow::Field>> schema_vector = {arrow::field("link_id", arrow::int64())};

    auto schema = std::make_shared<arrow::Schema>(schema_vector);
    // The final `table` variable is the one we then can pass on to other functions
    // that can consume Apache Arrow memory structures. This object has ownership of
    // all referenced data, thus we don't have to care about undefined references once
    // we leave the scope of the function building the table and its underlying arrays.
    *table = arrow::Table::Make(schema, {id_array});

    return arrow::Status::OK();
}
