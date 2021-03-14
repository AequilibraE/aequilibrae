#include "ParquetWriter.h"

#include <iostream>
#include <vector>
#include <string>

#include <arrow/api.h>
//#include <parquet/arrow/writer.h>
#include <arrow/io/file.h>
#include <arrow/io/api.h>
#include <arrow/ipc/feather.h>

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


//int ParquetWriter::write_parquet(std::vector<int64_t> vec, std::string filename)
//{
//    std::shared_ptr<arrow::Table> table;
//    EXIT_ON_FAILURE(VectorToColumnarTable(vec, &table));
//
//    std::shared_ptr<arrow::io::FileOutputStream> outfile;
//    PARQUET_ASSIGN_OR_THROW(
//        outfile,
//        arrow::io::FileOutputStream::Open(filename));
//
//    PARQUET_THROW_NOT_OK(
//        parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), outfile, 1));
//
//    PARQUET_THROW_NOT_OK(outfile->Close());
//
//    return 1;
//}

// Jan 14/3/21: ARROW macros give me compile errors, parquet macros work. For now, define your own
// macro for error checking, but look into problems a bit more

#define A_ASSIGN_OR_RAISE_IMPL(result_name, lhs, rexpr)          \
  do {                                                           \
    auto&& result_name = (rexpr);                                \
    if (!result_name.ok()) {                                     \
      std::cerr << "Feather conversion gone wrong" << std::endl; \
      return EXIT_FAILURE;                                       \
    }                                                            \
  } while (0);

#define A_ASSIGN_OR_RAISE(lhs, rexpr)                                              \
  A_ASSIGN_OR_RAISE_IMPL(ARROW_ASSIGN_OR_RAISE_NAME(_error_or_value, __COUNTER__), \
                             lhs, rexpr);

#define A_RETURN_NOT_OK(status)                               \
  do {                                                        \
    if (!status.ok()) {                                       \
      std::cerr << "Feather writing gone wrong" << std::endl; \
      return EXIT_FAILURE;                                    \
    }                                                         \
  } while (0);


int ParquetWriter::write_feather(std::vector<int64_t> vec, std::string filename)
{
    std::shared_ptr<arrow::Table> table;
    EXIT_ON_FAILURE(VectorToColumnarTable(vec, &table));

    std::shared_ptr<arrow::io::FileOutputStream> outfile;

    A_ASSIGN_OR_RAISE(outfile, arrow::io::FileOutputStream::Open(filename));

    A_RETURN_NOT_OK(arrow::ipc::feather::WriteTable(*table, outfile.get()));

    A_RETURN_NOT_OK(outfile->Close());

    return 1;
}

arrow::Status ParquetWriter::VectorToColumnarTable(const std::vector<int64_t>& rows,
                                    std::shared_ptr<arrow::Table>* table) {
    // from apache arrow docs:
    // The builders are more efficient using
    // arrow::jemalloc::MemoryPool::default_pool() as this can increase the size of
    // the underlying memory regions in-place. At the moment, arrow::jemalloc is only
    // supported on Unix systems, not Windows.
    arrow::MemoryPool* pool = arrow::default_memory_pool();

    Int64Builder id_builder(pool);
    id_builder.Resize(rows.size());
    id_builder.AppendValues(rows);

    std::shared_ptr<arrow::Array> id_array;
    arrow::Status st = id_builder.Finish(&id_array);

    std::vector<std::shared_ptr<arrow::Field>> schema_vector = {arrow::field("link_id", arrow::int64())};
    auto schema = std::make_shared<arrow::Schema>(schema_vector);

    *table = arrow::Table::Make(schema, {id_array});
    return arrow::Status::OK();
}
