#ifndef PARQUETWRITER_H
#define PARQUETWRITER_H

#include <iostream>
#include <vector>
#include <string>

#include <arrow/api.h>
#include <parquet/arrow/writer.h>
#include <arrow/io/file.h>

//#include <arrow/io/api.h>
#include <arrow/ipc/feather.h>





class ParquetWriter {

public:
    ParquetWriter();
    ~ParquetWriter();

    int write_parquet(std::vector<int64_t> vec, std::string filename);
    int write_feather(std::vector<int64_t> vec, std::string filename);

private:
    arrow::Status VectorToColumnarTable(const std::vector<int64_t>& rows,
                                    std::shared_ptr<arrow::Table>* table);
};

#endif

