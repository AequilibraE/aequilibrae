#ifndef PARQUETWRITER_H
#define PARQUETWRITER_H

#include <iostream>
#include <vector>
#include <string>

//#include <arrow/python/pyarrow.h>
#include <arrow/api.h>
#include <parquet/arrow/writer.h>
#include <arrow/io/file.h>





class ParquetWriter {

public:
    ParquetWriter();
    ~ParquetWriter();

    int write_parquet(std::vector<int64_t> vec, std::string filename);

private:
    arrow::Status VectorToColumnarTable(const std::vector<int64_t>& rows,
                                    std::shared_ptr<arrow::Table>* table);
};

#endif

