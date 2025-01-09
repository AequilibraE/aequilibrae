import numpy as np
import pandas as pd
import logging
from scipy.sparse import coo_matrix as sp_coo_matrix


cdef class GeneralisedCOODemand:
    def __init__(self, origin_col: str, destination_col: str, nodes_to_indices, shape=None):
        """
        A container for access to the float64 and float32 fields of a data frame.
        """
        self.df = pd.DataFrame(columns=[origin_col, destination_col]).set_index([origin_col, destination_col])
        self.shape = shape
        self.nodes_to_indices = nodes_to_indices

    def add_df(self, dfs: Union[pd.DataFrame, List[pd.DataFrame]], shape=None, fill: float = 0.0):
        """
        Add a DataFrame to the existing ones.

        Expects a DataFrame with a multi-index of (o, d).
        """
        if isinstance(dfs, pd.DataFrame):
            dfs = (dfs,)

        if shape is None and self.shape is None:
            raise ValueError("a shape must be provided initially to prevent oddly sized sparse matrices")
        if shape is not None and self.shape is None:
            self.shape = shape
        if shape is not None and self.shape is not None and shape != self.shape:
            raise ValueError(f"provided shape ({shape}) differs from previous shape ({self.shape})")

        new_dfs = [self.df]
        for df in dfs:
            if df.index.nlevels != 2:
                raise ValueError("provided pd.DataFrame doesn't have a 2-level multi-index")
            elif df.index.names != self.df.index.names:
                raise ValueError(f"mismatched index names. Expect {self.df.index.names}, provided {df.index.names}")

            shape = self.nodes_to_indices[df.index.to_frame(index=False)].max(axis=0)
            if shape[0] >= self.shape[0] or shape[1] >= self.shape[1]:
                raise ValueError(f"inferred max index ({(shape[0], shape[1])}) exceeds provided shape ({self.shape})")

            new_dfs.append(df.select_dtypes(["float64", "float32"]))

        self.df = pd.concat(new_dfs, axis=1).fillna(fill).sort_index(level=0)

    def add_matrix(self, matrix: AequilibraeMatrix, shape=None, fill: float = 0.0):
        """
        Add an AequilibraE matrix to the existing demand in a sparse manner.
        """
        dfs = []
        for i, name in enumerate(matrix.view_names):
            assert name not in self.df.columns, f"Matrix name ({name}) already exists in the matrix cube"
            m = matrix.matrix_view if len(matrix.view_names) == 1 else matrix.matrix_view[:, :, i]
            if np.nansum(m) == 0:
                continue

            coo_ = sp_coo_matrix(m)
            df = pd.DataFrame(
                data={
                    self.df.index.names[0]: matrix.index[coo_.row],
                    self.df.index.names[1]: matrix.index[coo_.col],
                    name: coo_.data,
                },
            ).set_index([self.df.index.names[0], self.df.index.names[1]])
            dfs.append(df.dropna())

        self.add_df(dfs, shape=shape, fill=fill)
        logging.info(f"There are {len(self.df):,} OD pairs with non-zero flows")

    def _initalise_col_names(self):
        """
        This function is a bit of a hack to allow keeping the same LinkLoadingResults between batched result
        writes. We just need to know how many columns of each type there are
        """
        self.f64_names = []
        self.f32_names = []
        for col in self.df:
            if self.df.dtypes[col] == "float64":
                self.f64_names.append(col)
            elif self.df.dtypes[col] == "float32":
                self.f32_names.append(col)
            else:
                raise TypeError(f"non-floating point column ({col}) in df. Something has gone wrong")

    def _initalise_c_data(self, df = None):
        if df is None:
            df = self.df
        else:
            assert all(self.df.columns == df.columns)

        self.ods = df.index  # MultiIndex[int, int] -> vector[pair[long long, long long]]

        self.f64.clear()
        self.f32.clear()

        cdef:
            double[::1] f64_array
            float[::1] f32_array
            vector[double] *f64_vec
            vector[float] *f32_vec

        for col in df:
            if df.dtypes[col] == "float64":
                f64_array = df[col].to_numpy()

                # The unique pointer will take ownership of this allocation
                f64_vec = new vector[double]()
                f64_vec.insert(f64_vec.begin(), &f64_array[0], &f64_array[0] + len(f64_array))
                # From here f63_vec should not be accessed. It is owned by the unique pointer
                self.f64.emplace_back(f64_vec)

            elif df.dtypes[col] == "float32":
                f32_array = df[col].to_numpy()

                # The unique pointer will take ownership of this allocation
                f32_vec = new vector[float]()
                f32_vec.insert(f32_vec.begin(), &f32_array[0], &f32_array[0] + len(f32_array))
                # From here f32_vec should not be accessed. It is owned by the unique pointer
                self.f32.emplace_back(f32_vec)
            else:
                raise TypeError(f"non-floating point column ({col}) in df. Something has gone wrong")

    def no_demand(GeneralisedCOODemand self) -> bool:
        return len(self.df.columns) == 0

    def is_empty(self) -> bool:
        return self.df.index.empty

    def batches(self):
        self.df = self.df.sort_index(level=0)
        return self.df.groupby(level=0)  # Group by the origin in the multi-index
