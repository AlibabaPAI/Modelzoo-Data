import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

file_name = 'eval'
src_file = 'kaggle/' + file_name + '.csv'
dst_file = './' + file_name + '.parquet'

chunksize=10000 # this is the number of lines

# Definition of some constants
LABEL_COLUMNS = ['clicked']
CONTINUOUS_COLUMNS = ['I' + str(i) for i in range(1, 14)]  # 1-13 inclusive
CATEGORICAL_COLUMNS = ['C' + str(i) for i in range(1, 27)]  # 1-26 inclusive
INPUT_COLUMNS = LABEL_COLUMNS + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS

label_dtype = {item : int for item in LABEL_COLUMNS}
continuous_dtype = {item : float for item in CONTINUOUS_COLUMNS}
categorical_dtype = {item : str for item in CATEGORICAL_COLUMNS}
input_dtype = {}
input_dtype.update(label_dtype)
input_dtype.update(continuous_dtype)
input_dtype.update(categorical_dtype)

label_field = [pa.field(item, pa.int32()) for item in LABEL_COLUMNS]
continuous_field = [pa.field(item, pa.float32()) for item in CONTINUOUS_COLUMNS]
categorical_field = [pa.field(item, pa.string()) for item in CATEGORICAL_COLUMNS]
input_field = label_field + continuous_field + categorical_field

label_default_values = {item : 0 for item in LABEL_COLUMNS}
continuous_default_values = {item : 0.0 for item in CONTINUOUS_COLUMNS}
categorical_default_values = {item : ' ' for item in CATEGORICAL_COLUMNS}
default_values = {}
default_values.update(label_default_values)
default_values.update(continuous_default_values)
default_values.update(categorical_default_values)

schema = pa.schema(input_field)

pqwriter = pq.ParquetWriter(dst_file, schema)            
for i, df in enumerate(pd.read_csv(src_file, \
                                   chunksize=chunksize, \
                                   names=INPUT_COLUMNS,
                                   dtype=input_dtype)):
    df = df.fillna(default_values)
    table = pa.Table.from_pandas(df, schema)
    pqwriter.write_table(table)

# close the parquet writer
if pqwriter:
    pqwriter.close()

output_table = pq.read_table(dst_file)
print(output_table.to_pandas())
