import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

path_prefix = './amazon/'
file_name = 'local_train_splitByUser'
src_file = path_prefix + file_name
dst_file = './' + file_name + '.parquet'
src_neg_file = path_prefix + file_name + '_neg'
dst_neg_file = './' + file_name + '_neg.parquet'

chunksize=10000 # this is the number of lines

# Definition of some constants
LABEL_COLUMNS = ['CLICKED']
UNSEQ_COLUMNS = ['UID', 'ITEM', 'CATEGORY']
HIS_COLUMNS = ['HISTORY_ITEM', 'HISTORY_CATEGORY']
NONEG_COLUMNS = LABEL_COLUMNS + UNSEQ_COLUMNS + HIS_COLUMNS
NEG_COLUMNS = ['NOCLK_HISTORY_ITEM', 'NOCLK_HISTORY_CATEGORY']

label_dtype = {item : int for item in LABEL_COLUMNS}
unseq_dtype = {item : str for item in UNSEQ_COLUMNS}
his_dtype = {item : str for item in HIS_COLUMNS}
no_neg_dtype = {}
no_neg_dtype.update(label_dtype)
no_neg_dtype.update(unseq_dtype)
no_neg_dtype.update(his_dtype)
neg_dtype = {item : str for item in NEG_COLUMNS}

label_field = [pa.field(item, pa.int32()) for item in LABEL_COLUMNS]
unseq_field = [pa.field(item, pa.string()) for item in UNSEQ_COLUMNS]
his_field = [pa.field(item, pa.string()) for item in HIS_COLUMNS]
no_neg_field = label_field + unseq_field + his_field
neg_field = [pa.field(item, pa.string()) for item in NEG_COLUMNS]

label_default_values = {item : 0 for item in LABEL_COLUMNS}
unseq_default_values = {item : ' ' for item in UNSEQ_COLUMNS}
his_default_values = {item : ' ' for item in HIS_COLUMNS}
no_neg_default_values = {}
no_neg_default_values.update(label_default_values)
no_neg_default_values.update(unseq_default_values)
no_neg_default_values.update(his_default_values)
neg_default_values = {item : '' for item in NEG_COLUMNS}

def csv_to_parquet(src_file, dst_file, INPUT_COLUMNS, input_dtype, input_field, default_values):
    schema = pa.schema(input_field)

    pqwriter = pq.ParquetWriter(dst_file, schema)            
    for i, df in enumerate(pd.read_csv(src_file, \
                                       chunksize=chunksize, \
                                       names=INPUT_COLUMNS,
                                       dtype=input_dtype,
                                       sep='\t')):
        df = df.fillna(default_values)
        table = pa.Table.from_pandas(df, schema)
        pqwriter.write_table(table)

    # close the parquet writer
    if pqwriter:
        pqwriter.close()

    output_table = pq.read_table(dst_file)
    print(output_table.to_pandas())

# convert src_file
print("convert " + src_file + " ==> " + dst_file)
csv_to_parquet(src_file, dst_file, NONEG_COLUMNS, no_neg_dtype,
               no_neg_field, no_neg_default_values)

# convert src_neg_file
print("convert " + src_neg_file + " ==> " + dst_neg_file)
csv_to_parquet(src_neg_file, dst_neg_file, NEG_COLUMNS, neg_dtype,
               neg_field, neg_default_values)
