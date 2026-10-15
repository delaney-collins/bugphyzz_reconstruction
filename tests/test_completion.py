from bugphyzz.preprocessing.completion import (
    complete_binary_data,
    complete_multistate_union,
)


binary_output = complete_binary_data(
    combined_binary_dataset
)

union_outputs = complete_multistate_union(
    filtered_union_table,
    taxonomy,
)

for attribute_name, prepared_table in union_outputs.items():
    print(attribute_name)
    print(prepared_table.head())