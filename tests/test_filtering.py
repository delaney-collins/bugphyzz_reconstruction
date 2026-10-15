from bugphyzz.preprocessing.filtering import filter_data
from bugphyzz.taxonomy.frozen import FrozenTaxonomyProvider


taxonomy = FrozenTaxonomyProvider.from_tsv(
    load_attributes=True,
)

filtered = filter_data(
    raw_trait_table,
    taxonomy=taxonomy,
)

import pandas as pd
from pandas.testing import assert_frame_equal


r_reference = pd.read_csv(
    "01_filtered.tsv",
    sep="\t",
    dtype={
        "NCBI_ID": "string",
        "Parent_NCBI_ID": "string",
    },
    na_values=["NA"],
)

python_result = filter_data(
    raw_trait_table,
    taxonomy=taxonomy,
)

assert_frame_equal(
    python_result.reset_index(drop=True),
    r_reference.reset_index(drop=True),
    check_dtype=False,
    check_like=False,
)