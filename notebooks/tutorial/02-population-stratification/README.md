## Population Stratification


### Hail Dataset Notes

The [load_dataset](https://hail.is/docs/0.2/experimental/index.html#hail.experimental.load_dataset) function takes a configuration file (```gs://hail-datasets/datasets.json``` by default) that determines where other public datasets are hosted in gs.  This points spark at "gs://" urls, but it should be 
possible to either browse [https://console.cloud.google.com/storage/browser/hail-datasets](https://console.cloud.google.com/storage/browser/hail-datasets)
and manually fetch things or figure out how to point that method at a custom config that doesn't require the gs protocol.

All of links in https://storage.googleapis.com/hail-datasets/datasets.json currently point to paths like ```gs://hail-datasets-hail-data/DANN.GRCh38.ht```

Hail 1KG Data: https://console.cloud.google.com/storage/browser/hail-datasets/1000_genomes.phase3.GRCh38.mt

### 1KG Data Notes

- ALL.2of4intersection.20100804.genotypes
    - count = (25488488, 629)
    - 10375501 variants have rsid "."
    - 15112987 are unique ids

- ALL.2of4intersection.20100804.genotypes_no_missing_IDs
    - count = (25488488, 629)
    - All rsids unique
    - code:
    ```
    mt = hl.read_matrix_table(osp.join(data_dir, 'ALL.2of4intersection.20100804.genotypes_no_missing_IDs.mt'))
    rows = mt.rows()
    cts = rows.group_by(rows.rsid).aggregate(n=hl.agg.count())
    cts.group_by(cts.n).aggregate(nn=hl.agg.count()).show()
    # shows 1 -> 25488488
    ```