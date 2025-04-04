CREATE VIEW lineitem AS SELECT * FROM read_parquet('/home/windy/postgres/duckdbDatasets/lineitem.parquet');        
CREATE VIEW orders AS SELECT * FROM read_parquet('/home/windy/postgres/duckdbDatasets/orders.parquet');        
CREATE VIEW partsupp AS SELECT * FROM read_parquet('/home/windy/postgres/duckdbDatasets/partsupp.parquet');        
CREATE VIEW part AS SELECT * FROM read_parquet('/home/windy/postgres/duckdbDatasets/part.parquet');        
CREATE VIEW customer AS SELECT * FROM read_parquet('/home/windy/postgres/duckdbDatasets/customer.parquet');        
CREATE VIEW supplier AS SELECT * FROM read_parquet('/home/windy/postgres/duckdbDatasets/supplier.parquet');        
CREATE VIEW nation AS SELECT * FROM read_parquet('/home/windy/postgres/duckdbDatasets/nation.parquet');        
CREATE VIEW region AS SELECT * FROM read_parquet('/home/windy/postgres/duckdbDatasets/region.parquet');


