
## Create tables.

```bash

# using GCP SQL
cd ~/MIAD_Final_project/postgresql

psql -h 34.173.85.2 -U postgres
#(password = rw,.12a)
# execute sql file
\i init.sql 

```

#### Useful commands


```bash
# list databases
\l

# list tables
\dt

# describe table
\d table_name
\d+ table_name

# command history
\s

```