
## commands in postgresql

```bash
# Create tables.


# using GCP SQL
cd ~/MIAD_Final_project/postgresql/postgres/

psql -h 34.173.85.2 -U postgres
#(password = rw,.12a)
# execute sql file
\i init_db.sql 


# using docker
cp cd MIAD_Final_project/postgresql/postgres/init_db.sql /data/postgres/
docker exec -it postgres bash -c "su - postgres"
psql
# execute sql file
\i /data/init_db.sql 


```