

## run ansible
ansible-playbook -i ../inventory/staging.yml miad_vms.yml

 
## commands in postgresql

```bash
# Create tables.

docker exec -it postgres bash -c "su - postgres"
psql
# execute sql file
\i /data/init_db.sql 

```