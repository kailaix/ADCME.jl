# Introducing ADCME Database and SQL Integration: an Efficient Approach to Simulation Data Management


## Introduction 

If you have a massive number of simulations and results from different simulation parameters, to facilitate the data analysis and improve reproducibility, database and  Structured Query Language (SQL) are convenient and powerful ways for data management. 

Database allows simulation parameters and results to be stored in a permanent storage, and  records can be inserted, queried, updated, and deleted as we proceed in our research. Specifically, databases are usually designed in a way that we can concurrently read and write in a transactional manner, which ensures that the reads are writes are done correctly even in the case of data conflicts. This characteristic is very useful for parallel simulations. Another important feature of databases is "indexing". By indexing tables, we can manipute tables in a more efficient way. 

![](https://github.com/ADCMEMarket/ADCMEImages/blob/master/ADCME/sqlite.png?raw=true)


SQL is a standad language for accessing and manipulating databases. The four main operations in SQLs are: create, insert, update, and delete. More advanced commands include `where`, `groupby`, `join`, etc. In ADCME, we implemented an interface to SQLite, a relational database management system contained in a C library. SQLite provides basic SQL engines, which is compliant to the SQL standard. One particular feature of SQLite is that the database is a single file or in-memory. This simplifies the client and server SQL logic, but bears the limitation of scalability. Nevertheless, SQLite is more than sufficient to store and manipulate our simulation parameters and results (typically a link to the data folder). 

The introduction primarily focuses on some commonly used features of database management in ADCME. 


## Database Structure

In ADCME, a database is created using [`Database`](@ref). There are two types of database:

```julia
db1 = Database() # in-memory 
db2 = Database("simulation.db") # file 
```

If you created a file-based database, whenever you finished operation, you need to [`commit`](@ref) to the database or [`close`](@ref) the database to make the changes effective. 

```julia
commit(db2)
close(db2)
```

To execute a SQL command in the database, we can use [`execute`](@ref) command. In the next section, we list some commonly used operations. By default, `commit` is called after `execute`. Users can disable this by using 
```julia
db1 = Database(commit_after_execute=false) # in-memory 
db2 = Database("simulation.db", commit_after_execute=false) # file 
```

## Commonly Used Operations

### Create a Database
```julia
execute(db2, """
CREATE TABLE simulation_parameters (
    name real primary key,
    dt real,
    h real, 
    result text 
)
""")
```

### Insert an Record

```julia
execute(db2, """
INSERT INTO simulation_parameters VALUES
("sim1", 0.1, 0.01, "file1.png")
""")
```

### Insert Many Records

```julia
params = [
    ("sim2", 0.3, "file2.png"),
    ("sim3", 0.5, "file3.png"),
    ("sim4", 0.9, "file4.png")
]
execute(db2, """
INSERT INTO simulation_parameters VALUES
(?, ?, 0.01, ?)
""", params)
```

### Look Up Records
```julia
c = execute(db2, """
SELECT * from simulation_parameters
""")
collect(c)
```

Expected output:
```
("sim1", 0.1, 0.01, "file1.png")
("sim2", 0.3, 0.01, "file2.png")
("sim3", 0.5, 0.01, "file3.png")
("sim4", 0.9, 0.01, "file4.png")
```


### Delete a Record
```julia
execute(db2, """
DELETE from simulation_parameters WHERE name LIKE "%3"
""")
```

Now the records are
```
("sim1", 0.1, 0.01, "file1.png")
("sim2", 0.3, 0.01, "file2.png")
("sim4", 0.9, 0.01, "file4.png")
```


### Update a Record
```julia
execute(db2, """
UPDATE simulation_parameters
SET h = 0.2
WHERE name = "sim4"
""")
```
Now the records are 
```
("sim1", 0.1, 0.01, "file1.png")
("sim2", 0.3, 0.01, "file2.png")
("sim4", 0.9, 0.2, "file4.png")
```

### Insert a Conflict Record 

Becase we set `name` as primary key, we cannot insert a record with the same name

```julia
execute(db2, """
INSERT INTO simulation_parameters VALUES
("sim1", 0.1, 0.1, "file1_2.png")
""")
```

We have an error 
```
IntegrityError('UNIQUE constraint failed: simulation_parameters.name')
```

Alternatively, we can do 

```julia
execute(db2, """
INSERT OR IGNORE INTO simulation_parameters VALUES
("sim1", 0.1, 0.1, "file1_2.png")
""")
```

### Querying Meta Data

- Get all tables in the database
```julia
keys(db2)
```

Output:
```
"simulation_parameters"
"sqlite_autoindex_simulation_parameters_1"
```
We can see a table that SQLites adds indexes to some fields: `sqlite_autoindex_simulation_parameters_1`. 

- Get column names in the database
```julia
keys(db2, "simulation_parameters")
```

Output:
```
"name"
"dt"
"h"
"result"
```
### Drop a Table 
```julia
execute(db2, """
DROP TABLE simulation_parameters
""")
```

## Next Steps

We introduced some basic usage of ADCME database and SQL integration for simulation data management. We used one common and lightweight database management system, SQLite, for managing data of moderate sizes. Due to SQL's wide adoption, it is possible to scale the data management system by adopting a full-fledged database, such as MySQL. In this case, developers can overload ADCME functions such as `execute`, `commit`, `close`, `keys`, etc., so that the top level codes requires little changes. 