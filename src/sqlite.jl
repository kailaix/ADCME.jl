export Database, execute, commit
mutable struct Database
    conn::PyObject
    c::PyObject
    sqlite3::PyObject
    commit_after_execute::Bool 
    filename::String 
end

"""
    Database(filename::Union{Missing, String} = missing; 
        commit_after_execute::Bool = true)

Creates a database from `filename`. If `filename` is not provided, an in-memory database is created. 
If `commit_after_execute` is false, no commit operation is performed after each [`execute`](@ref).

- do block syntax:
```julia 
Database() do db
    execute(db, "create table mytable (a real, b real)")
end
```
The database is automatically closed after execution. Therefore, if execute is a query operation, 
users need to store the results in a global variable. 

- Query meta information 
```julia 
keys(db) # all tables 
keys(db, "mytable") # all column names in `db.mytable` 
```
"""
function Database(filename::Union{Missing, String} = missing; 
    commit_after_execute::Bool = true)
    filename = coalesce(filename, ":memory:")
    sqlite3 = pyimport("sqlite3")
    conn = sqlite3.connect(filename)
    c = conn.cursor()
    Database(conn, c, sqlite3, commit_after_execute, filename)
end

function Database(f::Function, args...; kwargs...)
    db = Database(args...;kwargs...)
    out = f(db)
    close(db)
    out 
end

"""
    execute(db::Database, sql::String, args...)

Executes the SQL statement `sql` in `db`. Users can also use the do block syntax. 
```julia 
execute(db) do 
    "create table mytable (a real, b real)"
end
```

`execute` can also be used to insert a batch of records
```julia 
t1 = rand(10)
t2 = rand(10)
param = collect(zip(t1, t2))
execute(db, "INSERT TO mytable VALUES (?,?)", param)
```
or 
```julia
execute(db, "INSERT TO mytable VALUES (?,?)", t1, t2)
```
"""
function execute(db::Database, sql::String, args...)
    if length(args)>=1
        if length(args)>1
            param = collect(zip(args...))
        else
            param = args[1]
        end
        db.c.executemany(sql, param)
    else
        db.c.execute(sql)
    end
    if db.commit_after_execute
        commit(db)
    end
    db.c
end

function execute(f::Function, db::Database; kwargs...)
    sql = f()
    execute(db, sql; kwargs...)
end

"""
    commit(db::Database)

Commits changes to `db`.
"""
function commit(db::Database)
    db.conn.commit()
end


function Base.:close(db::Database)
    commit(db)
    db.conn.close()
end

function Base.:keys(db::Database)
    ret = execute(db, "select name from sqlite_master")|>collect
    tables = String[]
    for k = 1:length(ret)
        push!(tables, ret[k][1])
    end
    tables
end

function Base.:keys(db::Database, table::String)
    execute(db, "select * from $table limit 1")
    ret = db.c.description
    columns = String[]
    for r in ret 
        push!(columns, r[1])
    end
    columns
end

function Base.:delete!(db::Database, table::String)
    execute(db, "drop table $table")
    commit(db)
end

function Base.:getindex(db::Database, table::String)
    c = execute(db, "select * from $table")
    collect(c)
end