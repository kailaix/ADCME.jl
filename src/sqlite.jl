export Database, execute, commit
mutable struct Database
    conn::PyObject
    c::PyObject
    sqlite3::PyObject
    commit_after_execute::Bool 
    filename::String 
end

function Database(filename::Union{Missing, String} = missing; 
    commit_after_execute::Bool = true)
    filename = coalesce(filename, ":memory:")
    sqlite3 = pyimport("sqlite3")
    conn = sqlite3.connect(filename)
    c = conn.cursor()
    Database(conn, c, sqlite3, commit_after_execute, filename)
end

function execute(db::Database, sql::String, args...)
    if length(args)>=1
        db.c.executemany(sql, args[1])
    else
        db.c.execute(sql)
    end
    if db.commit_after_execute
        commit(db)
    end
    db.c
end

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


function Base.:push!(db::Database, table::String, nts::NamedTuple...)
    v1 = []
    v2 = []
    for nt in nts         
        cols = propertynames(nt)
        push!(v1, join(["\"$c\"" for c in cols], ","))
        push!(v2, join([isnothing(nt[i]) ? "NULL" : "\"$(nt[i])\"" for i = 1:length(nt)], ","))
    end
    v1 = join(v1, ",")
    v2 = join(v2, ",")
   execute(db, 
"""
    INSERT OR REPLACE INTO $table ($v1) VALUES ($v2)
"""
   ) 
end