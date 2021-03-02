@testset "sqlite" begin 
    db = Database()
    execute(db, """
    CREATE TABLE simulation_parameters (
        desc text,
        rho real, 
        gamma real, 
        dt real, 
        h real 
    )
    """)
    execute(db, """
    INSERT INTO simulation_parameters VALUES 
    ('baseline', 1.0, 2.0, 0.01, 0.02)
    """)
    params = [
        ("compare1", 1.0, 2.0, 0.01),
        ("compare2", 2.0, 3.0, 4.0)
    ]
    execute(db, """
    INSERT INTO simulation_parameters VALUES 
    (?, ?, ?, ?, 0.1)
    """, params)
    res = execute(db, "SELECT * FROM simulation_parameters")|>collect

    @test keys(db)==["simulation_parameters"]
    @test keys(db, "simulation_parameters")==[ "desc"
                                                "rho"
                                                "gamma"
                                                "dt"
                                                "h"]
    @test length(res)==3

    commit(db)
    close(db)
end

@testset "execute many" begin 
    db = Database()
    execute(db, 
"""
CREATE TABLE temp (
    t1 real, 
    t2 real 
)
""")
    execute(db, "INSERT INTO temp VALUES (?,?)", rand(10), rand(10))
    @test length(db["temp"])==10
end


@testset "sqlite do syntax" begin 
    db = Database("temp.db")
    execute(db) do 
"""
CREATE TABLE simulation_parameters (
    desc text,
    rho real, 
    gamma real, 
    dt real, 
    h real 
    )
"""
    end 

    @test "simulation_parameters" in keys(db)
    
    close(db)
    rm("temp.db", force=true)

    Database("temp.db") do db 
        execute(db) do 
            """
                CREATE TABLE simulation_parameters (
                    desc text,
                    rho real, 
                    gamma real, 
                    dt real, 
                    h real 
                    )
            """
        end
    end

    @test length(db["simulation_parameters"])==0
    delete!(db, "simulation_parameters")
    @test_throws PyCall.PyError keys(db, "simulation_parameters")

    db = Database("temp.db")
    @test "simulation_parameters" in keys(db)
    close(db)
    rm("temp.db", force=true)
end