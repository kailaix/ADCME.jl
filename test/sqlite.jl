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