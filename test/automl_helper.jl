using ADCME
reset_default_graph()

WORKSPACE = ARGS[1]
name = ARGS[2]
rep = parse(Bool, ARGS[3])
previous_ensembles = strip.(readlines("$WORKSPACE/ensembles.txt"))

@info  WORKSPACE, name
@info previous_ensembles


x = reshape(Array(LinRange(-1,1,100)), :, 1)
y = @. sin(2x)

x0 = reshape(rand(100)*2 .- 1, :, 1)
y0 = @. sin(2x0)

function create_neural_network(name)
    num_layers = parse(Int64, name)
    nn = x->fc(x, [num_layers,num_layers,num_layers,1], "nn$name")
    return nn 
end

function compute_loss_function(xp, yp, nn)
    xp, yp = constant(xp), constant(yp)
    loss = sum((nn(xp) - yp)^2)
    return loss
end

nn = create_neural_network(name)
loss = compute_loss_function(x, y, nn)
sess = Session(); init(sess)
if !isfile("$WORKSPACE/$name/data.mat")
    global loss0 = BFGS!(sess, loss)
    ADCME.save(sess, "$WORKSPACE/$name/data.mat")
else
    @info "\"$WORKSPACE/$name/data.mat\" exists." 
end

ensemble_names = filter(x->length(x)>0, String[previous_ensembles;name])
if rep 
    ensemble_names = ensemble_names[1:end-1]
end
@info "ensemble_names = ", ensemble_names
loss2 = average_ensemble(sess, ensemble_names, create_neural_network, nn->compute_loss_function(x0, y0, nn), WORKSPACE)
loss0 = run(sess, loss)
loss_ = run(sess, loss2)
println("standalone loss >>> $(loss0[end]) <<<")
println("ensemble loss >>> $(loss_) <<<")
