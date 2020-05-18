using ADCME
reset_default_graph()

WORKSPACE = ARGS[1]
name = ARGS[2]
rep = parse(Int64, ARGS[3])
previous_ensembles = strip.(readlines("$WORKSPACE/ensembles.txt"))

@info  WORKSPACE, name
@info previous_ensembles


x = reshape(Array(LinRange(-1,1,100)), :, 1)
y = @. sin(2*2π*x)

x0 = reshape(rand(100)*2 .- 1, :, 1)
y0 = @. sin(2*2π*x0)

# function create_neural_network(name)
#     num_layers = parse(Int64, name)
#     nn = x->fc(x, [num_layers,num_layers,num_layers,1], "nn$name")
#     return nn 
# end


function compute_loss_function(xp, yp, nn)
    xp, yp = constant(xp), constant(yp)
    loss = sum((nn(xp) - yp)^2)
    return loss
end

# nn = create_neural_network(name)
nn = ADCME.create_neural_network(name, 1)

loss = compute_loss_function(x, y, nn)
sess = Session(); init(sess)
if !isfile("$WORKSPACE/$name/data.mat")
    BFGS!(sess, loss)
    ADCME.save(sess, "$WORKSPACE/$name/data.mat")
    global loss0 = run(sess, compute_loss_function(x0, y0, nn))
else
    global loss0
    @info ">>>>>>>>>>>>>>>>>>>>>> \"$WORKSPACE/$name/data.mat\" exists. Retraining ..." 
    ADCME.load(sess, "$WORKSPACE/$name/data.mat")
    loss0 = run(sess, compute_loss_function(x0, y0, nn))
    init(sess); BFGS!(sess, loss)
    loss1 = run(sess, compute_loss_function(x0, y0, nn))
    if loss1<loss0
        loss0 = loss1
        ADCME.save(sess, "$WORKSPACE/$name/data.mat")
    else 
        ADCME.load(sess, "$WORKSPACE/$name/data.mat")
    end
end

previous_ensembles = filter(x->length(x)>0, previous_ensembles)
if rep > 0 
    previous_ensembles = String[previous_ensembles[1:rep-1];previous_ensembles[rep+1:end]]
end
ensemble_names = String[previous_ensembles;name]
@info "ensemble_names = ", ensemble_names
loss2 = average_ensemble(sess, ensemble_names, name -> ADCME.create_neural_network(name, 1), nn->compute_loss_function(x0, y0, nn), WORKSPACE)
loss_ = run(sess, loss2)
println("standalone loss >>> $(loss0[end]) <<<")
println("ensemble loss >>> $(loss_) <<<")
