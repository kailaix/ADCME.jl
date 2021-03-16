# reference: https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/

using PyCall
gym = pyimport("gym")

env = gym.make("Taxi-v3").env


# Training 

Q = zeros(env.observation_space.n, env.action_space.n)

alpha = 0.1
gamma = 0.6
epsilon = 0.1

all_epochs = []
all_penalties = []

for i = 1:10000
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = false
    
    while !done 
        if rand() < epsilon
            action = env.action_space.sample() # Explore action space
        else
            action = argmax(Q[state+1,:])-1 # Exploit learned values
        end

        next_state, reward, done, info = env.step(action) 
        
        next_max = maximum(Q[next_state+1, :])
        
        Q[state+1, action+1] = (1-alpha) * Q[state+1, action+1] + alpha * (reward + gamma * next_max)
        
        if reward == -10
            penalties += 1
        end
        
        state = next_state
        epochs += 1
        
    end
    
    if mod(i, 100)==0
        println("Episode = $i")
    end
end



# Testing the learned Q function on a new environment 

state = env.reset()
epochs, penalties, reward = 0, 0, 0
    
done = false

frames = []
while !done
    action = argmax(Q[state+1,:]) - 1
    state, reward, done, info = env.step(action)

    if reward == -10
        penalties += 1
    end

    push!(frames, Dict(
            "frame"=> env.render(mode="ansi"),
            "state"=> state,
            "action"=> action,
            "reward"=> reward
        )
    )
    epochs += 1
end

# Visualize the result 
for (k,f) in enumerate(frames)
    for _ = 1:9
        print("\033[F\033[K")
    end
    print(f["frame"])
    print("Time = $k\n")
    sleep(0.1)
end