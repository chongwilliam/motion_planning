"""
    finite-difference task momentum objective
"""
struct TaskMomentumObjective <: Objective
    Q
    n
    h
    task_momentum  # function to calculate momentum matrix Λ(q)J(q)
    target
    # l_terminal  # terminal cost 

    idx_angle
end

function task_momentum_objective(Q, n, task_momentum, target; h = 0.0, idx_angle = (1:0))
    TaskMomentumObjective(Q, n, h, task_momentum, target, idx_angle)
end

function angle_diff(a1, a2)
    mod(a2 - a1 + pi, 2 * pi) - pi
end

function objective(Z, obj::TaskMomentumObjective, idx, T)
    n = obj.n

    J = 0.0
    ϵ = 1.0e-5
    for t = 1:T-1
        q⁻ = view(Z, idx.x[t][n .+ (1:n)])
        q⁺ = view(Z, idx.x[t + 1][n .+ (1:n)])
        h = obj.h == 0.0 ? 1.0 : obj.h
        v = (q⁺ - q⁻) ./ h
        v[obj.idx_angle] = angle_diff.(view(q⁻, obj.idx_angle), view(q⁺, obj.idx_angle)) ./ h
        Λ = obj.task_momentum(q⁻)
        J += (Λ*v - obj.target[t][:])' * obj.Q[t] * (Λ*v - obj.target[t][:])
    end

    return J
end

function objective_gradient!(∇J, Z, obj::TaskMomentumObjective, idx, T)
    ∇J .+= ForwardDiff.gradient(x->objective(x, obj, idx, T), Z)
    return nothing
end
