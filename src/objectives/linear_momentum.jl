"""
    finite-difference linear momentum objective
"""
struct LinearMomentumObjective <: Objective
    Q
    n
    h
    task_momentum  # function to calculate momentum matrix Λ(q)J(q)
    target_momentum
    R
    # l_terminal

    idx_angle
end

function linear_momentum_objective(Q, n, task_momentum, target_momentum, R; h = 0.0, idx_angle = (1:0))
    LinearMomentumObjective(Q, n, h, task_momentum, target_momentum, R, idx_angle)
end

function angle_diff(a1, a2)
    mod(a2 - a1 + pi, 2 * pi) - pi
end

function objective(Z, obj::LinearMomentumObjective, idx, T)
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

        # J += v' * obj.Q[t] * Λ * v
        J += (obj.Q[t]*Λ*v - obj.target_momentum[t])' * obj.R[t] * (obj.Q[t]*Λ*v - obj.target_momentum[t])
        # J += v' * obj.Q[t] *  Λ * v
    end

    return J
end

function objective_gradient!(∇J, Z, obj::LinearMomentumObjective, idx, T)
    ∇J .+= ForwardDiff.gradient(x->objective(x, obj, idx, T), Z)
    return nothing
end
