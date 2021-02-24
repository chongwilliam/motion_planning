"""
    finite-difference task gravity objective (acceleration, posture control)
"""
struct TaskGravityObjective <: Objective
    Q
    n
    h
    task_gravity  # function to calculate task gravity (N, N-m)
    target
    # l_terminal

    idx_angle
end

function task_gravity_objective(Q, n, task_gravity, target; h = 0.0, idx_angle = (1:0))
    TaskGravityObjective(Q, n, h, task_gravity, target, idx_angle)
end

function angle_diff(a1, a2)
    mod(a2 - a1 + pi, 2 * pi) - pi
end

function objective(Z, obj::TaskGravityObjective, idx, T)
    n = obj.n

    J = 0.0
    ϵ = 1.0e-5
    for t = 1:T-1
        q⁻ = view(Z, idx.x[t][n .+ (1:n)])
        q⁺ = view(Z, idx.x[t + 1][n .+ (1:n)])
        h = obj.h == 0.0 ? 1.0 : obj.h
        v = (q⁺ - q⁻) ./ h
        v[obj.idx_angle] = angle_diff.(view(q⁻, obj.idx_angle), view(q⁺, obj.idx_angle)) ./ h
        p = obj.task_gravity(q⁻)
        J += (p - obj.target[t][:])' * obj.Q[t] * (p - obj.target[t][:])
    end

    return J
end

function objective_gradient!(∇J, Z, obj::TaskGravityObjective, idx, T)
    ∇J .+= ForwardDiff.gradient(x->objective(x, obj, idx, T), Z)
    return nothing
end
