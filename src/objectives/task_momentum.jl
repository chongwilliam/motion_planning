"""
    finite-difference task momentum objective
"""
struct TaskMomentumObjective <: Objective
    Q
    n
    h
    task_momentum  # function to calculate momentum matrix Λ(q)J(q)
    # l_terminal

    idx_angle
end

function task_momentum_objective(Q, n, task_momentum; h = 0.0, idx_angle = (1:0))
    TaskMomentumObjective(Q, n, h, task_momentum, idx_angle)
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

        # J += v' * obj.Q[t] * Λ * v
        J += (Λ*v)'*obj.Q[t]*(Λ*v)
        # J += v'*obj.Q[t]*(Λ*v)
    end

    return J
end

function objective_gradient!(∇J, Z, obj::TaskMomentumObjective, idx, T)

    # tmp(y) = objective(y, obj, idx, T)
    # ∇J .+= ForwardDiff.gradient(tmp, Z)
    ∇J .+= ForwardDiff.gradient(x->objective(x, obj, idx, T), Z)

    # for t = 1:T-1
    #     x = view(Z, idx.x[t])
    #     u = view(Z, idx.u[t])
    #
    #     lx(y) = obj.l_stage(y, u, t)
    #     lu(y) = obj.l_stage(x, y, t)
    #
    #     ∇J[idx.x[t]] += ForwardDiff.gradient(lx, x)
    #     ∇J[idx.u[t]] += ForwardDiff.gradient(lu, u)
    #
    # end
    #
    # x = view(Z, idx.x[T])
    # ∇J[idx.x[T]] += ForwardDiff.gradient(obj.l_terminal, x)

    return nothing
end

# function objective_gradient!(∇J, Z, obj::TaskVelocityObjective, idx, T)
#     # tmp(y) = objective(y, obj, idx, T)
#     # ∇J .+= ForwardDiff.gradient(tmp, Z)
#     n = obj.n
#
#     J = 0.0
#     ϵ = 1.0e-5
#     for t = 1:T-1
#         q⁻ = view(Z, idx.x[t][n .+ (1:n)])
#         q⁺ = view(Z, idx.x[t + 1][n .+ (1:n)])
#         h = obj.h == 0.0 ? 1.0 : obj.h
#
#         v = (q⁺ - q⁻) ./ h
#         v[obj.idx_angle] = angle_diff.(view(q⁻, obj.idx_angle), view(q⁺, obj.idx_angle)) ./ h
#
#         dJdv = 2.0 * obj.Q[t] * v
#         ∇J[idx.x[t][n .+ (1:n)]] += -1.0 ./ h * dJdv
#         ∇J[idx.x[t + 1][n .+ (1:n)]] += 1.0 ./ h * dJdv
#     end
#
#     return nothing
# end

# function objective(Z, obj::VelocityObjective, idx, T)
#     n = obj.n
#
#     J = 0.0
#
#     for t = 1:T-1
#         q⁻ = view(Z, idx.x[t][n .+ (1:n)])
#         q⁺ = view(Z, idx.x[t + 1][n .+ (1:n)])
#         h = obj.h == 0.0 ? view(Z, idx.u[t][end]) : obj.h
#         v = (q⁺ - q⁻) ./ h
#         v[obj.idx_angle] = angle_diff.(view(q⁻, obj.idx_angle), view(q⁺, obj.idx_angle)) ./ h
#         J += v' * obj.Q[t] * v
#     end
#
#     return J
# end
#
# function objective_gradient!(∇J, Z, obj::VelocityObjective, idx, T)
#     # _tmp(y) = objective(y, obj, idx, T)
#     # ∇J .= ForwardDiff.gradient(_tmp, Z)
#     n = obj.n
#
#     for t = 1:T-1
#         q⁻ = view(Z, idx.x[t][n .+ (1:n)])
#         q⁺ = view(Z, idx.x[t + 1][n .+ (1:n)])
#         h = obj.h == 0.0 ? view(Z, idx.u[t][end]) : obj.h
#
#         # v = (q⁺ - q⁻) ./ h
#         # # v[obj.idx_angle] = angle_diff.(view(q⁻, obj.idx_angle),
#         # #     view(q⁺, obj.idx_angle)) ./ h
#         #
#         # dJdv = 2.0 * obj.Q[t] * v
#
#         function tmp(x⁻, x⁺, g)
#             w = (x⁺ - x⁻) ./ g
#             w[obj.idx_angle] = angle_diff.(view(x⁻, obj.idx_angle), view(x⁺, obj.idx_angle)) ./ g
#             w' * obj.Q[t] * w
#         end
#
#         tmp1(y) = tmp(y, q⁺, h)
#         tmp2(y) = tmp(q⁻, y, h)
#         tmp3(y) = tmp(q⁻, q⁺, y)
#
#         ∇J[idx.x[t][n .+ (1:n)]] += ForwardDiff.gradient(tmp1, q⁻)
#         ∇J[idx.x[t + 1][n .+ (1:n)]] += ForwardDiff.gradient(tmp2, q⁺)
#         if obj.h == 0.0
#             ∇J[idx.u[t][end]] += ForwardDiff.gradient(tmp3, h)[1]
#         end
#     end
#
#     return nothing
# end

# using ForwardDiff
# b1 = -pi / 2.0
# b2 = pi / 2.0 + pi
#
# angle_diff(b1, b2)
#
# tmp1(y) = angle_diff(y, b2)
# tmp2(y) = angle_diff(b1, y)
# for i = 1:1000
#     b1 = rand(1)[1]
#     b2 = rand(1)[1]
#     h1 = rand(1)[1]
#     tmp1(y) = angle_diff(y, b2) / h1
#     tmp2(y) = angle_diff(b1, y) / h1
#     @assert ForwardDiff.derivative(tmp1, b1) == -1.0 / h1
#     @assert ForwardDiff.derivative(tmp2, b2) == 1.0 / h1
# end
#
# angle_diff.(ones(2), pi * ones(2))
