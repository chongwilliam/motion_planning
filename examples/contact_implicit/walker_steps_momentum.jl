# Model
include_model("walker")

# Visualize
# - Pkg.add any external deps from visualize.jl
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)

function get_q⁺(x)
	view(x, model.nq .+ (1:model.nq))
end

# Horizon
T_w = 26  # walking horizon
T_s = 0  # stopping horizon
T = T_w + T_s

# Time step
tf = 1.5
h = tf / (T - 1)

# Configurations
# 1: x pos
# 2: z pos
# 3: torso angle (rel. to downward vertical)
# 4: thigh 1 angle (rel. to downward vertical)
# 5: calf 1 (rel. to downward vertical)
# 6: thigh 2 (rel. to downward vertical)
# 7: calf 2 (rel. to downward vertical)
# 8: foot 1 (rel. to downward vertical)
# 9: foot 2 (rel. to downward vertical)

q1 = zeros(model.nq)
# q1[3] = - pi / 32.0  # initial upper torso offset
q1[8] = pi / 2.0
q1[9] = pi / 2.0
q1[2] = model.l_thigh1 + model.l_calf1

qT = copy(q1)
qT[1] = Inf
# q_ref = linear_interpolation(q1, qT, T)
q_ref = linear_interpolation(q1, q1, T)
visualize!(vis, model, q_ref, Δt = h)

# Control
# u = (τ1..7, λ1..4, β1..8, ψ1..4, η1...8, s1)
# τ1: torso angle
# τ2: thigh 1 angle
# τ3: calf 1
# τ4: thigh 2
# τ5: calf 2
# τ6: foot 1
# τ7: foot 2

# ul <= u <= uu
u1 = initial_torque(model, q1, h)[model.idx_u]  # gravity compensation for current q
_uu = Inf * ones(model.m)
# _uu[model.idx_u] = [100, 100, 50, 100, 50, 40, 40]
_uu[model.idx_u] .= 100
_ul = zeros(model.m)
# _ul[model.idx_u] = [-100, -100, -50, -100, -50, -40, -40]
_ul[model.idx_u] .= -100
ul, uu = control_bounds(model, T, _ul, _uu)

# qL = [-Inf; -Inf; q1[3:end] .- pi / 2.0; -Inf; -Inf; q1[3:end] .- pi / 2.0]
# qU = [Inf; Inf; q1[3:end] .+ pi / 2.0; Inf; Inf; q1[3:end] .+ pi / 2.0]

qL = [-Inf; -Inf; q1[3] - pi/16.0; q1[4:end] .- pi / 2.0; -Inf; -Inf; q1[3] - pi/16.0; q1[4:end] .- pi / 2.0]
qU = [Inf; Inf; q1[3] + pi/16.0; q1[4:end] .+ pi / 2.0; Inf; Inf; q1[3] + pi/16.0; q1[4:end] .+ pi / 2.0]

xl, xu = state_bounds(model, T,
    qL, qU,
    x1 = [q1; q1],
	xT = [Inf * ones(model.nq); qT[1]; Inf * ones(model.nq - 1)])

# xl, xu = state_bounds(model, T,
#     qL, qU,
#     x1 = [q1; q1], # initial state
#     xT = [qT; qT]) # goal state

# xl, xu = state_bounds(model, T, qL, qU, x1 = [q1; q1])

# Configurations
# 1: x pos
# 2: z pos
# 3: torso angle (rel. to downward vertical)
# 4: thigh 1 angle (rel. to downward vertical)
# 5: calf 1 (rel. to downward vertical)
# 6: thigh 2 (rel. to downward vertical)
# 7: calf 2 (rel. to downward vertical)
# 8: foot 1 (rel. to downward vertical)
# 9: foot 2 (rel. to downward vertical)

function joint_limits!(c, x, u)
	# [q_prev; q_curr] = [7; 7] (ineq > 0); torso and thigh limits wrt global downards vertical
	qL = [0; -pi/2; 0; -pi/2]  # (thigh 1 - calf 1, calf 1 - foot 1, thigh 2 - calf 2, calf 2 - foot 2)
	qU = [pi; -pi/4; pi; -pi/4]
	# qL = [0; 0; 0; 0]  # (thigh 1 - calf 1, calf 1 - foot 1, thigh 2 - calf 2, calf 2 - foot 2)
	# qU = [pi; pi/2; pi; pi/2]
	q1 = x[9+4:9+7]  # (thigh 1, calf 1, thigh 2, calf 2)
	q2 = [x[9+5]; x[9+8]; x[9+7]; x[9+9]]  # (calf 1, foot 1, calf 2, foot 2)
	c[1:4] = q1 - q2 - qL
	c[5:end] = qU - (q1 - q2)
	nothing
end

# Objective
include_objective(["task_momentum", "task_momentum_smoothing", "task_gravity"])

x0 = configuration_to_state(q_ref)

# penalty on slack variable
obj_penalty = PenaltyObjective(1e5, model.m)

### Experimental ###
function get_com_momentum(q)
	J = jacobian_4(model, q)
	M = M_func(model, q)
	Λ = inv(J*inv(M)*J')
	return Λ*J
end

function get_p_torso(q)
	Jv = jacobian_1(model, q, body = :torso, mode = :com)
	Jw = [0., 0, 1, 0, 0, 0, 0, 0, 0]'
	J = [Jv; Jw]
	M = M_func(model, q)
	Λ = inv(J*inv(M)*J')
	J_bar = inv(M)*J'*Λ
	grav = G(model, q)
	return -(J_bar'*grav)  # forward direction
end

function get_torso_momentum(q)
	Jv = jacobian_1(model, q, body = :torso, mode = :com)
	Jw = [0., 0, 1, 0, 0, 0, 0, 0, 0]'
	J = [Jv; Jw]
	M = M_func(model, q)
	Λ = inv(J*inv(M)*J')
	return Λ*J
end

function get_heel1_momentum(q)
	Jv = jacobian_3(model, q, body = :foot_1, mode = :heel)
	Jw = [0., 0, 0, 0, 0, 0, 0, 1, 0]'
	J = [Jv; Jw]
	M = M_func(model, q)
	Λ = inv(J*inv(M)*J')
	return Λ*J
end

function get_heel2_momentum(q)
	Jv = jacobian_3(model, q, body = :foot_2, mode = :heel)
	Jw = [0., 0, 0, 0, 0, 0, 0, 0, 1]'
	J = [Jv; Jw]
	M = M_func(model, q)
	Λ = inv(J*inv(M)*J')
	return Λ*J
end

function get_toe1_momentum(q)
	Jv = jacobian_3(model, q, body = :foot_1, mode = :toe)
	Jw = [0., 0, 0, 0, 0, 0, 0, 1, 0]'
	J = [Jv; Jw]
	M = M_func(model, q)
	Λ = inv(J*inv(M)*J')
	return Λ*J
end

function get_toe2_momentum(q)
	Jv = jacobian_3(model, q, body = :foot_2, mode = :toe)
	Jw = [0., 0, 0, 0, 0, 0, 0, 0, 1]'
	J = [Jv; Jw]
	M = M_func(model, q)
	Λ = inv(J*inv(M)*J')
	return Λ*J
end

function get_heel1_effective_mass(q)
	Jv = jacobian_2(model, q, body = :calf_1, mode = :ee)
	Jw = [0., 0, 0, 0, 1, 0, 0, 0, 0]'
	J = [Jv; Jw]
	M = M_func(model, q)
	Λ_inv = J*inv(M)*J'
	return Λ_inv*J
end

function get_heel2_effective_mass(q)
	Jv = jacobian_2(model, q, body = :calf_2, mode = :ee)
	Jw = [0., 0, 0, 0, 0, 0, 1, 0, 0]'
	J = [Jv; Jw]
	M = M_func(model, q)
	Λ_inv = J*inv(M)*J'
	return Λ_inv*J
end

""" Cost tuning """
com_linear_gain = 0*1e0
com_angular_gain = 0*1e0
com_linear_smooth_gain = 1e0
com_angular_smooth_gain = 1e0
###
heel_linear_smoothing_gain = 1e0
heel_angular_smoothing_gain = 1e0
heel_mass_gain = 1e0
###
torso_p_gain = 1e-2
### Primary gains
heel_linear_gain = 1e1
heel_angular_gain = 1e1
toe_linear_gain = 0*1e-1
toe_angular_gain = 0*1e-1
###
posture_gain = 1e0
control_gain = 1e-5

""" Heel 1 target momentum """
q_v = [heel_linear_gain, heel_linear_gain, heel_angular_gain]
target = Vector{Vector{Float64}}(undef, T-1)  # linear y, z, rotational x
cost = []
for t = 1:T-1
	if t <= T/4
		target[t] = [20., 0 - (40/(T/4))*t, 1 * (0 - (10/(T/4))*t)]  # standing start trajectory
		# target[t] = [5., 0 - (30/(T/4))*t, 1 * (0 - (5/(T/4))*t)]  # standing start trajectory
		push!(cost, Diagonal(q_v))
	elseif t <= T/2
		target[t] = [0., 0, 0]
		push!(cost, 0 * Diagonal(q_v))
	elseif t <= 3*T/4
		target[t] = [20., 40 - (80/(T/4))*(t-T/2), 1 * (10 - (20/(T/4))*(t-T/2))]  # swing start trajectory
		push!(cost, Diagonal(q_v))
	else
		target[t] = [0., 0, 0]
		push!(cost, 0 * Diagonal(q_v))
	end
end

heel1_linear_momentum = task_momentum_objective(
    cost,
    model.nq,
    h = h,
	get_heel1_momentum,
	target,
    idx_angle = collect([3, 4, 5, 6, 7, 8 ,9])
	)

""" Heel 2 target momentum """
q_v = [heel_linear_gain, heel_linear_gain, heel_angular_gain]
target = Vector{Vector{Float64}}(undef, T-1)
cost = []
for t = 1:T-1
	if t <= T/4
		target[t] = [0., 0, 0]
		push!(cost, 0 * Diagonal(q_v))
	elseif t <= T/2
		# target[t] = [20., 35 - (70/(T/2))*(t-T/2), 10 - (20/(T/2))*(t-T/2)]
		target[t] = [20., 40 - (80/(T/4))*(t-T/4), 1 * (10 - (20/(T/4))*(t-T/4))]
		push!(cost, Diagonal(q_v))
	elseif t <= 3*T/4
		target[t] = [0., 0, 0]
		push!(cost, 0 * Diagonal(q_v))
	else
		# target[t] = [0., 0, 0]
		# push!(cost, 1 * Diagonal(q_v))
		target[t] = [20., 40 - (80/(T/4))*(t-3*T/4), 1 * (10 - (20/(T/4))*(t-3*T/4))]
		push!(cost, Diagonal(q_v))
	end
end

heel2_linear_momentum = task_momentum_objective(
    cost,
    model.nq,
    h = h,
	get_heel2_momentum,
	target,
    idx_angle = collect([3, 4, 5, 6, 7, 8 ,9])
	)

""" Toe 1 target momentum """
q_v = [toe_linear_gain, toe_linear_gain, toe_angular_gain]
target = Vector{Vector{Float64}}(undef, T-1)  # linear y, z, rotational x
cost = []
for t = 1:T-1
	if t <= T_w/2
		target[t] = [0., 0, 0]
		push!(cost, 0 * Diagonal(q_v))
	elseif t <= T_w
		target[t] = [0., 0, 0]
		push!(cost, Diagonal(q_v))
	else
		target[t] = [0., 0, 0]
		push!(cost, 0 * Diagonal(q_v))
	end
end

toe1_linear_momentum = task_momentum_objective(
    cost,
    model.nq,
    h = h,
	get_toe1_momentum,
	target,
    idx_angle = collect([3, 4, 5, 6, 7, 8 ,9])
	)

""" Toe 2 target momentum """
q_v = [toe_linear_gain, toe_linear_gain, toe_angular_gain]
target = Vector{Vector{Float64}}(undef, T-1)
cost = []
for t = 1:T-1
	if t <= T_w/2
		target[t] = [0., 0, 0]
		push!(cost, Diagonal(q_v))
	elseif t <= T_w
		target[t] = [0., 0, 0]
		push!(cost, 0 * Diagonal(q_v))
	else
		target[t] = [0., 0, 0]
		push!(cost, 0 * Diagonal(q_v))
	end
end

toe2_linear_momentum = task_momentum_objective(
    cost,
    model.nq,
    h = h,
	get_toe2_momentum,
	target,
    idx_angle = collect([3, 4, 5, 6, 7, 8 ,9])
	)

""" Standing end constraint """
q_v = [com_linear_gain, com_linear_gain, com_angular_gain]
target_momentum = [[0., 0, 0] for i = 1:T-1]
cost = []
for t = 1:T-1
	if t <= T_w/2
		push!(cost, 0 *Diagonal(q_v))
	elseif t <= T_w
		push!(cost, 0 * Diagonal(q_v))
	else
		push!(cost, Diagonal(q_v))
	end
end
terminal_momentum = task_momentum_objective(
    cost,
    model.nq,
    h = h,
	get_torso_momentum,
	target_momentum,
    idx_angle = collect([3, 4, 5, 6, 7, 8 ,9])
	)

""" COM momentum """
q_v = [com_linear_gain, com_linear_gain, com_angular_gain]
target_momentum = [[20., 0, 0] for i = 1:T-1]
com_momentum = task_momentum_objective(
    [Diagonal(q_v) for t = 1:T-1],
    model.nq,
    h = h,
	get_com_momentum,
	target_momentum,
    idx_angle = collect([3, 4, 5, 6, 7, 8 ,9])
	)

""" COM momentum smoothing """
q_v = [com_linear_smooth_gain, com_linear_smooth_gain, com_angular_smooth_gain]
com_smoothing = task_momentum_smoothing_objective(
    [Diagonal(q_v) for t = 1:T-1],
    model.nq,
    h = h,
	get_com_momentum,
    idx_angle = collect([3, 4, 5, 6, 7, 8, 9])
	)

""" Torso task space gravity """
target = Vector{Vector{Float64}}(undef, T-1)
peak = 10.
for t = 1:T-1
	if t <= T/4
		# push!(target, [(peak/(T/4))*t, 0, 0.])
		target[t] = [(peak/(T/4))*t, 0, 0]
	elseif t <= T/2
		# push!(target, [peak-(peak/(T/4))*(t-T/4), 0, 0])
		target[t] = [peak-(peak/(T/4))*(t-T/4), 0, 0]
	elseif t <= 3*T/4
		# push!(target, [(peak/(T/4))*(t-T/2), 0, 0])
		target[t] = [(peak/(T/4))*(t-T/2), 0, 0]
	else
		# push!(target, [peak-(peak/(T/4))*(t-3*T/4), 0, 0])
		target[t] = [peak-(peak/(T/4))*(t-3*T/4), 0, 0]
	end
end
target = [[peak, 0, 0] for t = 1:T-1]
q_v = torso_p_gain * [1., 0, 0]
torso_task_gravity = task_gravity_objective(
    [Diagonal(q_v) for t = 1:T-1],
    model.nq,
    h = h,
	get_p_torso,
	target,
    idx_angle = collect([3, 4, 5, 6, 7, 8 ,9])
	)

""" Heel 1 effective mass """
q_v = heel_mass_gain * [1., 1, 0]
cost = []
for t = 1:T-1
	if t <= T/2
		push!(cost, Diagonal(q_v))
	else
		push!(cost, zeros(3,3))
	end
end
target = [[0., 0, 0] for t = 1:T-1]
heel1_effective_mass = task_momentum_objective(
    [Diagonal(q_v) for t = 1:T-1],
    model.nq,
    h = h,
	get_heel1_effective_mass,
	target,
    idx_angle = collect([3, 4, 5, 6, 7, 8 ,9])
	)

""" Heel 2 effective mass """
q_v = heel_mass_gain*[1., 1, 0]
cost = []
for t = 1:T-1
	if t > T/2
		push!(cost, Diagonal(q_v))
	else
		push!(cost, zeros(3,3))
	end
end
target = [[0., 0, 0] for t = 1:T-1]
heel2_effective_mass = task_momentum_objective(
    [Diagonal(q_v) for t = 1:T-1],
    model.nq,
    h = h,
	get_heel2_effective_mass,
	target,
    idx_angle = collect([3, 4, 5, 6, 7, 8, 9])
	)

""" Heel 1 L2 smoothing """
q_v = [heel_linear_smoothing_gain, heel_linear_smoothing_gain, heel_angular_smoothing_gain]
heel1_smoothing = task_momentum_smoothing_objective(
    [Diagonal(q_v) for t = 1:T-1],
    model.nq,
    h = h,
	get_heel1_momentum,
    idx_angle = collect([3, 4, 5, 6, 7, 8 ,9])
	)

""" Heel 2 L2 smoothing """
q_v = [heel_linear_smoothing_gain, heel_linear_smoothing_gain, heel_angular_smoothing_gain]
heel2_smoothing = task_momentum_smoothing_objective(
    [Diagonal(q_v) for t = 1:T-1],
    model.nq,
    h = h,
	get_heel2_momentum,
    idx_angle = collect([3, 4, 5, 6, 7, 8, 9])
	)

""" Quadratic state-energy cost """
q_v = [0.; 0 * posture_gain * ones(model.n-1)]  # including height
posture_cost = []
for t = 1:T
	if t > 3*T/4
		push!(posture_cost, Diagonal(q_v))
	else
		push!(posture_cost, 0 * Diagonal(q_v))
	end
end
obj_control = quadratic_tracking_objective(
    posture_cost,
    [Diagonal([control_gain * ones(model.nu)..., 0.0 * ones(model.m - model.nu)...]) for t = 1:T-1],
    [x0[end] for t = 1:T],
    [[0 * u1; zeros(model.m - model.nu)] for t = 1:T-1])

obj = MultiObjective([obj_penalty,
                      # obj_control,
					  # com_momentum,
					  # torso_task_gravity,
					  # heel1_smoothing,
					  # heel2_smoothing,
					  # com_angular_smoothing,
					  heel1_linear_momentum,
					  heel2_linear_momentum])
					  # terminal_momentum])
					  # toe1_linear_momentum,
					  # toe2_linear_momentum])
					  # heel1_effective_mass,
					  # heel2_effective_mass])

# Constraints
include_constraints(["contact", "loop", "contact_no_slip", "free_time", "stage"])
# con_loop = loop_constraints(model, collect([(2:7)...,(9:14)...]), 1, T)
con_contact = contact_constraints(model, T)
# con_free_time = free_time_constraints(T)

n_stage = 8
t_idx = [t for t = 1:T-1]
con_limits = stage_constraints(joint_limits!, n_stage, (1:n_stage), t_idx)

con = multiple_constraints([con_contact, con_limits])#, con_free_time])#, con_loop])

# Problem
prob = trajectory_optimization_problem(model,
               obj,
               T,
               h = h,
               xl = xl,
               xu = xu,
               ul = ul,
               uu = uu,
               con = con)

# τ1: torso angle
# τ2: thigh 1 angle
# τ3: calf 1
# τ4: thigh 2
# τ5: calf 2
# τ6: foot 1
# τ7: foot 2

# trajectory initialization
up = [_uu[model.idx_u][1], _ul[model.idx_u][2], _uu[model.idx_u][3], _uu[model.idx_u][4], _ul[model.idx_u][5], 0, 0]
u0 = [[0 * u1 + 0 * 1e-2 * up + 1e0 * randn(size(u1)); 1e-1 * randn(model.m - model.nu)] for t = 1:T-1] # random controls

# Pack trajectories into vector
# z0 = pack(x0, u0, prob) + 0.005 * rand(prob.num_var)
z0 = pack(x0, u0, prob)

# Solve
include_snopt()
@time z̄, info = solve(prob, copy(z0),
    nlp = :SNOPT7,
    tol = 1.0e-3, c_tol = 1.0e-3, mapl = 5,
    time_limit = 5 * 60, max_iter = 1000)
# @time z̄, info = solve(prob, copy(z0),
#     nlp = :ipopt,
#     tol = 1.0e-3, c_tol = 1.0e-3, mapl = 5,
#     time_limit = 5 * 60, max_iter = 1000)
@show check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)
visualize!(vis, model, state_to_configuration(x̄), Δt = h)

using JLD2
# @save "iteration0.jld2" z̄
@save "iteration1.jld2" z̄
# @save "iteration2.jld2" z̄

# @load "iteration0.jld2" z̄

# Warm-start
x0, u0 = x̄, ū
# z0 = pack(x̄, ū, prob)

# if false
#     include_snopt()
# 	@time z̄ , info = solve(prob, copy(z0),
# 		nlp = :SNOPT7,
# 		tol = 1.0e-3, c_tol = 1.0e-3, mapl = 5,
# 		time_limit = 60 * 3)
# 	@show check_slack(z̄, prob)
# 	x̄, ū = unpack(z̄, prob)
#     tfc, tc, h̄ = get_time(ū)

# 	#projection
# 	Q = [Diagonal(ones(model.n)) for t = 1:T]
# 	R = [Diagonal(0.1 * ones(model.m)) for t = 1:T-1]
# 	x_proj, u_proj = lqr_projection(model, x̄, ū, h̄[1], Q, R)
#
# 	@show tfc
# 	@show h̄[1]
# 	@save joinpath(pwd(), "examples/trajectories/walker_steps.jld2") x̄ ū h̄ x_proj u_proj
# else
# 	@load joinpath(pwd(), "examples/trajectories/walker_steps.jld2") x̄ ū h̄ x_proj u_proj
# end

# # Visualize
# vis = Visualizer()
# render(vis)
# visualize!(vis, model, state_to_configuration(x̄), Δt = h)
#
# using Plots
# fh1 = [kinematics_2(model,
#     state_to_configuration(x̄)[t], body = :calf_1, mode = :ee)[2] for t = 1:T]
# fh2 = [kinematics_2(model,
#     state_to_configuration(x̄)[t], body = :calf_2, mode = :ee)[2] for t = 1:T]
# plot(fh1, linetype = :steppost, label = "foot 1")
# plot!(fh2, linetype = :steppost, label = "foot 2")
#
# plot(hcat(ū...)[1:7, :]',
#     linetype = :steppost,
#     label = "",
#     color = :red,
#     width = 2.0)
# #
# # plot!(hcat(u_proj...)[1:4, :]',
# #     linetype = :steppost,
# #     label = "",
# #     color = :black)
# #
# # plot(hcat(u_proj...)[5:6, :]',
# #     linetype = :steppost,
# #     label = "",
# #     width = 2.0)
#
# plot(hcat(state_to_configuration(x̄)...)'[1:3],
#     color = :red,
#     width = 2.0,
#     label = "")
# #
# # plot!(hcat(state_to_configuration(x_proj)...)',
# #     color = :black,
# #     label = "")
