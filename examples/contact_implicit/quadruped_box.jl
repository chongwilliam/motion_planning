# Model
include_model("quadruped")

function ϕ_func(model::Quadruped, q)
	h = 0.25
	w = 0.5
	x = 1.25

	p_leg_1 = kinematics_2(model, q, body = :leg_1, mode = :ee)
	p_leg_2 = kinematics_2(model, q, body = :leg_2, mode = :ee)
	p_leg_3 = kinematics_3(model, q, body = :leg_3, mode = :ee)
	p_leg_4 = kinematics_3(model, q, body = :leg_4, mode = :ee)

	z1 = p_leg_1[2] - (((p_leg_1[1] > (x - w / 2.0)) && (p_leg_1[1] < (x + w / 2.0))) ? h : 0.0)
	z2 = p_leg_2[2] - (((p_leg_2[1] > (x - w / 2.0)) && (p_leg_2[1] < (x + w / 2.0))) ? h : 0.0)
	z3 = p_leg_3[2] - (((p_leg_3[1] > (x - w / 2.0)) && (p_leg_3[1] < (x + w / 2.0))) ? h : 0.0)
	z4 = p_leg_4[2] - (((p_leg_4[1] > (x - w / 2.0)) && (p_leg_4[1] < (x + w / 2.0))) ? h : 0.0)

	@SVector [z1, z2, z3, z4]
end

include(joinpath(pwd(), "src/objectives/velocity.jl"))
include(joinpath(pwd(), "src/objectives/nonlinear_stage.jl"))
include(joinpath(pwd(), "src/constraints/contact.jl"))

# Visualize
# - Pkg.add any external deps from visualize.jl
include(joinpath(pwd(), "models/visualize.jl"))
vis = Visualizer()
render(vis)

# Configurations
# 1: x pos
# 2: z pos
# 3: torso angle (rel. to downward vertical)
# 4: thigh 1 angle (rel. to downward vertical)
# 5: calf 1 (rel. to thigh 1)
# 6: thigh 2 (rel. to downward vertical)
# 7: calf 2 (rel. to thigh 2)
# θ = pi / 12.5
# q1 = initial_configuration(model, θ) # generate initial config from θ
# qT = copy(q1)
# qT[1] += 1.0
# q1[3] -= pi / 30.0
# q1[4] += pi / 20.0
# q1[5] -= pi / 10.0
# q1, qT = loop_configurations(model, θ)
# qT[1] += 1.0

θ = -pi / 5.0
q1 = initial_configuration(model, θ)
qT = copy(q1)
qT[1] = 2.5
visualize!(vis, model, [qT])

# Horizon
T = 26

# Time step
tf = 2.5
h = tf / (T - 1)

# Bounds

# control
# u = (τ1..4, λ1..2, β1..4, ψ1..2, η1...4, s1)
# ul <= u <= uu
_uu = Inf * ones(model.m)
_uu[model.idx_u] = model.uU

_ul = zeros(model.m)
_ul[model.idx_u] = model.uL
ul, uu = control_bounds(model, T, _ul, _uu)

u1 = initial_torque(model, q1, h)[model.idx_u]

xl, xu = state_bounds(model, T,
    x1 = [q1; q1],
    xT = [qT; qT])

# Objective
q_ref = linear_interp(q1, qT, T)
visualize!(vis, model, q_ref)

x0 = configuration_to_state(q_ref)

# penalty on slack variable
obj_penalty = PenaltyObjective(1.0e5, model.m)

# quadratic tracking objective
# Σ (x - xref)' Q (x - x_ref) + (u - u_ref)' R (u - u_ref)
obj_control = quadratic_tracking_objective(
    [zeros(model.n, model.n) for t = 1:T],
    [Diagonal([1.0e-1 * ones(model.nu)..., zeros(model.m - model.nu)...]) for t = 1:T-1],
    [zeros(model.n) for t = 1:T],
    [[copy(u1); zeros(model.m - model.nu)] for t = 1:T]
    )

# quadratic velocity penalty
# Σ v' Q v
obj_velocity = velocity_objective(
    [Diagonal(10.0 * ones(model.nq)) for t = 1:T-1],
    model.nq,
    h = h,
    idx_angle = collect([3, 4, 5, 6, 7, 8, 9, 10, 11]))

# torso height
q2_idx = (12:22)
t_h = kinematics_1(model, q1, body = :torso, mode = :com)[2]
l_stage_torso_h(x, u, t) = 100.0 * (kinematics_1(model, view(x, q2_idx), body = :torso, mode = :com)[2] - t_h)^2.0
l_terminal_torso_h(x) = 0.0 * (kinematics_1(model, view(x, q2_idx), body = :torso, mode = :com)[2] - t_h)^2.0
obj_torso_h = nonlinear_stage_objective(l_stage_torso_h, l_terminal_torso_h)

# torso lateral
l_stage_torso_lat(x, u, t) = (1.0 * (kinematics_1(model, view(x, q2_idx), body = :torso, mode = :com)[1] - kinematics_1(model, view(x0[t], q2_idx), body = :torso, mode = :com)[1])^2.0)
l_terminal_torso_lat(x) = (0.0 * (kinematics_1(model, view(x, q2_idx), body = :torso, mode = :com)[1] - kinematics_1(model, view(x0[T], q2_idx), body = :torso, mode = :com)[1])^2.0)
obj_torso_lat = nonlinear_stage_objective(l_stage_torso_lat, l_terminal_torso_lat)

# foot 1 height
l_stage_fh1(x, u, t) = 10.0 * (kinematics_2(model, view(x, q2_idx), body = :leg_1, mode = :ee)[2] - 0.1)^2.0
l_terminal_fh1(x) = 0.0 * (kinematics_2(model, view(x, q2_idx), body = :leg_1, mode = :ee)[2])^2.0
obj_fh1 = nonlinear_stage_objective(l_stage_fh1, l_terminal_fh1)

# foot 2 height
l_stage_fh2(x, u, t) = 10.0 * (kinematics_2(model, view(x, q2_idx), body = :leg_2, mode = :ee)[2] - 0.1)^2.0
l_terminal_fh2(x) = 0.0 * (kinematics_2(model, view(x, q2_idx), body = :leg_2, mode = :ee)[2])^2.0
obj_fh2 = nonlinear_stage_objective(l_stage_fh2, l_terminal_fh2)

# foot 3 height
l_stage_fh3(x, u, t) = 10.0 * (kinematics_3(model, view(x, q2_idx), body = :leg_3, mode = :ee)[2] - 0.1)^2.0
l_terminal_fh3(x) = 0.0 * (kinematics_3(model, view(x, q2_idx), body = :leg_3, mode = :ee)[2])^2.0
obj_fh3 = nonlinear_stage_objective(l_stage_fh3, l_terminal_fh3)

# foot 4 height
l_stage_fh4(x, u, t) = 10.0 * (kinematics_3(model, view(x, q2_idx), body = :leg_4, mode = :ee)[2] - 0.1)^2.0
l_terminal_fh4(x) = 0.0 * (kinematics_3(model, view(x, q2_idx), body = :leg_4, mode = :ee)[2])^2.0
obj_fh4 = nonlinear_stage_objective(l_stage_fh4, l_terminal_fh4)

obj = MultiObjective([obj_penalty,
                      obj_control,
                      obj_velocity,
                      obj_torso_h,
                      obj_torso_lat,
                      obj_fh1,
                      obj_fh2,
                      obj_fh3,
                      obj_fh4])

# Constraints
con_contact = contact_constraints(model, T)
con = multiple_constraints([con_contact])

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

# trajectory initialization
u0 = [[copy(u1); 1.0e-3 * rand(model.m - model.nu)] for t = 1:T-1] # random controls

# Pack trajectories into vector
z0 = pack(x0, u0, prob)
z0 .+= 1.0e-3 * randn(prob.num_var)

# Solve
include_snopt()

@time z̄ = solve(prob, copy(z0),
    nlp = :SNOPT7,
    tol = 1.0e-3, c_tol = 1.0e-3,
    time_limit = 60 * 2, mapl = 5)

@time z̄ = solve(prob, copy(z̄ .+ 1.0e-3 * rand(prob.num_var)),
    nlp = :SNOPT7,
    tol = 1.0e-3, c_tol = 1.0e-3,
    time_limit = 60 * 30, mapl = 5)

check_slack(z̄, prob)
x̄, ū = unpack(z̄, prob)

# Visualize
open(vis)
visualize!(vis, model, state_to_configuration(x̄), Δt = h)
# setobject!(vis["box"], HyperRectangle(Vec(0.0, 0.0, 0.0), Vec(0.5, 1.0, 0.25)))
# settransform!(vis["box"], Translation(1.0, -0.5, 0))
# # open(vis)