#= Modified from Lossless convexification rocket landing problem definition.

Sequential convex programming algorithms for trajectory optimization.
Copyright (C) 2021 Autonomous Controls Laboratory (University of Washington)

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>. 

Last edited: 5/25/2023 by Julia Briden.=#


LangServer = isdefined(@__MODULE__, :LanguageServer)

if LangServer
    include("parameters.jl")
end

using JuMP
using ECOS
using Debugger
using PyCall

"""
    solve_pdf_fft(rocket, tf)

Solve the rocket landing problem for a fixed flight time `tf`.

# Arguments
- `rocket`: the rocket object.
- `tf`: the time of flight.

# Returns
- `sol`: solution to the rocket landing problem for this `tf`.
"""
function solve_pdg_fft(rocket::Rocket, tf::RealValue; environment = nothing, tight_constraints = nothing, initial_values = nothing)::Solution

    # >> Discretize [0, tf] interval <<

    # If tf does not divide into rocket.Δt intervals evenly, then reduce Δt by
    # minimum amount to get an integer number of intervals
    #N = Int(floor(tf/rocket.Δt))+1+Int(tf%rocket.Δt!=0) # Number of time nodes

    # Use N as the input to regulate size of constraint vectors
    N = rocket.N
    Δt = tf/(N-1)

    t = RealVector(0.0:Δt:(N-1)*Δt)

    A, B, p = c2d(rocket.A_c, rocket.B_c, rocket.p_c, Δt) #noerr

    constraints = rocket.constraints

    # >> Make the optimization problem <<
    #mdl = Model(with_optimizer(Clarabel.Optimizer))
    mdl = Model(optimizer_with_attributes(ECOS.Optimizer, "verbose"=>false))

    # (Scaled) variables
    #nolint: r_s, v_s, z_s, u_s, ξ_s
    

    @variable(mdl, r_s[1:3, 1:N])
    @variable(mdl, v_s[1:3, 1:N])
    @variable(mdl, z_s[1:N])
    @variable(mdl, u_s[1:3, 1:N-1])
    @variable(mdl, ξ_s[1:N-1])

    # Scaling (for better numerical behaviour)
    # Scaling matrices
    s_r = zeros(3)
    S_r = Diagonal([max(1.0, abs(rocket.r0[i])) for i=1:3])
    s_v = zeros(3)
    S_v = Diagonal([max(1.0, abs(rocket.v0[i])) for i=1:3])
    s_z = (log(rocket.m_dry)+log(rocket.m_wet))/2
    S_z = log(rocket.m_wet)-s_z
    s_u = RealVector([0,
                        0,
                        0.5*(rocket.ρ_min/rocket.m_wet*cos(rocket.γ_p)+
                            rocket.ρ_max/rocket.m_dry)])
    S_u = Diagonal([rocket.ρ_max/rocket.m_dry*sin(rocket.γ_p),
                        rocket.ρ_max/rocket.m_dry*sin(rocket.γ_p),
                        rocket.ρ_max/rocket.m_dry-s_u[3]])
    s_ξ, S_ξ = s_u[3], S_u[3,3]

    # If initial values are given
    if ~isnothing(initial_values)
        r_val = initial_values.r
        set_start_value.(r_s, inv(S_r)*(r_val-repeat(s_r,1,N)))
        v_val = initial_values.v
        set_start_value.(v_s, inv(S_v)*(v_val-repeat(s_v,1,N)))
        z_val = initial_values.z
        set_start_value.(z_s, inv(S_z)*(z_val-repeat([s_z],N)))
        u_val = initial_values.u
        set_start_value.(u_s, inv(S_u)*(u_val-repeat(s_u,1,N-1)))
        ξ_val = initial_values.ξ
        set_start_value.(ξ_s, inv(S_ξ)*(ξ_val-repeat([s_ξ],N-1)))
    end

    # Physical variables
    r = S_r*r_s+repeat(s_r,1,N)
    v = S_v*v_s+repeat(s_v,1,N)
    z = S_z*z_s+repeat([s_z],N)
    u = S_u*u_s+repeat(s_u,1,N-1)
    ξ = S_ξ*ξ_s+repeat([s_ξ],N-1)

    # Cost function
    #nolint: Min
    @objective(mdl, Min, Δt*sum(ξ))

    # Dynamics
    X = (k) -> [r[:,k];v[:,k];z[k]] # State at time index k
    U = (k) -> [u[:,k];ξ[k]] # Input at time index k
    @constraint(mdl, [k=1:N-1], X(k+1).==A*X(k)+B*U(k)+p)

    z0 = (k) -> log(rocket.m_wet-rocket.α*rocket.ρ_max*t[k])
    μ_min = (k) -> rocket.ρ_min*exp(-z0(k))
    μ_max = (k) -> rocket.ρ_max*exp(-z0(k))
    δz = (k) -> z[k]-z0(k)
    e_z = RealVector([0,0,1])


    if ~isnothing(tight_constraints)
        H_gs = RealMatrix([cos(rocket.γ_gs) 0 -sin(rocket.γ_gs);
                        -cos(rocket.γ_gs) 0 -sin(rocket.γ_gs);
                        0 cos(rocket.γ_gs) -sin(rocket.γ_gs);
                        0 -cos(rocket.γ_gs) -sin(rocket.γ_gs)])
        h_gs = zeros(4)

        for k=1:length(tight_constraints)
            # Velocity upper bound
            if (k <= N)
                if tight_constraints[k] == 1
                    i = k
                    @constraint(mdl, vcat(rocket.v_max,v[:,i]) in MOI.SecondOrderCone(4))
                end
            # Glideslope
            elseif (k > N) & (k <= 2*N)
                if tight_constraints[k] == 1
                    i = k-N
                    @constraint(mdl, H_gs*r[:,i].<=h_gs)
                end
            # Thrust bounds (approximate)
            elseif (k > 2*N) & (k <= 2*N+(N-1))
                if tight_constraints[k] == 1
                    i = k-(2*N)
                    @constraint(mdl, ξ[i]>=μ_min(i)*(1-δz(i)+0.5*δz(i)^2))
                end
            elseif (k > 2*N+(N-1)) & (k <= 2*N+2*(N-1))
                if tight_constraints[k] == 1
                    i = k-(2*N+(N-1))
                    @constraint(mdl, ξ[i]<=μ_max(i)*(1-δz(i)))
                end
            elseif (k > 2*N+2*(N-1)) & (k <= 2*N+3*(N-1))
                if tight_constraints[k] == 1
                    i = k-(2*N+2*(N-1))
                    @constraint(mdl, vcat(ξ[i],u[:,i]) in
                    MOI.SecondOrderCone(4))
                end
            # Mass bounds
            elseif (k > 2*N+3*(N-1)) & (k <= 3*N+3*(N-1))
                if tight_constraints[k] == 1
                    i = k-(2*N+3*(N-1))
                    @constraint(mdl, z0(i)<=z[i])
                end
            elseif (k > 3*N+3*(N-1)) & (k <= 4*N+3*(N-1))
                if tight_constraints[k] == 1
                    i = k-(3*N+3*(N-1))
                    @constraint(mdl, z[i]<=log(rocket.m_wet-rocket.α*rocket.ρ_min*t[i]))
                end
            # Attitude pointing
            elseif (k > 4*N+3*(N-1)) & (k <= 4*N+4*(N-1))
                if tight_constraints[k] == 1
                    i = k-(4*N+3*(N-1))
                    @constraint(mdl, dot(u[:,i],e_z)>=ξ[i]*cos(rocket.γ_p))
                end
            # Boundary Conditions
            elseif (k > 4*(N-1)+4*N) & (k <= 4*(N-1)+4*N + 1)
                if tight_constraints[k] == 1
                    @constraint(mdl, z[N]>=log(rocket.m_dry))
                end
            end
        end

    else

        # Thrust bounds (approximate)
        if constraints["Thrust Bound"]
            if ~isnothing(tight_constraints)
                for k=1:N-1
                    if tight_constraints["Thrust Bound"][1][k] == 1
                        @constraint(mdl, ξ[k]>=μ_min(k)*(1-δz(k)+0.5*δz(k)^2))
                    end
                    if tight_constraints["Thrust Bound"][2][k] == 1
                        @constraint(mdl, ξ[k]<=μ_max(k)*(1-δz(k)))
                    end
                    if tight_constraints["Thrust Bound"][3][k] == 1
                        @constraint(mdl, vcat(ξ[k],u[:,k]) in
                        MOI.SecondOrderCone(4))
                    end
                end
            else
                @constraint(mdl, [k=1:N-1], ξ[k]>=μ_min(k)*(1-δz(k)+0.5*δz(k)^2))
                @constraint(mdl, [k=1:N-1], ξ[k]<=μ_max(k)*(1-δz(k)))

                # Thrust bounds LCvx
                # Look into further defining this constraint for bang-bang
                @constraint(mdl, [k=1:N-1], vcat(ξ[k],u[:,k]) in
                MOI.SecondOrderCone(4))
            end
        end

        # Mass physical bounds constraint
        if constraints["Mass Bound"]
            if ~isnothing(tight_constraints)
                for k=1:N
                    if tight_constraints["Mass Bound"][1][k] == 1
                        @constraint(mdl, z0(k)<=z[k])
                    end
                    if tight_constraints["Mass Bound"][2][k] == 1
                        @constraint(mdl, z[k]<=log(rocket.m_wet-rocket.α*rocket.ρ_min*t[k]))
                    end
                end
            else
                @constraint(mdl, [k=1:N], z0(k)<=z[k])
                @constraint(mdl, [k=1:N], z[k]<=log(rocket.m_wet-rocket.α*rocket.ρ_min*t[k]))
            end
        end

        
        # Attitude pointing constraint
        e_z = RealVector([0,0,1])
        if constraints["Attitude Pointing"]
            if ~isnothing(tight_constraints)
                for k=1:N-1
                    if tight_constraints["Attitude Pointing"][1][k] == 1
                        @constraint(mdl, dot(u[:,k],e_z)>=ξ[k]*cos(rocket.γ_p))
                    end
                end
            else
                @constraint(mdl, [k=1:N-1], dot(u[:,k],e_z)>=ξ[k]*cos(rocket.γ_p))
            end
        end

        # Glide slope constraint
        if constraints["Glideslope"]
            H_gs = RealMatrix([cos(rocket.γ_gs) 0 -sin(rocket.γ_gs);
                            -cos(rocket.γ_gs) 0 -sin(rocket.γ_gs);
                            0 cos(rocket.γ_gs) -sin(rocket.γ_gs);
                            0 -cos(rocket.γ_gs) -sin(rocket.γ_gs)])
            h_gs = zeros(4)
            if ~isnothing(tight_constraints)
                for k=1:N
                    if tight_constraints["Glideslope"][1][k] == 1
                        @constraint(mdl, H_gs*r[:,k].<=h_gs)
                    end
                end
            else
                @constraint(mdl, [k=1:N], H_gs*r[:,k].<=h_gs)
            end
        end

        # Velocity upper bound
        if constraints["Velocity Upper Bound"]
            if ~isnothing(tight_constraints)
                for k=1:N
                    if tight_constraints["Velocity Upper Bound"][1][k] == 1
                        @constraint(mdl, vcat(rocket.v_max,v[:,k]) in MOI.SecondOrderCone(4))
                    end
                end
            else
                @constraint(mdl, [k=1:N], vcat(rocket.v_max,v[:,k]) in MOI.SecondOrderCone(4))
            end
        end

        if ~isnothing(tight_constraints)
            if tight_constraints["Boundary Conditions"][1] == 1
                    @constraint(mdl, z[N]>=log(rocket.m_dry))
            end
        else
            @constraint(mdl, z[N]>=log(rocket.m_dry))
        end
    end
    
    # Boundary conditions
    @constraint(mdl, r[:,1].==rocket.r0)
    @constraint(mdl, v[:,1].==rocket.v0)
    @constraint(mdl, z[1]==log(rocket.m_wet))

    if ~isnothing(environment)
            # Target boundary constraint-------------------------------------------------------------------------
            min_val,min_idx = findmin(environment.evaluated_risk)
            
            # Using the minimum risk landing position (multiply by 100 for scaling: scale 1x1 risk bound grid -> 100m x 100m landing location grid)
            #target = [environment.x_plot[min_idx]*100, environment.y_plot[min_idx]*100, 10.0]

            # Working test constraint
            #target = [53.0, 99.0, 10.0]
            
            # Set risk-informed landing location
    end

    @constraint(mdl, r[:,N].==rocket.rf)
    #-----------------------------------------------------------------------------------------------------

    # Final velocity constraint
    @constraint(mdl, v[:,N].==rocket.vf)

    # >> Solve problem <<
    optimize!(mdl)
    if termination_status(mdl)!=MOI.OPTIMAL
        return FailedSolution()
    end

    # Save the solution
    r = value.(r)
    v = value.(v)
    z = value.(z)
    u = value.(u)
    ξ = value.(ξ)

    cost = objective_value(mdl)
    m = exp.(z)
    T = RealMatrix(transpose(hcat([m[1:end-1].*u[i,:] for i=1:3]...)))
    T_nrm = RealVector([norm(T[:,i],2) for i=1:N-1])
    γ = RealVector([acos(dot(T[:,k],e_z)/norm(T[:,k],2)) for k=1:N-1])

    sol = Solution(t,r,v,z,u,ξ,cost,T,T_nrm,m,γ)

  

  return sol

end


"""
    optimal_controller(t, x, sol)

Compute the optimal control at time `t` and state `x`, as specified by the
optimal solution `sol`.

# Arguments
- `t`: the current time.
- `x`: the current state.
- `sol`: the full optimal solution object.

# Returns
- `u`: the corresponding optimal input (for the rocket dynamics state-space
  form).
"""
function optimal_controller(t::RealValue,
                            x::RealVector,
                            sol::Solution)::RealVector

    # Get current mass
    z = x[7]
    m = exp.(z)

    # Get current optimal acceleration (ZOH interpolation)
    i = findlast(τ->τ<=t, sol.t)
    if typeof(i)==Nothing || i>=size(sol.u, 2)
        u = sol.u[:, end]
    else
        u = sol.u[:, i]
    end
    # Get current optimal thrust
    T = u*m

    # Create the input vector for the state-space dynamics
    u = RealVector(vcat(T/m, norm(T, 2)/m))

    return u
end # function

"""
    simulate(rocket, control, tf)

Integrate the rocket dynamics using a predefined control input trajectory.

# Arguments
- `rocket`: the rocket object.
- `sol`: the rocket landing optimal solution.

# Returns
- `sim`: the simulation output as a Solution object.
"""
function simulate(rocket::Rocket, sol::Solution)::Solution

    # >> Optimal control getter <<

    control = (t, x) -> optimal_controller(t, x, sol)

    # >> Simulate <<

    dynamics = (t,x) -> rocket.A_c*x+rocket.B_c*control(t, x)+rocket.p_c
    x0 = RealVector(vcat(rocket.r0, rocket.v0, log(rocket.m_wet)))
    tf = sol.t[end]
    Δt = 1e-2
    t = collect(LinRange(0, tf, round(Int, tf/Δt)+1))
    #nolint: rk4
    X = rk4(dynamics, x0, t; full=true)
    U = RealMatrix(hcat([control(t[k], X[:,k]) for k = 1:length(t)]...))
    N = length(t)
    e_z = RealVector([0; 0; 1])

    # >> Save solution <<

    r = X[1:3,:]
    v = X[4:6,:]
    z = X[7,:]
    u = U[1:3,:]
    ξ = U[4,:]

    m = exp.(z)
    T = RealMatrix(transpose(hcat([m.*u[i,:] for i=1:3]...)))
    T_nrm = RealVector([norm(T[:,i],2) for i=1:N])
    γ = RealVector([acos(dot(T[:,k], e_z)/norm(T[:,k],2)) for k=1:N])

    sim = Solution(t,r,v,z,u,ξ,0.0,T,T_nrm,m,γ)

    return sim
end # function

"""
    simulate(rocket, control, tf)

Integrate the rocket dynamics using a predefined control input trajectory.

# Arguments
- `rocket`: the rocket object.
- `sol`: the rocket landing optimal solution.

# Returns
- `sim`: the simulation output as a Solution object.
"""
function get_tight_constraints(rocket::Rocket, sol::Solution)::Dict{String,Any}
    # >> Tight constraint tolerance <<
    ϵ = 1
    
    # >> Discretize [0, tf] interval <<

    # If tf does not divide into rocket.Δt intervals evenly, then reduce Δt by
    # minimum amount to get an integer number of intervals
    tf = sol.t[end]
    #N = Int(floor(tf/rocket.Δt))+1+Int(tf%rocket.Δt!=0) # Number of time nodes
    N = rocket.N
    Δt = tf/(N-1)
    t = RealVector(0.0:Δt:tf)

    A, B, p = c2d(rocket.A_c, rocket.B_c, rocket.p_c, Δt) #noerr

    # >> Optimal control getter <<

    control = (t, x) -> optimal_controller(t, x, sol)

    # >> Simulate <<

    dynamics = (t,x) -> rocket.A_c*x+rocket.B_c*control(t, x)+rocket.p_c
    x0 = RealVector(vcat(rocket.r0, rocket.v0, log(rocket.m_wet)))
    
    t = collect(LinRange(0, tf, round(Int, tf/Δt)+1))
    #nolint: rk4
    X = rk4(dynamics, x0, t; full=true)
    U = RealMatrix(hcat([control(t[k], X[:,k]) for k = 1:length(t)]...))
    N = length(t)
    e_z = RealVector([0; 0; 1])

    # >> Save solution <<

    r = X[1:3,:]
    v = X[4:6,:]
    z = X[7,:]
    u = U[1:3,:]
    ξ = U[4,:]

    m = exp.(z)
    T = RealMatrix(transpose(hcat([m.*u[i,:] for i=1:3]...)))
    T_nrm = RealVector([norm(T[:,i],2) for i=1:N])
    γ = RealVector([acos(dot(T[:,k], e_z)/norm(T[:,k],2)) for k=1:N])

    # Dynamics
    X = (k) -> [r[:,k];v[:,k];z[k]] # State at time index k
    U = (k) -> [u[:,k];ξ[k]] # Input at time index k

    z0 = (k) -> log(rocket.m_wet-rocket.α*rocket.ρ_max*t[k])
    μ_min = (k) -> rocket.ρ_min*exp(-z0(k))
    μ_max = (k) -> rocket.ρ_max*exp(-z0(k))
    δz = (k) -> z[k]-z0(k)

    # Glideslope
    H_gs = RealMatrix([cos(rocket.γ_gs) 0 -sin(rocket.γ_gs);
                        -cos(rocket.γ_gs) 0 -sin(rocket.γ_gs);
                        0 cos(rocket.γ_gs) -sin(rocket.γ_gs);
                        0 -cos(rocket.γ_gs) -sin(rocket.γ_gs)])
    h_gs = zeros(4)

    tight_constraints = Dict(
    "Thrust Bound" => [
        [abs(ξ[k] - (μ_min(k)*(1-δz(k)+0.5*δz(k)^2))) <= ϵ*5 for k=1:N-1],
        [abs(ξ[k] - (μ_max(k)*(1-δz(k)))) <= ϵ*5 for k=1:N-1],
        [abs(ξ[k] - (norm(u[:,k]))) <= ϵ*5 for k=1:N-1]
    ],
    "Mass Bound" => [
        [abs(z0(k) - z[k]) <= ϵ*1e-3 for k=1:N],
        [abs(z[k] - (log(rocket.m_wet-rocket.α*rocket.ρ_min*t[k]))) <= ϵ*1e-3 for k=1:N]
    ],
    "Attitude Pointing" => [
        [abs(dot(u[:,k], e_z) - (ξ[k]*cos(rocket.γ_p))) <= ϵ*1e-3 for k=1:N-1]
    ],
    "Glideslope" => [
        [minimum(abs.(H_gs*r[:,k] - h_gs)) <= ϵ*1e-3 for k=1:N]
    ],
    "Velocity Upper Bound" => [
        [abs(rocket.v_max - norm(v[:,k])) <= ϵ*5 for k=1:N]
    ],
    "Boundary Conditions" => [
        [abs(z[N] - log(rocket.m_dry)) <= ϵ*1e-3]
    ]
)

    return tight_constraints
end # function
