#= Modified from Tests for lossless convexification rocket landing.

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

include("parameters.jl")
include("definition.jl")
include("../Tests/plots.jl")

using Printf
using Test
using Plots
using DataFrames
using SCPToolbox
using CSV
using Tables
using BenchmarkTools

export lcvx

include("../src/T_PDG.jl")
using Main.T_PDG.RocketLanding
using BenchmarkTools
using PyCall

# Structure for changing parameters
struct Problem_Parameters
    φ::RealValue      # [rad] Rocket engine cant angle
    γ_gs::RealValue   # [rad] Maximum approach angle
    γ_p::RealValue    # [rad] Maximum pointing angle
    r0::RealVector    # [m] Initial position
    rf::RealVector    # [m] Final position
    v0::RealVector    # [m/s] Initial velocity
end

# Structure for strategy
struct Strategy
    τ::RealMatrix     # Tight constraints
    t_f::RealValue   # Optimal final time
end

function solve_problem(φ::RealValue, γ_gs::RealValue, γ_p::RealValue, r0::RealVector, rf::RealVector, v0::RealVector, strategy::Strategy, plots_on::Bool, sim_on::Bool, timer_on::Bool)

    # Determine where the strategy switches
    function find_switch_positions(vec::Vector{Bool})
        diffs = diff(vec)
        start_inds = findall(diffs .!= 0)
        end_inds = start_inds .+ 1
        if vec[1]
            pushfirst!(start_inds .- 1, 1)
        end
        if vec[end]
            push!(end_inds, length(vec))
        end
        return Tuple.(zip(start_inds, end_inds))
    end

    # Static Problem parameters
    e_x = RealVector([1,0,0])
    e_y = RealVector([0,1,0])
    e_z = RealVector([0,0,1])
    g = -3.7114*e_z # m/s^2
    θ = 30*π/180 # [rad] Latitude of landing site
    # https://nssdc.gsfc.nasa.gov/planetary/factsheet/moonfact.html
    T_sidereal_mars = 24.6229*3600 # Mars sidereal [s]
    m_dry = 1505.0 # [kg]
    m_wet = 1905.0 # [kg]
    Isp = 225.0 # [s]
    n_eng = 6       # Number of engines
    T_max = 3.1e3   # [N] Max physical thrust of single engine
    v_max = 500*1e3/3600
    vf = 0*e_x+0*e_y+0*e_z
    N = 50 # Number of discretization nodes

    constraints = Dict("Thrust Bound"=>true, "Mass Bound"=>true, "Attitude Pointing"=>true, "Glideslope"=>true, "Velocity Upper Bound"=>true)

    # Dynamic Problem Parameters
    #φ = 27*π/180    # [rad] Engine cant angle off vertical
    #γ_gs = 86*π/180
    #γ_p = 40*π/180
    #r0 = (2*e_x+0*e_y+1.5*e_z)*1e3
    #rf = 0*e_x+0*e_y+0*e_z
    #v0 = 80*e_x+30*e_y-75*e_z

    # Write or recieve as Problem_Parameters data structure
    #problem_parameters = Problem_Parameters(φ,γ_gs,γ_p,r0,rf,v0)

    # Define rocket structure and solve the problem
    rocket = Rocket(g, θ, T_sidereal_mars, m_dry, m_wet, Isp, n_eng, φ, T_max, γ_gs, γ_p, v_max, r0, rf, v0, vf, N, constraints)

    soln,strategy,soln_sim,strategy_time,feasibility_time,full_problem_time,is_feasible_reduced, is_feasible_check = lcvx(plots_on, sim_on, timer_on, rocket, strategy = strategy)

    # Create dictionary of switch switch_positions_dict
    switch_positions_dict = Dict{String, Vector{Vector{Tuple{Int,Int}}}}()

    #for (key, value) in strategy[1]
    #    switch_positions_dict[key] = []
    #    for value_idx in value
    #        switch_positions = find_switch_positions(value_idx)
    #        switch_positions_dict[key] = append!(switch_positions_dict[key],[switch_positions])
    #    end
    #end

    return soln, strategy, switch_positions_dict, soln_sim, rocket, strategy_time, feasibility_time, full_problem_time, is_feasible_reduced, is_feasible_check
end

function lcvx(plots_on::Bool, sim_on::Bool, timer_on::Bool, rocket = Rocket(); strategy = nothing)

    #environment = Environment()
    strategy_time = 0
    feasibility_time = 0
    full_problem_time = 0
    is_feasible_reduced = 0
    is_feasible_check = 0

    tol = 1e-3
    tf_min = rocket.m_dry * norm(rocket.v0, 2) / rocket.ρ_max
    tf_max = (rocket.m_wet - rocket.m_dry) / (rocket.α * rocket.ρ_min)

    if ~isnothing(strategy)
        tf = strategy.t_f
        sim = nothing

        tight_constraints = round.(Int, strategy.τ)

        if timer_on
            #print("Time for reduced problem: ")
            pdg_reduced = solve_pdg_fft(rocket, tf, tight_constraints = tight_constraints); # Optimal 3-DoF PDG trajectory
            strategy_time = @elapsed solve_pdg_fft(rocket, tf, tight_constraints = tight_constraints); # Optimal 3-DoF PDG trajectory
            #print("Time for reduced solution as the initial guess: ")

            full_problem_time = @elapsed golden((tf) -> solve_pdg_fft(rocket, tf).cost, tf_min, tf_max; tol = tol, verbose=false);
            t_opt, cost_opt = golden((tf) -> solve_pdg_fft(rocket, tf).cost, tf_min, tf_max; tol = tol, verbose=false);
            full_problem_time += @elapsed solve_pdg_fft(rocket, t_opt); # Optimal 3-DoF PDG trajectory

            if !isinf(pdg_reduced.cost)
                is_feasible_reduced = 1
                pdg = solve_pdg_fft(rocket, tf, initial_values = pdg_reduced); # Optimal 3-DoF PDG trajectory
                feasibility_time = @elapsed solve_pdg_fft(rocket, tf, initial_values = pdg_reduced); # Optimal 3-DoF PDG trajectory

                if isinf(pdg.cost)
                    #print("Time for full problem: ")                
                    pdg = solve_pdg_fft(rocket, t_opt); # Optimal 3-DoF PDG trajectory

                    feasibility_time += full_problem_time
                else
                    is_feasible_check = 1
                end

            else
                #print("Time for full problem: ")                
                pdg = solve_pdg_fft(rocket, t_opt); # Optimal 3-DoF PDG trajectory

                feasibility_time += full_problem_time
            end


            if sim_on
                if !isinf(pdg.cost)
                    # Continuous-time simulation
                    sim = simulate(rocket, pdg);
                else
                    sim = nothing
                end
            end


        else
            pdg = solve_pdg_fft(rocket, tf, tight_constraints = tight_constraints); # Optimal 3-DoF PDG trajectory
            pdg = solve_pdg_fft(rocket, tf, initial_values = pdg); # Optimal 3-DoF PDG trajectory
        end

    else

        if timer_on
            print("Time for full problem: ")
            @time t_opt, cost_opt = golden((tf) -> solve_pdg_fft(rocket, tf).cost, tf_min, tf_max; tol = tol, verbose=false);
        else
            t_opt, cost_opt = golden((tf) -> solve_pdg_fft(rocket, tf).cost, tf_min, tf_max; tol = tol, verbose=false);
        end
    
        
        pdg = solve_pdg_fft(rocket, t_opt); # Optimal 3-DoF PDG trajectory
    
        @test !isinf(cost_opt)
    
        if sim_on
            # Continuous-time simulation
            sim = simulate(rocket, pdg);
        else
            sim = nothing
        end
    
        # Determine optimal strategy
        tight_constraints = get_tight_constraints(rocket, pdg);
        tf = pdg.t[end]
        strategy = (tight_constraints, tf)
    
        if sim_on
            if timer_on
                print("Time for reduced problem: ")
                @time pdg_w_strategy = solve_pdg_fft(rocket, t_opt, tight_constraints = tight_constraints); # Optimal 3-DoF PDG trajectory
                print("Time for reduced solution as the initial guess: ")
                @time pdg_w_strategy = solve_pdg_fft(rocket, t_opt, initial_values=pdg_w_strategy); # Optimal 3-DoF PDG trajectory
            else 
                pdg_w_strategy = solve_pdg_fft(rocket, t_opt, tight_constraints = tight_constraints);
                pdg_w_strategy = solve_pdg_fft(rocket, t_opt, initial_values=pdg_w_strategy); # Optimal 3-DoF PDG trajectory
            end        
            
            sim_w_strategy = simulate(rocket, pdg_w_strategy);
        else
            sim_w_strategy = nothing
        end

    end


    if plots_on
        # Write output to file
        CSV.write("r.csv",  Tables.table(sim.r), writeheader=false)
        CSV.write("γ.csv",  Tables.table(sim.γ), writeheader=false)
        CSV.write("Thrust.csv",  Tables.table(sim.T), writeheader=false)
        CSV.write("T_nrm.csv",  Tables.table(sim.T_nrm), writeheader=false)

        # Make plots
        
        plot_thrust(rocket, pdg, sim)
        plot_mass(rocket, pdg, sim)
        plot_pointing_angle(rocket, pdg, sim)
        plot_velocity(rocket, pdg, sim)
        plot_position(rocket, pdg, sim)

        plot_trajectory_gif(rocket, pdg, sim, "traj")
        #plot_trajectory_gif(rocket, pdg_w_strategy, sim_w_strategy, "traj_with_strategy")
    end


    return pdg,strategy,sim,strategy_time,feasibility_time,full_problem_time,is_feasible_reduced, is_feasible_check


end # function

