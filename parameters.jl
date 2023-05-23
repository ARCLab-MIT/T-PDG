#= Lossless convexification rocket landing data structures.

Sequential convex programming algorithms for trajectory optimization.
Copyright (C) 2021 Autonomous Controls Laboratory (University of Washington),
                   and Autonomous Systems Laboratory (Stanford University)

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program.  If not, see <https://www.gnu.org/licenses/>. =#

using LinearAlgebra
using Symbolics
using SCPToolbox
using Debugger
include("obstacles.jl")


# ..:: Data structures ::..

"""
`Rocket` holds the rocket parameters.
"""
struct Rocket
    g::RealVector     # [m/s²] Acceleration due to gravity
    ω::RealVector     # [rad/s] Planet angular velocity
    m_dry::RealValue  # [kg] Dry mass (structure)
    m_wet::RealValue  # [kg] Wet mass (structure+fuel)
    Isp::RealValue    # [s] Specific impulse
    φ::RealValue      # [rad] Rocket engine cant angle
    α::RealValue      # [s/m] 1/(rocket engine exit velocity)
    ρ_min::RealValue  # [N] Minimum thrust
    ρ_max::RealValue  # [N] Maximum thrust
    γ_gs::RealValue   # [rad] Maximum approach angle
    γ_p::RealValue    # [rad] Maximum pointing angle
    v_max::RealValue  # [m/s] Maximum velocity
    r0::RealVector    # [m] Initial position
    rf::RealVector    # [m] Final position
    v0::RealVector    # [m/s] Initial velocity
    vf::RealVector    # [m/s] Final velocity
    N::RealValue     # [s] Number of time nodes
    A_c::RealMatrix   # Continuous-time dynamics A matrix
    B_c::RealMatrix   # Continuous-time dynamics B matrix
    p_c::RealVector   # Continuous-time dynamics p vector
    n::Int            # Number of states
    m::Int            # Number of inputs
    constraints::Dict{String, Bool} # Toggleable list of constraints
end

"""
`Environment` holds the environment parameters.
"""
struct Environment
    obstacles::Vector{Ellipse_Obstacle}
    polynomial_obs::Vector{Num}
    expected_vals::Vector{Any}
    risk_contours::Vector{Tuple{Symbolics.Num, Symbolics.Num}}
    risk_equation::Vector{Symbolics.Num}
    evaluated_risk::Vector{Float64}
    evaluated_test_cond::Vector{Float64}
    x_plot::Vector{Float64}
    y_plot::Vector{Float64}
end

"""
`Solution` stores the LCvx solution.
"""
struct Solution
    # >> Raw data <<
    t::RealVector     # [s] Time vector
    r::RealMatrix     # [m] Position trajectory
    v::RealMatrix     # [m/s] Velocity trajectory
    z::RealVector     # [log(kg)] Log(mass) history
    u::RealMatrix     # [m/s^2] Acceleration vector
    ξ::RealVector     # [m/s^2] Acceleration magnitude
    # >> Processed data <<
    cost::RealValue   # Optimization's optimal cost
    T::RealMatrix     # [N] Thrust trajectory
    T_nrm::RealVector # [N] Thrust norm trajectory
    m::RealVector     # [kg] Mass history
    γ::RealVector     # [rad] Pointing angle
end


# ..:: Methods ::..

"""
    Rocket()

Constructor for the rocket.

# Returns
- `rocket`: the rocket definition.
"""
function Rocket(g, θ, T_sidereal_mars, m_dry, m_wet, Isp, n_eng, φ, T_max, γ_gs, γ_p, v_max, r0, rf, v0, vf, N, constraints)::Rocket

    e_x = RealVector([1,0,0])
    e_y = RealVector([0,1,0])
    e_z = RealVector([0,0,1])
    ω = (2π/T_sidereal_mars)*(e_x*cos(θ)+e_y*0+e_z*sin(θ)) # [rad/s^2]
    T_1 = 0.3*T_max # [N] Min allowed thrust of single engine
    T_2 = 0.8*T_max # [N] Max allowed thrust of single engine
    ρ_min = n_eng*T_1*cos(φ)
    ρ_max = n_eng*T_2*cos(φ)

    # >> Continuous-time dynamics <<
    gₑ = 9.807 # Standard gravity
    α = 1/(Isp*gₑ*cos(φ))
    ω_x = skew(ω) #noerr
    A_c = RealMatrix([zeros(3,3) I(3) zeros(3);
                      -(ω_x)^2 -2*ω_x zeros(3);
                      zeros(1,7)])
    B_c = RealMatrix([zeros(3,4);
                      I(3) zeros(3,1);
                      zeros(1,3) -α])
    p_c = RealVector(vcat(zeros(3),g,0))
    n,m = size(B_c)
    
    # >> Make rocket object <<
    rocket = Rocket(g,ω,m_dry,m_wet,Isp,φ,α,ρ_min,ρ_max,γ_gs,γ_p,v_max,
                    r0,rf,v0,vf,N,A_c,B_c,p_c,n,m,constraints)

    return rocket
end # function

"""
    τ()

Constructor for the constraints.

# Returns
- `environment`: the environment definition.
"""
function τ()::Environment
    # Number of obstacles to generate
    n = 3
    # Landing location size (using [0,1] for ease of computation and projecting back to the original landing location size)
    x_range = [0.0, 1.0]
    y_range = [0.0, 1.0]
    # Crater size: https://astronomy.swin.edu.au/~smaddiso/astro/moon/craters.html#:~:text=Most%20craters%20between%2020%20and,least%20300%20km%20in%20diameter.
    # Range for major axis values
    a_range = [.1, .2]
    # Range for minor axis values
    b_range = [.05, .1]

    # Generate a set of random elliptical obstacles for the given parameters
    obstacles = generateObstacles(n,x_range,y_range,a_range,b_range)
    
    # Pregenerated obstacle options:
    #obstacles = Ellipse_Obstacle[Ellipse_Obstacle(342.9, 278.6, 2.95, 162.6, 140.3), Ellipse_Obstacle(415.5, 432.7, 5.21, 109.1, 93.9), Ellipse_Obstacle(31.0, 60.2, 5.41, 80.1, 120.9), Ellipse_Obstacle(45.9, 114.6, 4.34, 21.6, 80.8), Ellipse_Obstacle(367.5, 487.1, 0.4, 31.7, 31.8), Ellipse_Obstacle(86.8, 434.0, 3.7, 32.0, 20.7), Ellipse_Obstacle(375.5, 89.5, 5.3, 106.5, 67.9), Ellipse_Obstacle(411.4, 71.3, 0.78, 60.5, 145.5), Ellipse_Obstacle(340.1, 301.2, 1.1400000000000001, 41.2, 159.3), Ellipse_Obstacle(78.1, 212.4, 0.63, 30.3, 46.9)]
    #obstacles = Ellipse_Obstacle[Ellipse_Obstacle(0, 0, 0, .4, .4)]

    # Landing site visualization
    #plotObstacles(obstacles,x_range,y_range)

    # Use Julia Symbolics to generate polynomial equations for each elliptical obstacle
    poly = generatePolynomial(obstacles)

    # l is the lower bound for the distribution that describes the uncertain parameters (major and minor axes)
    # u is the upper bound for the distribution that describes the uncertain parameters (major and minor axes)
    l = [ [a_range[1], b_range[1]] for i in 1:n ]
    u = [ [a_range[2], b_range[2]] for i in 1:n ]

    # Use Julia Symbolics to calculate the expected values or moments for the set of obstacles
    exp_vals = GenerateExpectedValues(poly,l,u)

    # Use the calucated moments to compute the inner approximation of the risk countor (Equation 10 in https://arxiv.org/pdf/2106.05489.pdf)
    risk_contours = generateRiskContours(exp_vals)

    # Sum the first equation in the risk contour set described in Equation 10 for all obstacles to generate a total risk equation
    risk_equation = generateTotalRiskEquation(risk_contours)

    # Determine resolution for evaluating the risk contour over the landing site (higher resolutions take a longer time to evaluate: O(N^2) runtime complexity)
    step_size = 0.01

    # Evaluate the risk contour equation over the discretized landing site
    (evaluated_risk, evaluated_test_cond, x_plot, y_plot) = calculateRisk(risk_equation, x_range, y_range, step_size)

    #plotRisk(evaluated_risk, x_plot, y_plot)

    # Save all environment information to the Environment structure
    environment = Environment(obstacles, poly, exp_vals, risk_contours, risk_equation, evaluated_risk, evaluated_test_cond, x_plot, y_plot)

    return environment
    
end # function

"""
    Environment()

Constructor for the environment.

# Returns
- `environment`: the environment definition.
"""
function Environment()::Environment
    # Number of obstacles to generate
    n = 3
    # Landing location size (using [0,1] for ease of computation and projecting back to the original landing location size)
    x_range = [0.0, 1.0]
    y_range = [0.0, 1.0]
    # Crater size: https://astronomy.swin.edu.au/~smaddiso/astro/moon/craters.html#:~:text=Most%20craters%20between%2020%20and,least%20300%20km%20in%20diameter.
    # Range for major axis values
    a_range = [.1, .2]
    # Range for minor axis values
    b_range = [.05, .1]

    # Generate a set of random elliptical obstacles for the given parameters
    obstacles = generateObstacles(n,x_range,y_range,a_range,b_range)
    
    # Pregenerated obstacle options:
    #obstacles = Ellipse_Obstacle[Ellipse_Obstacle(342.9, 278.6, 2.95, 162.6, 140.3), Ellipse_Obstacle(415.5, 432.7, 5.21, 109.1, 93.9), Ellipse_Obstacle(31.0, 60.2, 5.41, 80.1, 120.9), Ellipse_Obstacle(45.9, 114.6, 4.34, 21.6, 80.8), Ellipse_Obstacle(367.5, 487.1, 0.4, 31.7, 31.8), Ellipse_Obstacle(86.8, 434.0, 3.7, 32.0, 20.7), Ellipse_Obstacle(375.5, 89.5, 5.3, 106.5, 67.9), Ellipse_Obstacle(411.4, 71.3, 0.78, 60.5, 145.5), Ellipse_Obstacle(340.1, 301.2, 1.1400000000000001, 41.2, 159.3), Ellipse_Obstacle(78.1, 212.4, 0.63, 30.3, 46.9)]
    #obstacles = Ellipse_Obstacle[Ellipse_Obstacle(0, 0, 0, .4, .4)]

    # Landing site visualization
    #plotObstacles(obstacles,x_range,y_range)

    # Use Julia Symbolics to generate polynomial equations for each elliptical obstacle
    poly = generatePolynomial(obstacles)

    # l is the lower bound for the distribution that describes the uncertain parameters (major and minor axes)
    # u is the upper bound for the distribution that describes the uncertain parameters (major and minor axes)
    l = [ [a_range[1], b_range[1]] for i in 1:n ]
    u = [ [a_range[2], b_range[2]] for i in 1:n ]

    # Use Julia Symbolics to calculate the expected values or moments for the set of obstacles
    exp_vals = GenerateExpectedValues(poly,l,u)

    # Use the calucated moments to compute the inner approximation of the risk countor (Equation 10 in https://arxiv.org/pdf/2106.05489.pdf)
    risk_contours = generateRiskContours(exp_vals)

    # Sum the first equation in the risk contour set described in Equation 10 for all obstacles to generate a total risk equation
    risk_equation = generateTotalRiskEquation(risk_contours)

    # Determine resolution for evaluating the risk contour over the landing site (higher resolutions take a longer time to evaluate: O(N^2) runtime complexity)
    step_size = 0.01

    # Evaluate the risk contour equation over the discretized landing site
    (evaluated_risk, evaluated_test_cond, x_plot, y_plot) = calculateRisk(risk_equation, x_range, y_range, step_size)

    #plotRisk(evaluated_risk, x_plot, y_plot)

    # Save all environment information to the Environment structure
    environment = Environment(obstacles, poly, exp_vals, risk_contours, risk_equation, evaluated_risk, evaluated_test_cond, x_plot, y_plot)

    return environment
    
end # function


"""
    FailedSolution()

Constructor for a failure solution.

# Arguments
- `sol`: a standard "failed" solution.
"""
function FailedSolution()::Solution

    t = RealVector(undef,0)
    r = RealMatrix(undef,0,0)
    v = RealMatrix(undef,0,0)
    z = RealVector(undef,0)
    u = RealMatrix(undef,0,0)
    ξ = RealVector(undef,0)
    cost = Inf
    T = RealMatrix(undef,0,0)
    T_nrm = RealVector(undef,0)
    m = RealVector(undef,0)
    γ = RealVector(undef,0)

    sol = Solution(t,r,v,z,u,ξ,cost,T,T_nrm,m,γ)

    return sol
end # function
