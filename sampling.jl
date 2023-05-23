using Distributed
using Dates
@everywhere using LinearAlgebra
@everywhere using Distributions
@everywhere using StatsBase
@everywhere using CSV
@everywhere using SharedArrays
@everywhere include("../Tests/run_tests.jl")
@everywhere function uniformSphereSample(center::Array, radius::Array, n::Int)

    angle_radius = radius[1]
    vector_radius = radius[2]

    # Generate n points from uniform distribution, stored for each parameter
    points = []
    i = 1
    for parameter in center
        # Angles case
        if parameter isa RealValue
            angle_points = rand(Uniform(-radius[i]/2, radius[i]/2), n, 1) .+ parameter
            append!(points, [angle_points])

        # Vectors case
        elseif parameter isa RealVector 
             vector_points = rand(Uniform(-radius[i]/2, radius[i]/2), n, 3) .+ transpose(parameter)
             append!(points, [vector_points])
        end
        i += 1
    end

    # Restructure array to group each index of parameter with each other
    samples = []
    for i in range(1, n)
        sample = [[points[1][i], points[2][i], points[3][i], points[4][i,:], points[5][i,:], points[6][i,:]]]
        append!(samples, sample)
    end

    return samples

end

function Sample(theta0::Array, radius::Array, epsilon::Float64, numSamples::Int)

    # Initialize sample database and strategy
    Thetas = SharedMatrix{Float64}(12, numSamples)
    FinalTimes = SharedVector{Float64}(numSamples)
    VelocityUpper = SharedMatrix{Float64}(50, numSamples)
    Glideslope = SharedMatrix{Float64}(50, numSamples)
    MassBound_1 = SharedMatrix{Float64}(50, numSamples)
    MassBound_2 = SharedMatrix{Float64}(50, numSamples)
    ThrustBound_x = SharedMatrix{Float64}(49, numSamples)
    ThrustBound_y = SharedMatrix{Float64}(49, numSamples)
    ThrustBound_z = SharedMatrix{Float64}(49, numSamples)
    AttitudePointing = SharedMatrix{Float64}(49, numSamples)
    BoundaryConditions = SharedVector{Float64}(numSamples)
    
    succeeded = 0
    failed = 0
    unseen_prob = 1
    

    @sync @distributed for i=1:Int(numSamples)

        theta_new = uniformSphereSample(theta0, radius, 1)[1]
        thetaIndex = 12 * (i-1) + 1
        nIndex = 50 * (i-1) + 1
        n_1Index = 49 * (i-1) + 1
        n = 1 * (i-1) 
        
        try
            soln, strategy, switch_positions_dict = solve_problem(theta_new..., false, false, false)
    
                    # store vals in SharedArrays
                    Thetas[thetaIndex:thetaIndex+11] = collect(Iterators.flatten(theta_new))
                    FinalTimes[n+1] = strategy[2]
                    VelocityUpper[nIndex:nIndex+49] = strategy[1]["Velocity Upper Bound"][1]
                    Glideslope[nIndex:nIndex+49] = strategy[1]["Glideslope"][1]
                    MassBound_1[nIndex:nIndex+49] = strategy[1]["Mass Bound"][1]
                    MassBound_2[nIndex:nIndex+49] = strategy[1]["Mass Bound"][2]
                    ThrustBound_x[n_1Index:n_1Index+48] = strategy[1]["Thrust Bound"][1]
                    ThrustBound_y[n_1Index:n_1Index+48] = strategy[1]["Thrust Bound"][2]
                    ThrustBound_z[n_1Index:n_1Index+48] = strategy[1]["Thrust Bound"][3]
                    AttitudePointing[n_1Index:n_1Index+48] = strategy[1]["Attitude Pointing"][1]
                    BoundaryConditions[n+1] = strategy[1]["Boundary Conditions"][1][1]

            succeeded += 1
       
        catch 
            failed += 1                
        end
    end
    println("Succeeded: ", succeeded, " Failed: ", failed)
        
    
        # Get frequency of strategies
        # strat_counts = Dict()
        # for (key, strat) in Database
        #     if haskey(strat_counts, strat)
        #         strat_counts[strat] += 1
        #     else
        #         strat_counts[strat] = 1
        #     end
        # end

        # display(strat_counts)

        # # Ensure that there are samples 
        # if length(strat_counts) != 0

        #     # Compute Good Turing probability of unseen strategies
        #     unseen_prob = good_turing(strat_counts)
        #     display(unseen_prob)
        # end


    return Thetas, FinalTimes, VelocityUpper, Glideslope, ThrustBound_x, ThrustBound_y, 
    ThrustBound_z, MassBound_1, MassBound_2, AttitudePointing, BoundaryConditions

end

function good_turing(freq::Dict)

    n = length(freq)

    r_counts = Dict()
    # Get frequencies of frequencies
    for (key, val) in freq
        r_counts[val] = get!(r_counts, val, 0) + 1
    end

    r_count = sort(collect(r_counts))
    display(r_counts)

    r_star = Dict()

    # Get probabilities of unobserved counts
    for (i, (r, count)) in enumerate(r_count)
        display(i)
        if i == length(r_counts)
            r_star[r] = r
        else
            r_star[r] = ((r+1)*r_counts[i+1])/count
        end
    end

    # Create dict of frequencies and their probabilities
    gt_dict = Dict()
    for (k, v) in freq
        if v == 1
            gt_dict[k] = r_star[1]
        elseif haskey(r_star, v)
            gt_dict[k] = r_star[v]
        else
            gt_dict[k] = r_star[1]
        end
    end
    # Get the total probability with count 1
    unseen_prob = sum([gt_dict[k] for (k, v) in freq if v == 1]) / length(freq)
    return unseen_prob
end


# Test sampling
numSamples = 10
e_x = RealVector([1,0,0])
e_y = RealVector([0,1,0])
e_z = RealVector([0,0,1])
theta0 = [10*π/180, 80*π/180, 60*π/180, (2*e_x+2*e_y+1*e_z)*1e3, 0*e_x+0*e_y+0*e_z, -15*e_x-15*e_y-30*e_z]
radius = [10*pi/180, 10*pi/180, 10*pi/180, 100, 100, 10]
Thetas, FinalTimes, VelocityUpper, Glideslope, ThrustBound_x, ThrustBound_y, ThrustBound_z, 
MassBound_1, MassBound_2, AttitudePointing, BoundaryConditions = Sample(theta0, radius, 0.9, numSamples)

# Restructure SharedArrays into DataFrame
db = DataFrame("Theta"=>[], "Final Time"=>[], "Velocity Upper Bound"=>[], "Glideslope"=>[], "Thrust Bound (x)"=>[], "Thrust Bound (y)"=>[], 
"Thrust Bound (z)"=>[], "Mass Bound 1"=>[], "Mass Bound 2"=>[], "Attitude Pointing Bound"=>[], "Boundary Conditions"=>[])

for i=1:numSamples
    db[!, "Theta"] = append!( db[!, "Theta"], [Thetas[ (i-1)*12+1:(i-1)*12+12 ]] )
    db[!, "Final Time"] = append!( db[!, "Final Time"], [FinalTimes[ i ]] ) 
    db[!, "Velocity Upper Bound"] = append!( db[!, "Velocity Upper Bound"], [VelocityUpper[ (i-1)*50+1:(i-1)*50+50 ]] ) 
    db[!, "Glideslope"] = append!( db[!, "Glideslope"], [Glideslope[ (i-1)*50+1:(i-1)*50+50 ]] )
    db[!, "Thrust Bound (x)"] = append!( db[!, "Thrust Bound (x)"], [ThrustBound_x[ (i-1)*49+1:(i-1)*49+49 ]] )
    db[!, "Thrust Bound (y)"] = append!( db[!, "Thrust Bound (y)"], [ThrustBound_y[ (i-1)*49+1:(i-1)*49+49 ]] ) 
    db[!, "Thrust Bound (z)"] = append!( db[!, "Thrust Bound (z)"], [ThrustBound_z[ (i-1)*49+1:(i-1)*49+49 ]] )
    db[!, "Mass Bound 1"] = append!( db[!, "Mass Bound 1"], [MassBound_1[ (i-1)*50+1:(i-1)*50+50 ]] )
    db[!, "Mass Bound 2"] = append!( db[!, "Mass Bound 2"], [MassBound_2[ (i-1)*50+1:(i-1)*50+50 ]] )
    db[!, "Attitude Pointing Bound"] = append!( db[!, "Attitude Pointing Bound"], [AttitudePointing[(i-1)*49+1:(i-1)*49+49]] )
    db[!, "Boundary Conditions"] = append!( db[!, "Boundary Conditions"], [BoundaryConditions[ i ]])
end

# Get rid of any failed samples
badSamples = []

for i=1:numSamples
    if Thetas[(i-1)*12+1:(i-1)*12+12] == zeros(12)
        append!(badSamples, i)
    end
end

deleteat!(db, badSamples)
csvFilename = "./RiskGuaranteedGuidance/src/Sampling/" * Dates.format(now(), "mm-dd-yy_HH-MM") * "_database.csv"
CSV.write(csvFilename, db)