include("../src/RiskGuaranteedGuidance.jl")
using Main.RiskGuaranteedGuidance.RocketLanding
using BenchmarkTools
using PyCall
using Qt5Base_jll
# Required packages for the script
using PyCall
using DataFrames
using Plots
using JLD2



#using StatProfilerHTML
include("../src/tests.jl")
include("../src/parameters.jl")
include("../src/definition.jl")
include("plots.jl")
#@btime realtimeGuidance();

# Python NN helper functions

    # Predict the tight constraints for the strategy
    pytorch = pyimport("torch")
    nn = pyimport("torch.nn")
    optim = pyimport("torch.optim")
    @pyimport torch.utils.data as tdata
    @pyimport numpy as np
    @pyimport torch

    py"""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np
    import math
    from torch.utils.data import DataLoader

    class TransformerModel(nn.Module):
        def __init__(self, input_size, output_size, d_model, nhead, num_layers):
            super(TransformerModel, self).__init__()

            self.d_model = d_model
            self.encoder = nn.Linear(input_size, d_model)
            self.pos_encoder = PositionalEncoding(d_model)
            self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, nhead), num_layers)
            self.decoder = nn.Linear(d_model, output_size)

        def forward(self, src):
            # src shape: (batch_size, input_size)
            x = self.encoder(src) * math.sqrt(self.d_model)
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
            x = self.decoder(x[:, -1, :])
            return x
        
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.dropout = nn.Dropout(p=0.1)
            
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0).transpose(0, 1)
            self.register_buffer('pe', pe)

        def forward(self, x):
            x = x + self.pe[:x.size(0), :]
            return self.dropout(x)

    def load_model(model_path):
        model = torch.load(model_path)
        model.eval()
        return model
    """

    py"""
    def predict_with_model(model, input_data):

        with torch.no_grad():
                #inputs = torch.tensor(input_data, dtype=torch.float32).unsqueeze(1)
                outputs = model(input_data)
                return outputs.numpy().tolist(), input_data.numpy().tolist()

    """
        
    py"""
    import numpy as np
    import torch
    def create_tensor(data_path):
        data = torch.load(data_path)
        return data
    """
    

    constraints_model_path = "RiskGuaranteedGuidance/src/Model/transformer_model_vf_0_constraints_normalized.pt"
    time_model_path = "RiskGuaranteedGuidance/src/Model/transformer_model_vf_0_time_normalized.pt"
    constraints_data_path = "RiskGuaranteedGuidance/src/Data/normalized_Test/standardized_test_dataset_constraints.pt"
    time_data_path = "RiskGuaranteedGuidance/src/Data/normalized_Test/standardized_test_dataset_time.pt"

    create_tensor = py"create_tensor"
    input_data = create_tensor(constraints_data_path)


    # Load the models
    load_model = py"load_model"
    constraints_model = load_model(constraints_model_path)
    time_model = load_model(time_model_path)


    # Functio to use the model to make a prediction
    predict_with_model = py"predict_with_model"

function T_PDG()

# Get the DataLoader for the input data
test_dataloader = tdata.DataLoader(input_data, batch_size=1, shuffle=false)

# Load Mean and Standard Deviation
mean_data = 552.0598
std_data = 842.1304

# A container to record if the trajectory is feasible

e_x = RealVector([1,0,0])
e_y = RealVector([0,1,0])
e_z = RealVector([0,0,1])

soln_array = []
soln_sim_array = []
rocket_array = []
strategy = []

strategy_time_array = []
feasibility_time_array = []
model_time_array = []
full_problem_time_array = []

is_feasible_reduced = 0
is_feasible_check = 0
is_feasible = 0
total = 0

switch_positions_dict = Dict{Int64, RealVector}()

# For each input in the test dataset
for inputs in test_dataloader


    inputs,labels = inputs

    # Make the prediction with the model
    predicted_constraints, constraint_inputs = predict_with_model(constraints_model, inputs)
    model_time = @elapsed predict_with_model(time_model, inputs)
    predicted_time, time_inputs = predict_with_model(time_model, inputs)
    model_time += @elapsed predict_with_model(constraints_model, inputs)

    # Create the strategy
    strategy = Strategy(RealMatrix(predicted_constraints),RealValue(predicted_time[1]))

    # Extract the parameters from the inputs
    r0 = zeros(3)
    v0 = zeros(3)
    φ, γ_gs, γ_p, r0[1], r0[2], r0[3], v0[1], v0[2], v0[3] = time_inputs*std_data.+mean_data

    rf = 0*e_x+0*e_y+0*e_z

    # Solve the problem
    soln, strategy, switch_positions_dict, soln_sim, rocket, strategy_time, feasibility_time, full_problem_time, is_feasible_reduced_val, is_feasible_check_val = solve_problem(RealValue(φ), RealValue(γ_gs), RealValue(γ_p), RealVector(r0), RealVector(rf), RealVector(v0), strategy, false, true, true)

    if !isinf(soln.cost)
        is_feasible += 1

        push!(model_time_array,model_time)

        push!(soln_array, soln)

        push!(soln_sim_array, soln_sim)

        push!(rocket_array, rocket)

        push!(strategy_time_array, strategy_time)

        push!(feasibility_time_array, feasibility_time)

        push!(full_problem_time_array, full_problem_time)

        is_feasible_reduced += is_feasible_reduced_val

        is_feasible_check += is_feasible_check_val

        #jldsave("RiskGuaranteedGuidance/src/results/T_PDG_results_scitech2.jld2"; is_feasible, model_time_array, soln_array, soln_sim_array, rocket_array, strategy_time_array, feasibility_time_array, full_problem_time_array, is_feasible_reduced, is_feasible_check, total)

        plot_trajectory_gif_array(rocket_array, soln_array, soln_sim_array, "T-PDG Trajectory test 3")
    end
  
    println(soln.cost)
    
    total += 1

    end
end

T_PDG()


