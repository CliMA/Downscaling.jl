using CliMAgen
using Flux
using CUDA
using cuDNN
using Test
using Random
using Statistics

FT = Float32
include("./test_sampling.jl")
include("./test_preprocessing.jl")
include("./tests_models.jl")
include("./test_networks.jl")
include("./tests_optimizers.jl")
include("./tests_utils.jl")
