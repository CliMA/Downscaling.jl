using BSON
using Flux: chunk
using Images
using ProgressMeter
using Plots
using Statistics: mean

# our training file
include("train_diffusion_new.jl")

"""
Sample from a diffusion model using the Euler-Maruyama method.

# References
https://arxiv.org/abs/1505.04597
"""
function Euler_Maruyama_sampler(model::DiffusionModels.AbstractDiffusionModel, init_x, time_steps, Δt)
    x = mean_x = init_x

    @showprogress "Euler-Maruyama Sampling" for time_step in time_steps
        batch_time_step = fill!(similar(init_x, size(init_x)[end]), 1) .* time_step
        g = DiffusionModels.diffusion(model, batch_time_step)
        score = DiffusionModels.score(model, x, batch_time_step)

        mean_x = x .+ DiffusionModels.expand_dims(g, 3) .^ 2 .* score .* Δt
        x = mean_x .+ sqrt(Δt) .* DiffusionModels.expand_dims(g, 3) .* randn(Float32, size(x))
    end
    return mean_x
end

function plot_result(model, args)
    args = Args(; args...)
    args.seed > 0 && Random.seed!(args.seed)
    device = args.cuda && CUDA.has_cuda() ? gpu : cpu
    device = cpu
    @info "Using device: $device"
    model = model |> device
    time_steps, Δt, init_x = setup_sampler(model, device)

    # Euler-Maruyama
    euler_maruyama = Euler_Maruyama_sampler(model, init_x, time_steps, Δt)
    sampled_noise = convert_to_image(init_x, size(init_x)[end])
    save(joinpath(args.save_path, "sampled_noise.jpeg"), sampled_noise)
    em_images = convert_to_image(euler_maruyama, size(euler_maruyama)[end])
    save(joinpath(args.save_path, "em_images.jpeg"), em_images)
end

"""
Helper function that produces images from a batch of images.
"""
function convert_to_image(x, y_size)
    Gray.(permutedims(vcat(reshape.(chunk(x |> cpu, y_size), 32, :)...), (2, 1)))
end

"""
Helper to make an animation from a batch of images.
"""
function convert_to_animation(x)
    frames = size(x)[end]
    batches = size(x)[end-1]
    animation = @animate for i = 1:frames+frames÷4
        if i <= frames
            heatmap(
                convert_to_image(x[:, :, :, :, i], batches),
                title="Iteration: $i out of $frames"
            )
        else
            heatmap(
                convert_to_image(x[:, :, :, :, end], batches),
                title="Iteration: $frames out of $frames"
            )
        end
    end
    return animation
end

"""
Helper function that generates inputs to a sampler.
"""
function setup_sampler(model::DiffusionModels.AbstractDiffusionModel, device; num_images=5, num_steps=500, ϵ=1.0f-3)
    t = ones(Float32, num_images) |> device
    init_z = randn(Float32, (32, 32, 1, num_images))
    _, σ_0 = DiffusionModels.marginal_prob(model, zero(init_z), t)
    init_x = (σ_0 .* init_z) |> device
    time_steps = LinRange(1.0f0, ϵ, num_steps)
    Δt = time_steps[1] - time_steps[2]
    return time_steps, Δt, init_x
end

if abspath(PROGRAM_FILE) == @__FILE__
    ############################################################################
    # Issue loading function closures with BSON:
    # https://github.com/JuliaIO/BSON.jl/issues/69
    #
    BSON.@load "output/checkpoint_model.bson" model_ema args
    #
    # BSON.@load does not work if defined inside plot_result(⋅) because
    # it contains a function closure, GaussFourierProject(⋅), containing W.
    ###########################################################################
    plot_result(model_ema, args)
end