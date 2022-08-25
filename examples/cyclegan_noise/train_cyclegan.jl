using Flux 
using Flux: params, update!, loadmodel!
using FluxTraining
using BSON: @save, @load
using CUDA
using Dates
using HDF5
using MLUtils
using ProgressBars
using Statistics: mean

using Downscaling

examples_dir = joinpath(pkgdir(Downscaling), "examples")
cyclegan_dir = joinpath(examples_dir, "cyclegan_noise")
include(joinpath(cyclegan_dir, "utils.jl"))
include(joinpath(examples_dir, "artifact_utils.jl"))

# Parameters
Base.@kwdef struct HyperParams{FT}
    λ = FT(10.0)
    λid = FT(5.0)
    lr = FT(0.0002)
    nepochs = 100
end

function generator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, hparams)
    a_fake = generator_B(b, noise) # Fake image generated in domain A
    b_fake = generator_A(a, noise) # Fake image generated in domain B

    b_fake_prob = discriminator_B(b_fake) # Probability that generated image in domain B is real
    a_fake_prob = discriminator_A(a_fake) # Probability that generated image in domain A is real

    gen_A_loss = mean((a_fake_prob .- 1) .^ 2)
    rec_A_loss = mean(abs.(b - generator_A(a_fake, noise))) # Cycle-consistency loss for domain B
    idt_A_loss = mean(abs.(generator_A(b, noise) .- b)) # Identity loss for domain B
    gen_B_loss = mean((b_fake_prob .- 1) .^ 2)
    rec_B_loss = mean(abs.(a - generator_B(b_fake, noise))) # Cycle-consistency loss for domain A
    idt_B_loss = mean(abs.(generator_B(a, noise) .- a)) # Identity loss for domain A

    return gen_A_loss + gen_B_loss + hparams.λ * (rec_A_loss + rec_B_loss) + hparams.λid * (idt_A_loss + idt_B_loss)
end

function discriminator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise)
    a_fake = generator_B(b, noise) # Fake image generated in domain A
    b_fake = generator_A(a, noise) # Fake image generated in domain B

    a_fake_prob = discriminator_A(a_fake) # Probability that generated image in domain A is real
    a_real_prob = discriminator_A(a) # Probability that an original image in domain A is real
    b_fake_prob = discriminator_B(b_fake) # Probability that generated image in domain B is real
    b_real_prob = discriminator_B(b) # Probability that an original image in domain B is real

    real_A_loss = mean((a_real_prob .- 1) .^ 2)
    fake_A_loss = mean((a_fake_prob .- 0) .^ 2)
    real_B_loss = mean((b_real_prob .- 1) .^ 2)
    fake_B_loss = mean((b_fake_prob .- 0) .^ 2)

    return real_A_loss + fake_A_loss + real_B_loss + fake_B_loss
end

function train_step!(opt_gen, opt_dis, generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, hparams)
    # Optimize Discriminators
    ps = params(params(discriminator_A)..., params(discriminator_B)...)
    gs = gradient(() -> discriminator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise), ps)
    update!(opt_dis, ps, gs)

    # Optimize Generators
    ps = params(params(generator_A)..., params(generator_B)...)
    gs = gradient(() -> generator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, hparams), ps)
    update!(opt_gen, ps, gs)
end

function fit!(opt_gen, opt_dis, generator_A, generator_B, discriminator_A, discriminator_B, data, hparams, output_filepath)
    # Training loop
    g_loss, d_loss = 0, 0
    iter = ProgressBar(data) 
    for epoch in 1:hparams.nepochs
        for (a, b, noise) in iter
            train_step!(opt_gen, opt_dis, generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, hparams)
            set_multiline_postfix(iter, "Epoch $epoch\nGenerator Loss: $g_loss\nDiscriminator Loss: $d_loss")
        end

        # print current error estimates
        a, b, noise = first(data)
        g_loss = generator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b, noise, hparams)
        d_loss = discriminator_loss(generator_A, generator_B, discriminator_A, discriminator_B, a, b,noise)

        # store current model
        model = (generator_A, generator_B, discriminator_A, discriminator_B) |> cpu
        @save output_filepath model opt_gen opt_dis
    end
end

function train(path, field, hparams, output_filepath; cuda=true, restart = false)
    if cuda && CUDA.has_cuda()
        dev = gpu
        CUDA.allowscalar(false)
        @info "Training on GPU"
    else
        dev = cpu
        @info "Training on CPU"
    end

    # training data
    data = get_dataloader(path, field=field, split_ratio=0.5, batch_size=1, dev=dev).training

    nchannels = 1
    if restart && isfile(output_filepath)
        @info "Initializing with existing model and optimizers"
        
        # First we need to make the model structure
        generator_A = NoisyUNetGenerator(nchannels) # Generator For A->B
        generator_B = NoisyUNetGenerator(nchannels) # Generator For B->A
        discriminator_A = PatchDiscriminator(nchannels) # Discriminator For Domain A
        discriminator_B = PatchDiscriminator(nchannels) # Discriminator For Domain B
        
        # Now load the existing model parameters and fill in the parameters of the models we just made
        # This also loads the optimizers
        @load output_filepath model opt_gen opt_dis
        loadmodel!(generator_A, model[1])
        loadmodel!(generator_B, model[2])
        loadmodel!(discriminator_A, model[3])
        loadmodel!(discriminator_B, model[4])

        # Push to device
        generator_A = generator_A |> dev
        generator_B = generator_B |> dev
        discriminator_A = discriminator_A |> dev
        discriminator_B = discriminator_B |> dev
    else
        @info "Initializing a new model and optimizers from scratch"
        generator_A = NoisyUNetGenerator(nchannels) |> dev # Generator For A->B
        generator_B = NoisyUNetGenerator(nchannels) |> dev # Generator For B->A
        discriminator_A = PatchDiscriminator(nchannels) |> dev # Discriminator For Domain A
        discriminator_B = PatchDiscriminator(nchannels) |> dev # Discriminator For Domain B

        opt_gen = ADAM(hparams.lr, (0.5, 0.999))
        opt_dis = ADAM(hparams.lr, (0.5, 0.999))
    end

    fit!(opt_gen, opt_dis, generator_A, generator_B, discriminator_A, discriminator_B, data, hparams, output_filepath)
end

# run if file is called directly but not if just included
if abspath(PROGRAM_FILE) == @__FILE__

    # This downloads the data locally, if it not already present, and obtains the location of the directory holding it.
    local_dataset_directory = obtain_local_dataset_path(examples_dir, moist2d.dataname, moist2d.url, moist2d.filename)
    local_dataset_path = joinpath(local_dataset_directory, moist2d.filename)

    output_dir = joinpath(cyclegan_dir, "output")
    mkpath(output_dir)
    output_filepath = joinpath(output_dir, "checkpoint_latest.bson")
    field = "moisture"
    hparams = HyperParams{Float32}()
    train(local_dataset_path, field, hparams, output_filepath; restart = true)
end
