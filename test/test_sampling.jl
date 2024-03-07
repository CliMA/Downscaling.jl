@testset "Biased vs unbiased sampling" begin
    net = CliMAgen.NoiseConditionalScoreNetwork(noised_channels=1)
    σ_min = FT(1e-2)
    σ_max = FT(1)
    model = VarianceExplodingSDE{FT, typeof(net)}(σ_max, σ_min, net)
    nx = 32
    ny = 16
    init_x = rand(Float32, nx, ny, 1, 1)
    ϵ=FT(1.0f-3)
    num_steps = 2
    time_steps = LinRange(FT(1.0), ϵ, num_steps)
    Δt = time_steps[1] - time_steps[2]
    samples_no_bias = Euler_Maruyama_ld_sampler(model, deepcopy(init_x), time_steps, Δt)
    samples_original = Euler_Maruyama_sampler(model, deepcopy(init_x), time_steps, Δt)
    bias_function = x -> fill!(similar(x), eltype(x)(0))
    samples_zero_bias = Euler_Maruyama_ld_sampler(model, deepcopy(init_x), time_steps, Δt, bias = bias_function, use_shift = true)
    @test samples_no_bias == samples_original
    @test samples_zero_bias == samples_original
end
