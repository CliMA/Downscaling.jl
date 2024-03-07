@testset "ResnetBlockDDN" begin
    # constructor
    channels = 6 => 12
    nspatial = 2
    nembed = 128
    resnetblock = ResnetBlockDDN(channels, nspatial, nembed)
    @test resnetblock isa ResnetBlockDDN

    x = randn(FT, 12, 12, 6, 21)
    tembed = randn(FT, 128, 21)
    @test size(resnetblock(x, tembed)) == (12, 12, 12, 21)
    @test eltype(resnetblock(x, tembed)) == FT
end


@testset "CircularConv" begin
    channels = 1 => 1
    nspatial = 2
    kernel_size = 5
    stride = 1
    conv_kernel = Tuple(kernel_size for _ in 1:nspatial)
    pad = Tuple(div(kernel_size,2) for _ in 1:2*nspatial)
    conv = Conv(conv_kernel, channels; stride=stride, pad=0)
    circ_conv = CircularConv(kernel_size, nspatial, channels; stride = stride)
    @test circ_conv.pad == pad
    @test typeof(circ_conv.conv) == typeof(conv)
    circ_conv.conv.weight .= 1.0f0
    conv.weight .= 1.0f0
    nx = 8
    ny = 16
    x = randn(FT, nx, ny, 1, 1)
    y = circ_conv(x)
    circ_x = NNlib.pad_circular(x, circ_conv.pad)
    @test circ_x[3:end-2,3:end-2,:,:] == x
    @test circ_x[[1, 2], 3:end-2, :, :] == x[nx-1:nx, :, :, :]
    @test circ_x[nx+kernel_size-2:nx+kernel_size-1, 3:end-2, :, :] == x[[1, 2], :, :, :]
    @test circ_x[3:end-2,[1,2], :, :] == x[:, ny-1:ny, :, :]
    @test circ_x[3:end-2,ny+kernel_size-2:ny+kernel_size-1, :, :] == x[:, [1, 2], :, :]
    @test y == conv(circ_x)
end


@testset "AttentionBlock" begin
    # constructor
    channels = 32 => 32
    nspatial = 2
    attnblock = AttentionBlock(channels, nspatial)
    @test attnblock isa AttentionBlock

    x = randn(FT, 12, 12, 32, 21)
    @test size(attnblock(x)) == (12, 12, 32, 21)
    @test eltype(attnblock(x)) == FT
end


 @testset "DDPM Network" begin
    net = CliMAgen.DenoisingDiffusionNetwork(inchannels=3)
    ps = Flux.params(net)
    k = 5
    x = rand(Float32, 2^k, 2^k, 3, 11)
    t = rand(Float32)
    # forward pass
    @test net(x, t) |> size == size(x)
    # backward pass of dummy loss
    loss, grad = Flux.withgradient(ps) do
        sum(net(x, t) .^ 2)
    end
    @test loss isa Real

    if CUDA.has_cuda()
        net = CliMAgen.DenoisingDiffusionNetwork(inchannels=3) |> Flux.gpu
        ps = Flux.params(net)
        k = 5
        x = rand(Float32, 2^k, 2^k, 3, 11) |> Flux.gpu
        t = rand(Float32) |> Flux.gpu
        # forward pass
        @test net(x, t) |> Flux.cpu |> size == size(x |> Flux.cpu)
        # backward pass of dummy loss
        loss, grad = Flux.withgradient(ps) do
            sum(net(x, t) .^ 2) |> Flux.cpu
        end
        @test loss isa Real
    end
end

@testset "NCSN" begin
    net = CliMAgen.NoiseConditionalScoreNetwork(;noised_channels=2, outer_kernelsize=5, channels=[32, 64, 128, 256])
    @test size(Flux.params(net.layers.tconv2)[1]) == (5, 5, 128, 32)

    ps = Flux.params(net)
    k = 5
    nx = 2^k
    ny = 2^(k-1)
    x = rand(Float32, nx, ny, 2, 11)
    c=nothing
    t = rand(Float32)
    # forward pass
    @test net(x, c, t) |> size == size(x)

    # backward pass of dummy loss
    loss, grad = Flux.withgradient(ps) do
        sum(net(x, c, t) .^ 2)
    end
    @test loss isa Real

    shift_input_net = CliMAgen.NoiseConditionalScoreNetwork(;noised_channels=2, shift_input=true, outer_kernelsize=5,)
    Flux.loadmodel!(shift_input_net, net)
    @test net(x.-mean(x, dims = (1,2)), c, t) ≈ shift_input_net(x, c, t)

    shift_output_net = CliMAgen.NoiseConditionalScoreNetwork(;noised_channels=2, shift_output=true, outer_kernelsize=5,)
    Flux.loadmodel!(shift_output_net, net)
    @test (net(x, c, t) .-mean(net(x,c, t), dims = (1,2))) ≈ shift_output_net(x, c, t)

    mean_bypass_net = CliMAgen.NoiseConditionalScoreNetwork(;noised_channels=2, shift_input=true, shift_output=true, mean_bypass=true)
    ps = Flux.params(net)
    k = 5
    x = rand(Float32, nx, ny, 2, 11)
    t = rand(Float32)
    # forward pass
    @test  mean_bypass_net(x, c, t) |> size == size(x)

    # backward pass of dummy loss
    loss, grad = Flux.withgradient(ps) do
        sum(net(x, c, t) .^ 2)
    end
    @test loss isa Real
end


@testset "NCSN in and out channels" begin
    # with context
    net = CliMAgen.NoiseConditionalScoreNetwork(context=true, noised_channels=2, context_channels = 3)
    ps = Flux.params(net)
    k = 5
    nx = 2^k
    ny = 2^(k-1)
    x = rand(Float32, nx, ny, 2, 11)
    c = rand(Float32, nx, ny, 3, 11)
    t = rand(Float32)
    # forward pass
    @test net(x, c, t) |> size == (nx, ny, 2, 11)

    # backward pass of dummy loss
    loss, grad = Flux.withgradient(ps) do
        sum(net(x, c, t) .^ 2)
    end
    @test loss isa Real

    # once without context (default is context=false and context_channels = 0)
    net2 = CliMAgen.NoiseConditionalScoreNetwork(noised_channels=3)
    ps = Flux.params(net2)
    k = 5
    x = rand(Float32, nx, ny, 3, 11)
    t = rand(Float32)
    # forward pass
    @test net2(x, nothing, t) |> size == (nx, ny, 3, 11)

    # backward pass of dummy loss
    loss, grad = Flux.withgradient(ps) do
        sum(net2(x, c, t) .^ 2)
    end
    @test loss isa Real
end

@testset "Periodic NCSN" begin
    k = 5
    x = zeros(Float32, 2^k, 2^k, 1, 1)
    xx = Float32.(1:1:2^k)
    for i in 1:2^k
        for j in 1:2^k
            x[i,j] = sin(2π*xx[i]*2/2^k-π/2) + 2*abs(xx[j]-2^k/2)/2^k
        end
    end
    c=nothing
    t = rand(Float32)

    periodic_net = CliMAgen.NoiseConditionalScoreNetwork(;noised_channels=1, outer_kernelsize=5, channels=[32, 64, 128, 256], periodic=true)
    periodic_sc = periodic_net(x, c, t)

    nonperiodic_net = CliMAgen.NoiseConditionalScoreNetwork(;noised_channels=1, outer_kernelsize=5, channels=[32, 64, 128, 256]);
    nonperiodic_sc = nonperiodic_net(x,c,t);
end

@testset "NCSN3D" begin
    # test for correct kernel shape
    net = CliMAgen.NoiseConditionalScoreNetwork3D(;noised_channels=2, outer_kernelsize=5, nspatial=3, channels=[32, 64, 128, 256])
    @test size(Flux.params(net.layers.tconv2)[1]) == (5, 5, 5, 128, 32)

    # test forward pass
    k = 4
    x = rand(Float32, 2^k, 2^k, 16, 2, 11)
    t = rand(Float32)
    c=nothing
    @test net(x, c, t) |> size == size(x)

    # test backward pass
    ps = Flux.params(net)
    loss, grad = Flux.withgradient(ps) do
        sum(net(x, c, t) .^ 2)
    end
    @test loss isa Real
end