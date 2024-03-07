"""
    NoiseConditionalScoreNetwork

The struct containing the parameters and layers
of the Noise Conditional Score Network architecture,
with the option to include a mean-bypass layer.

# References
Unet: https://arxiv.org/abs/1505.04597
"""
struct NoiseConditionalScoreNetwork
    "The layers of the network"
    layers::NamedTuple
    "A boolean indicating if non-noised context channels are present"
    context::Bool
    "A boolean indicating if a mean-bypass layer should be used"
    mean_bypass::Bool
    "A boolean indicating if the output of the mean-bypass layer should be scaled"
    scale_mean_bypass::Bool
    "A boolean indicating if the input is demeaned before being passed to the U-net"
    shift_input::Bool
    "A boolean indicating if the output of the Unet is demeaned"
    shift_output::Bool
    "A boolean indicating if a groupnorm should be used in the mean-bypass layer"
    gnorm::Bool
end

"""
    NoiseConditionalScoreNetwork(; context=false,
                                 mean_bypass=false, 
                                 scale_mean_bypass=false,
                                 shift_input=false,
                                 shift_output=false,
                                 gnorm=false,
                                 nspatial=2,
                                 dropout_p=0.0f0,
                                 num_residual=8,
                                 noised_channels=1,
                                 context_channels=0,
                                 channels=[32, 64, 128, 256],
                                 embed_dim=256,
                                 scale=30.0f0,
                                 periodic=false,
                                 proj_kernelsize=3,
                                 outer_kernelsize=3,
                                 middle_kernelsize=3,
                                 inner_kernelsize=3)

Returns a NoiseConditionalScoreNetwork, given
- context: boolean indicating whether or not contextual information is 
           present
- mean_bypass: boolean indicating if a mean-bypass layer should be used
- scale_mean_bypass: boolean indicating if the output of the mean-bypass 
                     layer should be scaled
- shift_input: boolean indicating if the input is demeaned before being 
               passed to the U-net
- shift_output: boolean indicating if the output of the Unet is demeaned
- gnorm: boolean indicating if a groupnorm should be used in the mean-bypass 
         layer
- nspatial: integer giving the number of spatial dimensions; images are assumed 
            to be square.
- dropout_p: float giving the dropout probability
- num_residual: integer giving the the number of residual blocks in the center of 
                the Unet
- noised_channels: integer giving the number of channels that are being noised
- context_channels: integer giving the number of context channels (not noised)
- channels: array of integers containing the number of channels for each layer of 
            the Unet during downsampling/upsampling
- embed_dim: integer of the time-embedding dimension
- scale: float giving the scale of the time-embedding layers
- periodic: whether or not spatial convolutions should respect periodicity
- proj_kernelsize: integer giving the kernel size in projection layers
- outer_kernelsize: integer giving the kernel size in the outermost down/upsample 
                    layers
- middle_kernelsize: integer giving the kernel size in the middle down/upsample 
                     layers
- inner_kernelsize: integer giving the kernel size in the innermost down/upsample 
                    layers
"""
function NoiseConditionalScoreNetwork(; context=false,
                                      mean_bypass=false, 
                                      scale_mean_bypass=false,
                                      shift_input=false,
                                      shift_output=false,
                                      gnorm=false,
                                      nspatial=2,
                                      dropout_p=0.0f0,
                                      num_residual=8,
                                      noised_channels=1,
                                      context_channels=0,
                                      channels=[32, 64, 128, 256],
                                      embed_dim=256,
                                      scale=30.0f0,
                                      periodic=false,
                                      proj_kernelsize=3,
                                      outer_kernelsize=3,
                                      middle_kernelsize=3,
                                      inner_kernelsize=3)
    if scale_mean_bypass & !mean_bypass
        @error("Attempting to scale the mean bypass term without adding in a mean bypass connection.")
    end
    if gnorm & !mean_bypass
        @error("Attempting to gnorm without adding in a mean bypass connection.")
    end
    if context & (context_channels == 0)
        @error("Attempting to use context-aware network without context input.")
    end
    if !context & (context_channels > 0)
        @error("Attempting to use context-unaware network with context input.")
    end
    
    inchannels = noised_channels+context_channels
    outchannels = noised_channels
    # Mean processing as indicated by boolean mean_bypass
    if mean_bypass
        if gnorm
            mean_bypass_layers = (
                mean_skip_1 = Conv((1, 1), inchannels => embed_dim),
                mean_skip_2 = Conv((1, 1), embed_dim => embed_dim),
                mean_skip_3 = Conv((1, 1), embed_dim => outchannels),
                mean_gnorm_1 = GroupNorm(embed_dim, 32, swish),
                mean_gnorm_2 = GroupNorm(embed_dim, 32, swish),
                mean_dense_1 = Dense(embed_dim, embed_dim),
                mean_dense_2 = Dense(embed_dim, embed_dim),
            )
        else
            mean_bypass_layers = (
                mean_skip_1 = Conv((1, 1), inchannels => embed_dim),
                mean_skip_2 = Conv((1, 1), embed_dim => embed_dim),
                mean_skip_3 = Conv((1, 1), embed_dim => outchannels),
                mean_dense_1 = Dense(embed_dim, embed_dim),
                mean_dense_2 = Dense(embed_dim, embed_dim),
            )
        end
    else
        mean_bypass_layers = ()
    end

    # Lifting/Projection layers depend on periodicity of data
    if periodic
        conv1 = CircularConv(3, nspatial, inchannels => channels[1] ; stride=1)
        tconv1 = CircularConv(proj_kernelsize, nspatial, channels[1] + channels[1] => outchannels; stride=1)
    else
        conv1=Conv((3, 3), inchannels => channels[1], stride=1, pad=SamePad())
        tconv1=Conv((proj_kernelsize, proj_kernelsize), channels[1] + channels[1] => outchannels, stride=1, pad=SamePad())
    end
    
    layers = (gaussfourierproj=GaussianFourierProjection(embed_dim, scale),
              linear=Dense(embed_dim, embed_dim, swish),
              
              # Lifting
              conv1=conv1,
              dense1=Dense(embed_dim, channels[1]),
              gnorm1=GroupNorm(channels[1], 4, swish),
              
              # Encoding
              conv2=Downsampling(channels[1] => channels[2], nspatial, kernel_size=3, periodic=periodic),
              dense2=Dense(embed_dim, channels[2]),
              gnorm2=GroupNorm(channels[2], 32, swish),
              
              conv3=Downsampling(channels[2] => channels[3], nspatial, kernel_size=3, periodic=periodic),
              dense3=Dense(embed_dim, channels[3]),
              gnorm3=GroupNorm(channels[3], 32, swish),
              
              conv4=Downsampling(channels[3] => channels[4], nspatial, kernel_size=3, periodic=periodic),
              dense4=Dense(embed_dim, channels[4]),
              
              # Residual Blocks
              resnet_blocks = 
              [ResnetBlockNCSN(channels[end], nspatial, embed_dim; p = dropout_p, periodic=periodic) for _ in range(1, length=num_residual)],
              
              # Decoding
              gnorm4=GroupNorm(channels[4], 32, swish),
              tconv4=Upsampling(channels[4] => channels[3], nspatial, kernel_size=inner_kernelsize, periodic=periodic),
              denset4=Dense(embed_dim, channels[3]),
              tgnorm4=GroupNorm(channels[3], 32, swish),
              
              tconv3=Upsampling(channels[3]+channels[3] => channels[2], nspatial, kernel_size=middle_kernelsize, periodic=periodic),
              denset3=Dense(embed_dim, channels[2]),
              tgnorm3=GroupNorm(channels[2], 32, swish),
              
              tconv2=Upsampling(channels[2]+channels[2] => channels[1], nspatial, kernel_size=outer_kernelsize, periodic=periodic),
              denset2=Dense(embed_dim, channels[1]),
              tgnorm2=GroupNorm(channels[1], 32, swish),
              
              # Projection
              tconv1=tconv1,
              mean_bypass_layers...
              )
    
    return NoiseConditionalScoreNetwork(layers, context, mean_bypass, scale_mean_bypass, shift_input, shift_output, gnorm)
end

@functor NoiseConditionalScoreNetwork

"""
    (net::NoiseConditionalScoreNetwork)(x, c, t)

Evaluates the neural network of the NoiseConditionalScoreNetwork
model on (x,c,t), where `x` is the tensor of noised input,
`c` is the tensor of contextual input, and `t` is a tensor of times.
"""
function (net::NoiseConditionalScoreNetwork)(x, c, t)
    # Get size of spatial dimensions
    nspatial = ndims(x) - 2

    # Embedding
    embed = net.layers.gaussfourierproj(t)
    embed = net.layers.linear(embed)

    # Encoder
    if net.shift_input
        h1 = x .- mean(x, dims=(1:nspatial)) # remove mean of noised variables before input
    else
        h1 = x
    end
    h1 = concatenate_channels(Val(net.context), h1, c)
    h1 = net.layers.conv1(h1)
    h1 = h1 .+ expand_dims(net.layers.dense1(embed), nspatial)
    h1 = net.layers.gnorm1(h1)
    h2 = net.layers.conv2(h1)
    h2 = h2 .+ expand_dims(net.layers.dense2(embed), nspatial)
    h2 = net.layers.gnorm2(h2)
    h3 = net.layers.conv3(h2)
    h3 = h3 .+ expand_dims(net.layers.dense3(embed), nspatial)
    h3 = net.layers.gnorm3(h3)
    h4 = net.layers.conv4(h3)
    h4 = h4 .+ expand_dims(net.layers.dense4(embed), nspatial)

    # middle
    h = h4
    for block in net.layers.resnet_blocks
        h = block(h, embed)
    end

    # Decoder
    h = net.layers.gnorm4(h)
    h = net.layers.tconv4(h)
    h = h .+ expand_dims(net.layers.denset4(embed), nspatial)
    h = net.layers.tgnorm4(h)
    h = net.layers.tconv3(cat(h, h3; dims=nspatial+1))
    h = h .+ expand_dims(net.layers.denset3(embed), nspatial)
    h = net.layers.tgnorm3(h)
    h = net.layers.tconv2(cat(h, h2, dims=nspatial+1))
    h = h .+ expand_dims(net.layers.denset2(embed), nspatial)
    h = net.layers.tgnorm2(h)
    h = net.layers.tconv1(cat(h, h1, dims=nspatial+1))
    if net.shift_output
        h = h .- mean(h, dims=(1:nspatial)) # remove mean after output
    end

    # Mean processing of noised variable channels
    if net.mean_bypass
        hm = net.layers.mean_skip_1(mean(concatenate_channels(Val(net.context), x, c), dims=(1:nspatial)))
        hm = hm .+ expand_dims(net.layers.mean_dense_1(embed), nspatial)
        if net.gnorm
            hm = net.layers.mean_gnorm_1(hm)
        end
        hm = net.layers.mean_skip_2(hm)
        hm = hm .+ expand_dims(net.layers.mean_dense_2(embed), nspatial)
        if net.gnorm
            hm = net.layers.mean_gnorm_2(hm)
        end
        hm = net.layers.mean_skip_3(hm)
        if net.scale_mean_bypass
            scale = convert(eltype(x), sqrt(prod(size(x)[1:nspatial])))
            hm = hm ./ scale
        end
        # Add back in noised channel mean to noised channel spatial variatons
        return h .+ hm
    else
        return h
    end
end

"""
    NoiseConditionalScoreNetwork3D

The struct containing the parameters and layers
of the Noise Conditional Score Network architecture,
with the option to include a mean-bypass layer.

# References
Unet: https://arxiv.org/abs/1505.04597
"""
struct NoiseConditionalScoreNetwork3D
    layers::NamedTuple
    context::Bool
end

"""
    NoiseConditionalScoreNetwork3D(; context=false,
                                 nspatial=2,
                                 dropout_p=0.0f0,
                                 num_residual=8,
                                 noised_channels=1,
                                 context_channels=0,
                                 channels=[32, 64, 128, 256],
                                 embed_dim=256,
                                 scale=30.0f0,
                                 periodic=false,
                                 proj_kernelsize=3,
                                 outer_kernelsize=3,
                                 middle_kernelsize=3,
                                 inner_kernelsize=3)

Returns a NoiseConditionalScoreNetwork3D, given
- nspatial: integer giving the number of spatial dimensions; images are assumed 
            to be square.
- dropout_p: float giving the dropout probability
- num_residual: integer giving the the number of residual blocks in the center of 
                the Unet
- noised_channels: integer giving the number of channels that are being noised
- context_channels: integer giving the number of context channels (not noised)
- channels: array of integers containing the number of channels for each layer of 
            the Unet during downsampling/upsampling
- embed_dim: integer of the time-embedding dimension
- scale: float giving the scale of the time-embedding layers
- periodic: whether or not spatial convolutions should respect periodicity
- proj_kernelsize: integer giving the kernel size in projection layers
- outer_kernelsize: integer giving the kernel size in the outermost down/upsample 
                    layers
- middle_kernelsize: integer giving the kernel size in the middle down/upsample 
                     layers
- inner_kernelsize: integer giving the kernel size in the innermost down/upsample 
                    layers
"""
function NoiseConditionalScoreNetwork3D(; context=false, 
                                      nspatial=3,
                                      dropout_p=0.0f0,
                                      num_residual=8,
                                      noised_channels=1,
                                      context_channels=0,
                                      channels=[32, 64, 128, 256],
                                      embed_dim=256,
                                      scale=30.0f0,
                                      periodic=false,
                                      proj_kernelsize=3,
                                      outer_kernelsize=3,
                                      middle_kernelsize=3,
                                      inner_kernelsize=3)

    @assert context && context_channels > 0 || !context && context_channels == 0

    inchannels = noised_channels + context_channels
    outchannels = noised_channels

    # Mean bypass
    mean_bypass_layers = (
        mean_skip_1 = Conv(Tuple(1 for _ in 1:nspatial), inchannels => embed_dim),
        mean_skip_2 = Conv(Tuple(1 for _ in 1:nspatial), embed_dim => embed_dim),
        mean_skip_3 = Conv(Tuple(1 for _ in 1:nspatial), embed_dim => outchannels),
        mean_gnorm_1 = GroupNorm(embed_dim, 32, swish),
        mean_gnorm_2 = GroupNorm(embed_dim, 32, swish),
        mean_dense_1 = Dense(embed_dim, embed_dim),
        mean_dense_2 = Dense(embed_dim, embed_dim),
    )

    # Lifting/Projection layers depend on periodicity of data
    if periodic
        conv1 = CircularConv(proj_kernelsize, nspatial, inchannels => channels[1] ; stride=1)
        tconv1 = CircularConv(proj_kernelsize, nspatial, channels[1] + channels[1] => outchannels; stride=1)
    else
        conv1=Conv(Tuple(proj_kernelsize for _ in 1:nspatial), inchannels => channels[1], stride=1, pad=SamePad())
        tconv1=Conv(Tuple(proj_kernelsize for _ in 1:nspatial), channels[1] + channels[1] => outchannels, stride=1, pad=SamePad())
    end
    
    # Bulk of the network
    layers = (
        gaussfourierproj=GaussianFourierProjection(embed_dim, scale),
        linear=Dense(embed_dim, embed_dim, swish),
        
        # Lifting
        conv1=conv1,
        dense1=Dense(embed_dim, channels[1]),
        gnorm1=GroupNorm(channels[1], 4, swish),
        
        # Encoding
        conv2=Downsampling(channels[1] => channels[2], nspatial, kernel_size=3, periodic=periodic),
        dense2=Dense(embed_dim, channels[2]),
        gnorm2=GroupNorm(channels[2], 32, swish),
        
        conv3=Downsampling(channels[2] => channels[3], nspatial, kernel_size=3, periodic=periodic),
        dense3=Dense(embed_dim, channels[3]),
        gnorm3=GroupNorm(channels[3], 32, swish),
        
        conv4=Downsampling(channels[3] => channels[4], nspatial, kernel_size=3, periodic=periodic),
        dense4=Dense(embed_dim, channels[4]),
        
        # Residual Blocks
        resnet_blocks = 
        [ResnetBlockNCSN(channels[end], nspatial, embed_dim; p = dropout_p, periodic=periodic) for _ in range(1, length=num_residual)],
        
        # Decoding
        gnorm4=GroupNorm(channels[4], 32, swish),
        tconv4=Upsampling(channels[4] => channels[3], nspatial, kernel_size=inner_kernelsize, periodic=periodic),
        denset4=Dense(embed_dim, channels[3]),
        tgnorm4=GroupNorm(channels[3], 32, swish),
        
        tconv3=Upsampling(channels[3]+channels[3] => channels[2], nspatial, kernel_size=middle_kernelsize, periodic=periodic),
        denset3=Dense(embed_dim, channels[2]),
        tgnorm3=GroupNorm(channels[2], 32, swish),
        
        tconv2=Upsampling(channels[2]+channels[2] => channels[1], nspatial, kernel_size=outer_kernelsize, periodic=periodic),
        denset2=Dense(embed_dim, channels[1]),
        tgnorm2=GroupNorm(channels[1], 32, swish),
        
        # Projection
        tconv1=tconv1,
        mean_bypass_layers...
    )
    
    return NoiseConditionalScoreNetwork3D(layers, context)
end

@functor NoiseConditionalScoreNetwork3D

"""
    (net::NoiseConditionalScoreNetwork3D)(x, c, t)

Evaluates the neural network of the NoiseConditionalScoreNetwork
model on (x,c,t), where `x` is the tensor of noised input,
`c` is the tensor of contextual input, and `t` is a tensor of times.
"""
function (net::NoiseConditionalScoreNetwork3D)(x, c, t)
    # Get size of spatial dimensions
    nspatial = ndims(x) - 2

    # Embedding
    embed = net.layers.gaussfourierproj(t)
    embed = net.layers.linear(embed)

    # remove mean of noised variables before input
    h1 = x .- mean(x, dims=(1:nspatial)) 

    # Encoder
    h1 = concatenate_channels(Val(net.context), h1, c, nspatial)
    h1 = net.layers.conv1(h1)
    h1 = h1 .+ expand_dims(net.layers.dense1(embed), nspatial)
    h1 = net.layers.gnorm1(h1)
    h2 = net.layers.conv2(h1)
    h2 = h2 .+ expand_dims(net.layers.dense2(embed), nspatial)
    h2 = net.layers.gnorm2(h2)
    h3 = net.layers.conv3(h2)
    h3 = h3 .+ expand_dims(net.layers.dense3(embed), nspatial)
    h3 = net.layers.gnorm3(h3)
    h4 = net.layers.conv4(h3)
    h4 = h4 .+ expand_dims(net.layers.dense4(embed), nspatial)

    # middle
    h = h4
    for block in net.layers.resnet_blocks
        h = block(h, embed)
    end

    # Decoder
    h = net.layers.gnorm4(h)
    h = net.layers.tconv4(h)
    h = h .+ expand_dims(net.layers.denset4(embed), nspatial)
    h = net.layers.tgnorm4(h)
    h = net.layers.tconv3(cat(h, h3; dims=nspatial+1))
    h = h .+ expand_dims(net.layers.denset3(embed), nspatial)
    h = net.layers.tgnorm3(h)
    h = net.layers.tconv2(cat(h, h2, dims=nspatial+1))
    h = h .+ expand_dims(net.layers.denset2(embed), nspatial)
    h = net.layers.tgnorm2(h)
    h = net.layers.tconv1(cat(h, h1, dims=nspatial+1))

    # remove mean after output
    h = h .- mean(h, dims=(1:nspatial)) 

    # Mean processing of noised variable channels
    hm = net.layers.mean_skip_1(mean(concatenate_channels(Val(net.context), x, c, nspatial), dims=(1:nspatial)))
    hm = hm .+ expand_dims(net.layers.mean_dense_1(embed), nspatial)
    hm = net.layers.mean_gnorm_1(hm)
    hm = net.layers.mean_skip_2(hm)
    hm = hm .+ expand_dims(net.layers.mean_dense_2(embed), nspatial)
    hm = net.layers.mean_gnorm_2(hm)
    hm = net.layers.mean_skip_3(hm)
    scale = convert(eltype(x), sqrt(prod(size(x)[1:nspatial])))
    hm = hm ./ scale

    # Add back in noised channel mean to noised channel spatial variatons
    return h .+ hm
end

"""
    concatenate_channels(context::Val{true}, x, c, nspatial=2)

Concatenates the context channels `c` with the noised data
channels `x` if `context` is true.
"""
function concatenate_channels(context::Val{true}, x, c, nspatial=2)
    return cat(x, c, dims = nspatial+1)
end

"""
    concatenate_channels(context::Val{false}, x, c)

Returns `x` if `context` is false.
"""
function concatenate_channels(context::Val{false}, x, c, nspatial=2)
    return x
end

"""
    GaussianFourierProjection{FT}

Concrete type used in the Gaussian Fourier Projection method
of embedding a continuous time variable.
"""
struct GaussianFourierProjection{FT}
    "Array used to scale and embed the time variable."
    W::AbstractArray{FT}
end

"""
    GaussianFourierProjection(embed_dim::Int, scale::FT) where {FT}

Outer constructor for the GaussianFourierProjection.

W is not trainable and is sampled once upon construction.
"""
function GaussianFourierProjection(embed_dim::Int, scale::FT) where {FT}
    W = randn(FT, embed_dim ÷ 2) .* scale
    return GaussianFourierProjection(W)
end

@functor GaussianFourierProjection

"""
    (gfp::GaussianFourierProjection{FT})(t) where {FT}


Embeds a continuous time `t`  into a periodic domain
using a random vector of Gaussian noise `gfp.W`.

# References
https://arxiv.org/abs/2006.10739
"""
function (gfp::GaussianFourierProjection{FT})(t) where {FT}
    t_proj = t' .* gfp.W .* FT(2π)
    return [sin.(t_proj); cos.(t_proj)]
end

"""
    Flux.params(::GaussianFourierProjection)

Returns the trainable parameters of the GaussianFourierProjection,
which are `nothing`.
"""
Flux.params(::GaussianFourierProjection) = nothing

"""
    ClimaGen.DenoisingDiffusionNetwork

# Notes
Images stored in (spatial..., channels, batch) order. 

This currently does not support periodic boundary conditions.

# References
Ho, Jain, Abbeel: Denoising diffusion probabilistic models.
"""
struct DenoisingDiffusionNetwork
    layers::NamedTuple
end

"""
User Facing API for DenoisingDiffusionNetwork architecture.

N.B. this recreates a channels inflation of inchannels->nf->nf*1->nf*2->nf*2->nf*2, which is expressed
in the Song et al. code by inchannels, nf = 128, ch multiplier = [1,2,2,2].

Their embedding dimension is nf*4.

Here for now we set nf = 64, ch_multiplier = [1,2,2,4]...
"""
function DenoisingDiffusionNetwork(; kernel_size=3, nspatial=2, inchannels=1, channels=[64, 128, 128, 256], nembed=256, scale=30.0f0, p=0.1f0, σ=Flux.swish)
    conv_kernel = Tuple(kernel_size for _ in 1:nspatial)
    pad = SamePad()
    
    return DenoisingDiffusionNetwork(
        (
        gaussfourierproj=GaussianFourierProjection(nembed, scale),
        embedding_network=Chain(Dense(nembed, nembed, σ), Dense(nembed, nembed)),

        lift = Conv(conv_kernel, inchannels => channels[1], pad=pad),

        # Encoding
        encode_block1a=ResnetBlockDDN(channels[1] => channels[1], nspatial, nembed, p, σ),
        encode_block1b=ResnetBlockDDN(channels[1] => channels[1], nspatial, nembed, p, σ),
        downsample1=Downsampling(channels[1]=>channels[1], nspatial),
            encode_block2a=ResnetBlockDDN(channels[1] => channels[2], nspatial, nembed, p, σ),
            encode_block2b=ResnetBlockDDN(channels[2] => channels[2], nspatial, nembed, p, σ),
            downsample2=Downsampling(channels[2]=> channels[2], nspatial),
                encode_block3a=ResnetBlockDDN(channels[2] => channels[3], nspatial, nembed, p, σ),
                encode_block3b=ResnetBlockDDN(channels[3] => channels[3], nspatial, nembed, p, σ),
                downsample3=Downsampling(channels[3]=>channels[3], nspatial),
                    encode_block4a=ResnetBlockDDN(channels[3] => channels[4], nspatial, nembed, p, σ),
                    encode_block4b=ResnetBlockDDN(channels[4] => channels[4], nspatial, nembed, p, σ),

                    # Transformations in middle
                    middle_transform1=ResnetBlockDDN(channels[4] => channels[4], nspatial, nembed, p, σ),
                    middle_transform2=ResnetBlockDDN(channels[4] => channels[4], nspatial, nembed, p, σ),

                    # Decoding
                    decode_block4a=ResnetBlockDDN(channels[4] + channels[4] => channels[4], nspatial, nembed, p, σ),
                    decode_block4b=ResnetBlockDDN(channels[4] + channels[4] => channels[4], nspatial, nembed, p, σ),
                    decode_block4c=ResnetBlockDDN(channels[4] + channels[3] => channels[4], nspatial, nembed, p, σ),
                    upsample4=Upsampling(channels[4]=>channels[4], nspatial),
                decode_block3a=ResnetBlockDDN(channels[4] + channels[3] => channels[3], nspatial, nembed, p, σ),
                decode_block3b=ResnetBlockDDN(channels[3] + channels[3] => channels[3], nspatial, nembed, p, σ),
                decode_block3c=ResnetBlockDDN(channels[3] + channels[2] => channels[3], nspatial, nembed, p, σ),
                upsample3=Upsampling(channels[3]=> channels[3], nspatial),
            decode_block2a=ResnetBlockDDN(channels[3] + channels[2] => channels[2], nspatial, nembed, p, σ),
            decode_block2b=ResnetBlockDDN(channels[2] + channels[2] => channels[2], nspatial, nembed, p, σ),
            decode_block2c=ResnetBlockDDN(channels[2] + channels[1] => channels[2], nspatial, nembed, p, σ),
            upsample2=Upsampling(channels[2]=>channels[2], nspatial),
        decode_block1a=ResnetBlockDDN(channels[2] + channels[1] => channels[1], nspatial, nembed, p, σ),
        decode_block1b=ResnetBlockDDN(channels[1] + channels[1] => channels[1], nspatial, nembed, p, σ),
        decode_block1c=ResnetBlockDDN(channels[1] + channels[1] => channels[1], nspatial, nembed, p, σ),

        # Transformations at end
        project=Chain(
            GroupNorm(channels[1], min(channels[1] ÷ 4, 32), σ),
            Conv(conv_kernel, channels[1] => inchannels, pad=pad)
        ),
    )
    )
end

@functor DenoisingDiffusionNetwork

function (net::DenoisingDiffusionNetwork)(x, t)
    # Embedding
    tembed = net.layers.gaussfourierproj(t)
    tembed = net.layers.embedding_network(tembed)

    # Lifting
    cx = net.layers.lift(x)

    # Encoder
    h1a = net.layers.encode_block1a(cx, tembed)
    h1b = net.layers.encode_block1b(h1a, tembed)
    h1c = net.layers.downsample1(h1b)

    h2a = net.layers.encode_block2a(h1c, tembed)
    h2b = net.layers.encode_block2b(h2a, tembed)
    h2c = net.layers.downsample2(h2b)

    h3a = net.layers.encode_block3a(h2c, tembed)
    h3b = net.layers.encode_block3b(h3a, tembed)
    h3c = net.layers.downsample3(h3b)

    h4a = net.layers.encode_block4a(h3c, tembed)
    h4b = net.layers.encode_block4b(h4a, tembed)

    # Middle transformations
    h = net.layers.middle_transform1(h4a, tembed)
    h = net.layers.middle_transform2(h, tembed)

    # Decoder
    h = net.layers.decode_block4a(cat(h, h4b; dims=3), tembed)
    h = net.layers.decode_block4b(cat(h, h4a; dims=3), tembed)
    h = net.layers.decode_block4c(cat(h, h3c; dims=3), tembed)
    h = net.layers.upsample4(h)

    h = net.layers.decode_block3a(cat(h, h3b; dims=3), tembed)
    h = net.layers.decode_block3b(cat(h, h3a; dims=3), tembed)
    h = net.layers.decode_block3c(cat(h, h2c; dims=3), tembed)
    h = net.layers.upsample3(h)

    h = net.layers.decode_block2a(cat(h, h2b; dims=3), tembed)
    h = net.layers.decode_block2b(cat(h, h2a; dims=3), tembed)
    h = net.layers.decode_block2c(cat(h, h1c; dims=3), tembed)
    h = net.layers.upsample2(h)

    h = net.layers.decode_block1a(cat(h, h1b; dims=3), tembed)
    h = net.layers.decode_block1b(cat(h, h1a; dims=3), tembed)
    h = net.layers.decode_block1c(cat(h, cx; dims=3), tembed)

    # End transformations
    return net.layers.project(h)
end

"""
    CliMAgen.ResnetBlockDDN

The ResNet block structure for the Denoising Diffusion 
Network with GroupNorm and GaussianFourierProjection.

This currently does not support periodic boundary conditions.

References:
https://arxiv.org/abs/1505.04597
https://arxiv.org/abs/1712.09763
"""
struct ResnetBlockDDN
    norm1
    conv1
    norm2
    conv2
    dense
    dropout
    bypass
end

function ResnetBlockDDN(channels::Pair, nspatial::Int, nembed::Int, p=0.1f0, σ=Flux.swish)
    # channels needs to be larger than 4
    @assert channels.first ÷ 4 > 0
    @assert channels.second ÷ 4 > 0

    # Require same input and output spatial size
    pad = SamePad()

    return ResnetBlockDDN(
        GroupNorm(channels.first, min(channels.first ÷ 4, 32), σ),
        Conv(Tuple(3 for _ in 1:nspatial), channels, pad=pad),
        GroupNorm(channels.second, min(channels.second ÷ 4, 32), σ),
        Conv(Tuple(3 for _ in 1:nspatial), channels.second => channels.second, pad=pad),
        Dense(nembed => channels.second, σ),
        Dropout(p),
        Conv(Tuple(1 for _ in 1:nspatial), channels, pad=pad),
    )
end

@functor ResnetBlockDDN

function (net::ResnetBlockDDN)(x, tembed)
    # add on temporal embeddings to condition on time
    h = net.norm1(x)
    h = net.conv1(h) .+ expand_dims(net.dense(tembed), 2)

    # dropout is needed for low complexity datasets to
    # avoid overfitting
    h = net.norm2(h)
    h = net.dropout(h)
    h = net.conv2(h)

    return h .+ net.bypass(x)
end

"""
    CliMAgen.ResnetBlockNCSN

Struct holding the layers of the ResNet block 
used in the NoiseConditionalScoreNetwork model,
using GroupNorm and GaussianFourierProjection.

References:
https://arxiv.org/abs/1505.04597
https://arxiv.org/abs/1712.09763
"""
struct ResnetBlockNCSN
    "The group-normalization layer for the input"
    norm1
    "The first convolutional layer"
    conv1
    "The second group-normalization layer"
    norm2
    "The second convolutional layer"
    conv2
    "A dense layer for handling the time variable"
    dense
    "A dropout layer"
    dropout
end
"""
     ResnetBlockNCSN(channels::Int, nspatial::Int, nembed::Int; p=0.1f0, σ=Flux.swish, periodic=false)

Constructor for the ResnetBlockNCSN, which preserves the `channel` number and image
size of the input.

Here, `nspatial` is the number of spatial dimensions, `nembed` is the embedding
size used in the GaussianFourierProjection,
`p` is the dropout probability, `σ` is the nonlinearity used in the group
norms and the dense layer, and `periodic` is a boolean indicating if spatial
convolutions respect periodicity at the boundaries.
"""
function ResnetBlockNCSN(channels::Int, nspatial::Int, nembed::Int; p=0.1f0, σ=Flux.swish, periodic=false)
    # channels needs to be larger than 4 for group norms
    @assert channels ÷ 4 > 0
    if periodic
        conv1 = CircularConv(3, nspatial, channels => channels)
        conv2 = CircularConv(3, nspatial, channels => channels)
    else
        conv_kernel = Tuple(3 for _ in 1:nspatial)
        conv1 = Conv(conv_kernel, channels => channels, pad = SamePad())
        conv2 =  Conv(conv_kernel, channels => channels, pad = SamePad())
    end

    return ResnetBlockNCSN(
        GroupNorm(channels, min(channels ÷ 4, 32), σ),
        conv1,
        GroupNorm(channels, min(channels ÷ 4, 32), σ),
        conv2,
        Dense(nembed => channels, σ),
        Dropout(p),
    )
end

@functor ResnetBlockNCSN

"""
   (net::ResnetBlockNCSN)(x, tembed)

Applies the ResnetBlockNCSN to (x,tembed).
"""
function (net::ResnetBlockNCSN)(x, tembed)
    # number of spatial dimensions
    nspatial = ndims(x) - 2

    # add on temporal embeddings to condition on time
    h = net.norm1(x)
    h = net.conv1(h) .+ expand_dims(net.dense(tembed), nspatial)

    # dropout is needed for low complexity datasets to
    # avoid overfitting
    h = net.norm2(h)
    h = net.dropout(h)
    h = net.conv2(h)

    return h .+ x
end

"""
    CliMAGen.AttentionBlock

Softmax attention block with group normalization and bypass connection.

This currently does not support periodic boundary conditions.
References:
https://arxiv.org/abs/1712.09763
"""
struct AttentionBlock
    normalize
    queries
    keys
    values
    project
    bypass
end

function AttentionBlock(channels, nspatial, ngroups=32)
    @assert channels.first % ngroups == 0
    return AttentionBlock(
        GroupNorm(channels.first, ngroups),
        Conv(Tuple(1 for _ in 1:nspatial), channels),
        Conv(Tuple(1 for _ in 1:nspatial), channels),
        Conv(Tuple(1 for _ in 1:nspatial), channels),
        Conv(Tuple(1 for _ in 1:nspatial), channels),
        Conv(Tuple(1 for _ in 1:nspatial), channels),
    )
end

@functor AttentionBlock

function (net::AttentionBlock)(x::AbstractArray{FT}) where {FT}
    nspatial = ndims(x) - 2
    nchannels = size(x, nspatial + 1)

    # use queries, keys, values framework to compute attention
    h = net.normalize(x)
    q = net.queries(h)
    k = net.keys(h)
    v = net.values(h)

    w = compute_attention_weights(k, q, Val(nspatial))

    # normalize since attention weights need to sum to one
    w = w ./ FT(sqrt(nchannels))
    w = Flux.softmax(w; dims=1:nspatial)

    h = apply_attention_weights(w, v, Val(nspatial))
    h = net.project(h)

    return h .+ net.bypass(x)
end

"""
    CliMAgen.compute_attention_weights(keys, queries, Val{nspatialdims})

Helper function that returns the softmax attention weights 
for given keys 'k' and queries 'q' by computing the dot product 
of the keys and queries across channels.
"""
compute_attention_weights(k, q, ::Val{2}) =
    @tullio w[h1, w1, h2, w2, b] := k[h1, w1, c, b] * q[h2, w2, c, b]

"""
    CliMAgen.apply_attention_weights(weights, values, Val{nspatialdims})

Helper function that applies attention weights for given weights 
'w' and values 'v' by taking the weighted sum of the values at 
each location and batch.
"""
apply_attention_weights(w, v, ::Val{2}) =
    @tullio v_weighted[h2, w2, c, b] := w[h1, w1, h2, w2, b] * v[h1, w1, c, b]

"""
    CliMAgen.Downsampling(channels::Pair,
                          nspatial::Int;
                          factor::Int=2,
                          kernel_size::Int=3,
                          periodic=false)

Creates a downsampling layer using convolutional kernels.

Here, 
- `channels = inchannels => outchannels` is the pair of incoming and outgoing channels,
- `nspatial` is the number of spatial dimensions of the image,
- `factor` indicates the downsampling factor, 
- `kernel_size` is in the kernel size, and
- `periodic` is a boolean indicating if the spatial convolutions should respect periodicity.
"""
function Downsampling(channels::Pair, nspatial::Int; factor::Int=2, kernel_size::Int=3, periodic=false)
    if periodic
        return CircularConv(kernel_size, nspatial, channels; stride=factor)
    else
        conv_kernel = Tuple(kernel_size for _ in 1:nspatial)
        return Conv(conv_kernel, channels; stride=factor, pad = SamePad())
    end
    return CircularConv(conv_kernel, channels, stride=factor, periodic=periodic)
end

"""
    CliMAgen.Upsampling(channels::Pair, nspatial::Int; factor::Int=2, kernel_size::Int=3, periodic=false)

Creates an upsampling layer using nearest-neighbor interpolation and 
convolutional kernels, so that checkerboard artifacts are avoided.

Here, 
- `channels = inchannels => outchannels` is the pair of incoming and outgoing channels,
- `nspatial` is the number of spatial dimensions of the image,
- `factor` indicates the downsampling factor, 
- `kernel_size` is in the kernel size, and
- `periodic` is a boolean indicating if the spatial convolutions should respect periodicity.

References:
https://distill.pub/2016/deconv-checkerboard/
"""
function Upsampling(channels::Pair, nspatial::Int; factor::Int=2, kernel_size::Int=3, periodic=false)
    if periodic
        conv = CircularConv(kernel_size, nspatial, channels)
    else
        conv_kernel = Tuple(kernel_size for _ in 1:nspatial)
        conv = Conv(conv_kernel, channels; pad = SamePad())
    end
    return Chain(
        Flux.Upsample(factor, :nearest),
        conv
    )
end

"""
    CliMAgen.CircularConv

Struct for holding required data for carrying out convolutions
 respecting periodicity at the boundaries.

- `pad`: a Tuple holding the number of elements to pad with on each of
the boundaries. 
- `conv`: the convolutional layer 
"""
struct CircularConv{C<:Conv}
    pad::Tuple
    conv::C
end

"""
    CircularConv(kernel_size::Int, nspatial::Int, channels::Pair;stride::Int=1)

Creates a convolutional layer that respects periodicity at the boundaries, given
- `kernel_size`: the size of the kernel, in pixels
- `nspatial`: the number of spatial dimensions,
- `channels`: a Pair indicating the number of input and output channels
- `stride`: the stride to use for the convolution.
"""
function CircularConv(kernel_size::Int, nspatial::Int, channels::Pair;stride::Int=1)
    conv_kernel = Tuple(kernel_size for _ in 1:nspatial)
    pad = Tuple(div(kernel_size,2) for _ in 1:2*nspatial)
    conv = Conv(conv_kernel, channels; stride=stride, pad=0)
    return CircularConv{typeof(conv)}(pad, conv)
end

@functor CircularConv

"""
    (layer::CircularConv)(x)

Carries on the spatial convolution respecting periodicity at the boundaries.
"""
function (layer::CircularConv)(x)
    layer.conv(NNlib.pad_circular(x, layer.pad))
end

"""
    CliMAgen.ControlNet

The struct containing the parameters and layers of the ControlNet architecture.
"""
struct ControlNet{N}
    net::N
    trainable::Bool  # whether the network is trainable
end

"""
    ControlNet(net::N; trainable::Bool=false)

Creates a ControlNet with the given neural network `net` and whether it is trainable.
"""
function ControlNet(net::N; trainable::Bool=false) where N
    return ControlNet{N}(net, trainable)
end

@functor ControlNet

"""
    (net::ControlNet)(x)

Evaluates the neural network of the ControlNet model on `x`.
"""
function (c::ControlNet)(x)
    return c.net(x)
end

"""
    Flux.params(::ControlNet)
Returns the trainable parameters of the ControlNet).
"""
Flux.params(c::ControlNet) = c.trainable ? Flux.params(c.net) : nothing

"""
    CliMAgen.ControlledNoiseConditionalScoreNetwork

The struct containing the parameters and layers
of the Noise Conditional Score Network architecture,
with the option to include a mean-bypass layer.

# References
Unet: https://arxiv.org/abs/1505.04597
"""
struct ControlledNoiseConditionalScoreNetwork{N}
    "The layers of the network"
    layers::NamedTuple
    "A control network to condition the output of the U-net"
    control_net::N
    "A boolean indicating if a mean-bypass layer should be used"
    mean_bypass::Bool
    "A boolean indicating if the output of the mean-bypass layer should be scaled"
    scale_mean_bypass::Bool
    "A boolean indicating if the input is demeaned before being passed to the U-net"
    shift_input::Bool
    "A boolean indicating if the output of the Unet is demeaned"
    shift_output::Bool
    "A boolean indicating if a groupnorm should be used in the mean-bypass layer"
    gnorm::Bool
end

function ControlledNoiseConditionalScoreNetwork(; control_net,
                                      mean_bypass=false, 
                                      scale_mean_bypass=false,
                                      shift_input=false,
                                      shift_output=false,
                                      gnorm=false,
                                      nspatial=2,
                                      dropout_p=0.0f0,
                                      num_residual=8,
                                      noised_channels=1,
                                      channels=[32, 64, 128, 256],
                                      embed_dim=256,
                                      scale=30.0f0,
                                      periodic=false,
                                      proj_kernelsize=3,
                                      outer_kernelsize=3,
                                      middle_kernelsize=3,
                                      inner_kernelsize=3)
    if scale_mean_bypass & !mean_bypass
        @error("Attempting to scale the mean bypass term without adding in a mean bypass connection.")
    end
    if gnorm & !mean_bypass
        @error("Attempting to gnorm without adding in a mean bypass connection.")
    end
    inchannels = noised_channels
    outchannels = noised_channels
    
    # Mean processing as indicated by boolean mean_bypass
    if mean_bypass
        if gnorm
            mean_bypass_layers = (
                mean_skip_1 = Conv((1, 1), inchannels => embed_dim),
                mean_skip_2 = Conv((1, 1), embed_dim => embed_dim),
                mean_skip_3 = Conv((1, 1), embed_dim => outchannels),
                mean_gnorm_1 = GroupNorm(embed_dim, 32, swish),
                mean_gnorm_2 = GroupNorm(embed_dim, 32, swish),
                mean_dense_1 = Dense(embed_dim, embed_dim),
                mean_dense_2 = Dense(embed_dim, embed_dim),
            )
        else
            mean_bypass_layers = (
                mean_skip_1 = Conv((1, 1), inchannels => embed_dim),
                mean_skip_2 = Conv((1, 1), embed_dim => embed_dim),
                mean_skip_3 = Conv((1, 1), embed_dim => outchannels),
                mean_dense_1 = Dense(embed_dim, embed_dim),
                mean_dense_2 = Dense(embed_dim, embed_dim),
            )
        end
    else
        mean_bypass_layers = ()
    end

    # Lifting/Projection layers depend on periodicity of data
    if periodic
        conv1 = CircularConv(3, nspatial, inchannels => channels[1] ; stride=1)
        tconv1 = CircularConv(proj_kernelsize, nspatial, channels[1] + channels[1] => outchannels; stride=1)
    else
        conv1=Conv((3, 3), inchannels => channels[1], stride=1, pad=SamePad())
        tconv1=Conv((proj_kernelsize, proj_kernelsize), channels[1] + channels[1] => outchannels, stride=1, pad=SamePad())
    end
    
    layers = (gaussfourierproj=GaussianFourierProjection(embed_dim, scale),
              linear=Dense(embed_dim, embed_dim, swish),
              
              # Lifting
              conv1=conv1,
              dense1=Dense(embed_dim, channels[1]),
              control_dense1=Dense(embed_dim, channels[1]),
              gnorm1=GroupNorm(channels[1], 4, swish),
              
              # Encoding
              conv2=Downsampling(channels[1] => channels[2], nspatial, kernel_size=3, periodic=periodic),
              dense2=Dense(embed_dim, channels[2]),
              control_dense2=Dense(embed_dim, channels[2]),
              gnorm2=GroupNorm(channels[2], 32, swish),
              
              conv3=Downsampling(channels[2] => channels[3], nspatial, kernel_size=3, periodic=periodic),
              dense3=Dense(embed_dim, channels[3]),
              control_dense3=Dense(embed_dim, channels[3]),
              gnorm3=GroupNorm(channels[3], 32, swish),
              
              conv4=Downsampling(channels[3] => channels[4], nspatial, kernel_size=3, periodic=periodic),
              dense4=Dense(embed_dim, channels[4]),
              control_dense4=Dense(embed_dim, channels[4]),
              
              # Residual Blocks
              resnet_blocks = 
              [ResnetBlockNCSN(channels[end], nspatial, embed_dim; p = dropout_p, periodic=periodic) for _ in range(1, length=num_residual)],
              
              # Decoding
              gnorm4=GroupNorm(channels[4], 32, swish),
              tconv4=Upsampling(channels[4] => channels[3], nspatial, kernel_size=inner_kernelsize, periodic=periodic),
              denset4=Dense(embed_dim, channels[3]),
              control_denset4=Dense(embed_dim, channels[3]),
              tgnorm4=GroupNorm(channels[3], 32, swish),
              
              tconv3=Upsampling(channels[3]+channels[3] => channels[2], nspatial, kernel_size=middle_kernelsize, periodic=periodic),
              denset3=Dense(embed_dim, channels[2]),
              control_denset3=Dense(embed_dim, channels[2]),
              tgnorm3=GroupNorm(channels[2], 32, swish),
              
              tconv2=Upsampling(channels[2]+channels[2] => channels[1], nspatial, kernel_size=outer_kernelsize, periodic=periodic),
              denset2=Dense(embed_dim, channels[1]),
              control_denset2=Dense(embed_dim, channels[1]),
              tgnorm2=GroupNorm(channels[1], 32, swish),
              
              # Projection
              tconv1=tconv1,
              mean_bypass_layers...
              )

    return ControlledNoiseConditionalScoreNetwork(layers, control_net, mean_bypass, scale_mean_bypass, shift_input, shift_output, gnorm)
end

@functor ControlledNoiseConditionalScoreNetwork

"""
    (net::ControlledNoiseConditionalScoreNetwork)(x, c, t)

Evaluates the neural network of the NoiseConditionalScoreNetwork
model on (x,c,t), where `x` is the tensor of noised input,
`c` is the tensor of contextual input, and `t` is a tensor of times.
"""
function (net::ControlledNoiseConditionalScoreNetwork)(x, c, t)
    # Get size of spatial dimensions
    nspatial = ndims(x) - 2

    # Embeddings
    embed = net.layers.gaussfourierproj(t)
    embed = net.layers.linear(embed)
    control_embed = net.control_net(c)

    # Encoder
    if net.shift_input
        h1 = x .- mean(x, dims=(1:nspatial)) # remove mean of noised variables before input
    else
        h1 = x
    end
    h1 = net.layers.conv1(h1)
    h1 = h1 .+ expand_dims(net.layers.dense1(embed) .+ net.layers.control_dense1(control_embed), nspatial)
    h1 = net.layers.gnorm1(h1)
    h2 = net.layers.conv2(h1)
    h2 = h2 .+ expand_dims(net.layers.dense2(embed) .+ net.layers.control_dense2(control_embed), nspatial)
    h2 = net.layers.gnorm2(h2)
    h3 = net.layers.conv3(h2)
    h3 = h3 .+ expand_dims(net.layers.dense3(embed) .+ net.layers.control_dense3(control_embed), nspatial)
    h3 = net.layers.gnorm3(h3)
    h4 = net.layers.conv4(h3)
    h4 = h4 .+ expand_dims(net.layers.dense4(embed) .+ net.layers.control_dense4(control_embed), nspatial)

    # middle
    h = h4
    for block in net.layers.resnet_blocks
        h = block(h, embed .+ control_embed) # add in control embedding, can perhaps be done better.
    end

    # Decoder
    h = net.layers.gnorm4(h)
    h = net.layers.tconv4(h)
    h = h .+ expand_dims(net.layers.denset4(embed) .+ net.layers.control_denset4(control_embed), nspatial)
    h = net.layers.tgnorm4(h)
    h = net.layers.tconv3(cat(h, h3; dims=nspatial+1))
    h = h .+ expand_dims(net.layers.denset3(embed) .+ net.layers.control_denset3(control_embed), nspatial)
    h = net.layers.tgnorm3(h)
    h = net.layers.tconv2(cat(h, h2, dims=nspatial+1))
    h = h .+ expand_dims(net.layers.denset2(embed) .+ net.layers.control_denset2(control_embed), nspatial)
    h = net.layers.tgnorm2(h)
    h = net.layers.tconv1(cat(h, h1, dims=nspatial+1))
    if net.shift_output
        h = h .- mean(h, dims=(1:nspatial)) # remove mean after output
    end

    # Mean processing of noised variable channels
    if net.mean_bypass
        hm = net.layers.mean_skip_1(mean(x, dims=(1:nspatial)))
        hm = hm .+ expand_dims(net.layers.mean_dense_1(embed), nspatial)
        if net.gnorm
            hm = net.layers.mean_gnorm_1(hm)
        end
        hm = net.layers.mean_skip_2(hm)
        hm = hm .+ expand_dims(net.layers.mean_dense_2(embed), nspatial)
        if net.gnorm
            hm = net.layers.mean_gnorm_2(hm)
        end
        hm = net.layers.mean_skip_3(hm)
        if net.scale_mean_bypass
            scale = convert(eltype(x), sqrt(prod(size(x)[1:nspatial])))
            hm = hm ./ scale
        end
        # Add back in noised channel mean to noised channel spatial variatons
        return h .+ hm
    else
        return h
    end
end
