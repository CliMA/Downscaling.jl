struct PatchDiscriminator
    initial
    model
end

@functor PatchDiscriminator

function PatchDiscriminator(
    in_channels::Int=3,
    features::Any=[64, 128, 256, 512]
)
    layers = []
    channel = features[1]
    for index in range(2, length(features))
        if features[index] != last(features)
            push!(layers,
                PatchBlock(channel, features[index], stride=2)
            )
        else
            push!(layers,
                PatchBlock(channel, features[index], stride=1)
            )
        end
        channel = features[index]
    end
    push!(layers,
        Conv((4, 4), channel => 1; stride=1, pad=1)
    )
    return PatchDiscriminator(
        Chain(
            Conv((4, 4), in_channels => features[1]; stride=2, pad=1),
            x -> leakyrelu.(x, 0.2f0)
        ),
        Chain(layers...)
    )
end

function (net::PatchDiscriminator)(x)
    input = net.initial(x)
    return sigmoid.(net.model(input))
end


struct PatchBlock
    conv
end

@functor PatchBlock

function PatchBlock(
    in_channels::Int,
    out_channels::Int;
    stride::Int
)
    return PatchBlock(
        Chain(
            Conv((4, 4), in_channels => out_channels; stride=stride, pad=1),
            InstanceNorm(out_channels),
            x -> leakyrelu.(x, 0.2f0)
        )
    )
end

function (net::PatchBlock)(x)
    return net.conv(x)
end
