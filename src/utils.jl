
struct LeakyReLU{T}
    a::T
end

function (m::LeakyReLU)(x)
    leakyrelu.(x, m.a)
end

function conv(in_planes, out_planes; kernel_size=3, stride=1)
    Chain(
        Conv((kernel_size, kernel_size), in_planes => out_planes, stride=stride, 
             pad=(kernel_size-1) ÷ 2, ),
        LeakyReLU(0.1f0)
    )
end

function deconv(in_planes, out_planes)
    Chain(
        ConvTranspose((4, 4), in_planes => out_planes, stride=2, pad=1),
        LeakyReLU(0.1f0)
    )
end

function predict_flow(in_planes, out_dims)
    Conv((3, 3), in_planes => out_dims, stride=1, pad=1)
end

function crop_like(input, target)
    if size(input)[1:2] == size(target)[1:2]
        return input
    else
        return input[1:size(target, 1), 1:size(target,2), :, :]
    end
end