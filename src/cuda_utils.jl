
function condition_fill_cuda!(a, b, cond)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i in index:stride:length(a)
        if cond[i]
            a[i] = b
        end
    end
end

function condition_fill_cpu!(a, b, cond)
    a[cond] .= b
end

function condition_fill!(a::CuArray, b, cond)
    nthreads = 256
    nblocks = cld(prod(size(a)), nthreads)
    @cuda threads=nthreads blocks=nblocks condition_fill_cuda!(a, b, cond)
end

function condition_fill!(a::AbstractArray, b, cond)
    condition_fill_cpu!(a, b, cond)
end


function re_map_gpu!(y, x, my_idx)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i in index:stride:prod(size(y))
        y[i] = x[my_idx[i]]
    end
end

function re_map_cpu!(y, x, my_idx)
    y .= x[my_idx]
end

function re_map!(y::AbstractArray, x, my_idx)
    re_map_cpu!(y, x, my_idx)
end

function re_map!(y::CuArray, x, my_idx)
    nthreads = 256
    nblocks = cld(prod(size(y)), nthreads)
    @cuda threads=nthreads blocks=nblocks re_map_gpu!(y, x, my_idx)
end


function add_dim3_2_gpu!(y, x1, x2)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    
    offset = size(y)[1] * size(y)[2]
    for i in index:stride:prod(size(y))
        b = (i-1) ÷ offset
        b_i = (i-1) % offset
        d = b ÷ 2
        d_i = b % 2
        if d_i == 0 # x1
            y[i] += x1[d*offset + b_i + 1]
        else
            y[i] += x2[d*offset + b_i + 1]
        end
    end
end

function add_dim3_2_cpu!(y, x1, x2)
    y[:,:,1:1,:] += x1
    y[:,:,2:2,:] += x2
end

function add_dim3_2!(y::CuArray, x1, x2)
    nthreads = 256
    nblocks = cld(prod(size(y)), nthreads)
    @cuda threads=nthreads blocks=nblocks add_dim3_2_gpu!(y, x1, x2)
end

function add_dim3_2!(y::AbstractArray, x1, x2)
    add_dim3_2_cpu!(y, x1, x2)
end

function fill_onehot_cpu!(y, x)
    for w=axes(x, 1), h=axes(x, 2), n=axes(x, 4)
        i = x[w,h,1,n]
        @inbounds y[w,h,i,n] = 1
    end
    return
end

function fill_onehot_gpu!(y, x)
    
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    
    offset = size(y, 1) * size(y, 2)
    channel = size(y, 3)
    
    for i = index:stride:length(x)
        # @inbounds y[i] += x[i]
        extra = ((i-1) ÷ offset) * offset * channel
        offset_i = (i-1) % offset + 1
        @inbounds y[(x[i]-1) * offset + offset_i + extra] = 1
    end
    
    return
end

function onehot_dense(x::Array, limit::Int)
    y = zeros(Float32, size(x, 1), size(x, 2), limit, size(x, 4))
    fill!(y, 0)
    fill_onehot_cpu!(y, x)
    y
end

function onehot_dense(x::CuArray, limit::Int)
    y = CuArrays.zeros(Float32, size(x, 1), size(x, 2), limit, size(x, 4))
    fill!(y, 0)
    N = prod(size(x))
    numblocks = ceil(Int, N/256)
    # @show typeof(y) typeof(x)
    @cuda threads=256 blocks=numblocks fill_onehot_gpu!(y, x)
    y
end
