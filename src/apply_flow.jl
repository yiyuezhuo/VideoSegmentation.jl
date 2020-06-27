

struct BackTracker{T}
    i::Int
    j::Int
    w::T
    di::Vector{T}
    dj::Vector{T}
end

function get_left_right_weight(val)
    left, right = floor(Int, val), ceil(Int, val)
    if left == right
        return left, right, 1., 0.
    end
    left_weight = 1. - val + left
    right_weight = 1. - right + val
    return left, right, left_weight, right_weight
end

function get_left_right_weight_broadcast(val::AbstractArray{T, 3}, limit::Int) where T
    # left = floor.(Int, val)
    # right = ceil.(Int, val)
    # for CUDA.jl compatible
    left = Int.(floor.(val))
    right = Int.(ceil.(val))
    
    left_weight = T(1) .- val .+ left
    right_weight = T(1) .- right .+ val

    mask = left .== right

    # @show sum(mask)
    if sum(mask) > 0 # for CUDA.jl compatible
        # left_weight[mask] .= T(1)
        # right_weight[mask] .= T(0)
        condition_fill!(left_weight, T(1), mask)
        condition_fill!(right_weight, T(0), mask)
    end

    
    mask_left = left .< 1
    # @show sum(mask_left)
    if sum(mask_left) > 0 # for CUDA.jl compatible
        # left[mask_left] .= 1
        # left_weight[mask_left] .= 0
        condition_fill!(left, T(1), mask_left)
        condition_fill!(left_weight, T(0), mask_left)

        # @show sum(left .< 1)
    end

    mask_left = left .> limit
    # @show sum(mask_left)
    if sum(mask_left) > 0
        # left[mask_left] .= limit
        # left_weight[mask_left] .= 0
        condition_fill!(left, T(limit), mask_left)
        condition_fill!(left_weight, T(0), mask_left)

        # @show sum(left .> limit)
    end

    mask_right = right .< 1
    # @show sum(mask_right)
    if sum(mask_right) > 0
        # right[mask_right] .= 1
        # right_weight[mask_right] .= 0
        r1 = condition_fill!(right, 1, mask_right)
        r2 = condition_fill!(right_weight, T(0), mask_right)

        # @show r1 r2

        # @show sum(right .< 1)
        # @show typeof(right) typeof(mask_right) T(1)
    end

    mask_right = right .> limit
    # @show sum(mask_right)
    if sum(mask_right) > 0
        # right[mask_right] .= limit
        # right_weight[mask_right] .= 0
        condition_fill!(right, T(limit), mask_right)
        condition_fill!(right_weight, T(0), mask_right)

        # @show sum(right .> limit)
    end

    return left, right, left_weight, right_weight
end

function apply_flow_batch_broadcast(img1_tensor, flow_tensor)
    width = size(img1_tensor, 1)
    height = size(img1_tensor, 2)
    channel = size(img1_tensor, 3)
    batch_size = size(img1_tensor, 4)

    img2_t_tensor = zero(img1_tensor)

    size_i = size(img1_tensor, 2)
    size_j = size(img1_tensor, 1)

    based_i = reshape(1:size_i, 1, size_i, 1) .+ flow_tensor[:, :, 1, :]
    based_j = reshape(1:size_j, size_j, 1, 1) .+ flow_tensor[:, :, 2, :]

    bottom, top, bottom_weight, top_weight = get_left_right_weight_broadcast(based_i, height)
    left, right, left_weight, right_weight = get_left_right_weight_broadcast(based_j, width)

    right_top_weight = right_weight .* top_weight
    right_bottom_weight = right_weight .* bottom_weight
    left_top_weight = left_weight .* top_weight
    left_bottom_weight = left_weight .* bottom_weight

    idx_list = Array{CartesianIndex{4},4}[]
    w_r_list = typeof(img1_tensor)[] # dim=4, but dim3=1
    di_list = typeof(img1_tensor)[]
    dj_list = typeof(img1_tensor)[]

    it = zip(
        [right_top_weight, right_bottom_weight, left_top_weight, left_bottom_weight],
        [[top, right], [bottom, right], [top, left], [bottom, left]],
        [right_weight, -right_weight, left_weight, -left_weight],
        [top_weight, bottom_weight, -top_weight, -bottom_weight]
    )

    for (w, v, c_di, c_dj) in it
        a1 = repeat(reshape(v[2], width, height, 1, batch_size), 1, 1, channel, 1)
        a2 = repeat(reshape(v[1], width, height, 1, batch_size), 1, 1, channel, 1)

        a3_raw = similar(a1, channel)
        a3_raw .= 1:channel
        a3 = repeat(reshape(a3_raw, 1, 1, channel, 1), width, height, 1, batch_size)

        a4_raw = similar(a1, batch_size)
        a4_raw .= 1:batch_size
        a4 = repeat(reshape(a4_raw, 1, 1, 1, batch_size), width, height, channel, 1)

        # @show typeof(a1) size(a1) typeof(a2) size(a2) typeof(a3) size(a3) typeof(a4) size(a4)

        idx = CartesianIndex.(a1, a2, a3, a4)
        # @show size(idx) typeof(idx) size(img1_tensor) typeof(img1_tensor)
        # ov = img1_tensor[idx]
        ov = similar(img1_tensor)
        re_map!(ov, img1_tensor, idx)

        w_r = reshape(w, size(w,1), size(w, 2), 1, size(w, 3))
        # @show img2_t_tensor[[CartesianIndex(5, 24, 1, 2)]]
        img2_t_tensor += w_r .* ov

        push!(idx_list, idx)
        push!(w_r_list, w_r)

        di = ov .* reshape(c_di, width, height, 1, batch_size)
        dj = ov .* reshape(c_dj, width, height, 1, batch_size)

        push!(di_list, di)
        push!(dj_list, dj)
    end

    # @show img2_t_tensor[[CartesianIndex(5, 24, 1, 2)]]

    return img2_t_tensor, idx_list, w_r_list, di_list, dj_list
end

function ∇apply_flow_broadcast(grad_tensor, idx_list, w_r_list, di_list, dj_list)
    width = size(grad_tensor, 1)
    height = size(grad_tensor, 2)
    batch_size = size(grad_tensor, 4)   

    #grad_i = zero(grad_tensor)
    # grad_flow = zeros(eltype(grad_tensor), width, height, 2, batch_size)
    grad_flow = similar(grad_tensor, width, height, 2, batch_size)
    fill!(grad_flow, 0)

    # grad_i_cpu = Array(grad_tensor) # copy to cpu
    # fill!
    grad_i_cpu = zeros(size(grad_tensor)...)

    for (idx, w_r, di, dj) in zip(idx_list, w_r_list, di_list, dj_list)
        # c_idx = CartesianIndex(29, 5, 1, 2)
        # @show grad_i[c_idx] grad_tensor[c_idx] w_r[c_idx]
        #@show grad_i[[CartesianIndex(5, 24, 1, 2)]] grad_tensor[[CartesianIndex(5, 24, 1, 2)]] w_r[[CartesianIndex(5, 24, 1, 2)]]
        s_idx_list = findall(idx .== [CartesianIndex(5, 24, 1, 2)])
        # @show s_idx_list
        # @show [grad_tensor[[s_idx]] for s_idx in s_idx_list]
        # @show [w_r[[s_idx]] for s_idx in s_idx_list]
        # @show sum(w_r) maximum(w_r) sum(di) maximum(di) sum(dj) maximum(dj)

        # grad_i[idx] += grad_tensor .* w_r
        gr_cpu = Array(grad_tensor .* w_r)
        idx_cpu = Array(idx)
        for (s_idx, v_cpu) in zip(idx_cpu, gr_cpu)
            grad_i_cpu[s_idx] += v_cpu
        end

        # grad_flow[:, :, 1:1, :] += sum(grad_tensor .* di, dims=3)
        # grad_flow[:, :, 2:2, :] += sum(grad_tensor .* dj, dims=3)
        # @show typeof(grad_flow) size(grad_flow) typeof(grad_tensor) size(grad_tensor) typeof(di) size(di)
        
        #=
        add_dim3_2!(grad_flow, sum(grad_tensor .* di, dims=3), 
                               sum(grad_tensor .* dj, dims=3))
        =#
        
        view(grad_flow, :, :, 1:1, :) .+= sum(grad_tensor .* di, dims=3)
        view(grad_flow, :, :, 2:2, :) .+= sum(grad_tensor .* dj, dims=3)
        
    end

    grad_i = typeof(grad_tensor)(grad_i_cpu)

    # @show grad_i[[CartesianIndex(5, 24, 1, 2)]]

    return grad_i, grad_flow

    #=
    grad_i = zero(grad) |> collect
    grad_flow = zeros(eltype(grad), 2, size(grad, 2), size(grad, 3))
    
    for i in 1:size(grad, 2), j in 1:size(grad, 3)
        tracker_list = back_tracker_mat[i, j]
        for tracker in tracker_list
            grad_i[:, tracker.i, tracker.j] += grad[:, i, j] * tracker.w
            # flow di grad
            grad_flow[1, i, j] += sum(grad[:, i, j] .* tracker.di) 
            # flow dj grad
            grad_flow[2, i, j] += sum(grad[:, i, j] .* tracker.dj) 
        end
    end
    grad_i, grad_flow
    =#
end

function apply_flow_2(img1_c, flow)
    img2_t = zero(img1_c)
    back_tracker_vec_mat = [BackTracker[] for i in 1:size(img1_c, 2), j in 1:size(img1_c, 3)]
    
    for i in 1:size(img1_c, 2), j in 1:size(img1_c, 3)
        bottom, top, bottom_weight, top_weight = get_left_right_weight(i + flow[1, i, j])
        left, right, left_weight, right_weight = get_left_right_weight(j + flow[2, i, j])

        if (left < 1) | (left > size(img1_c, 3))
            left_weight = 0.
        end
        if (bottom < 1) | (bottom > size(img1_c, 2))
            bottom_weight = 0.
        end
        if (right < 1) | (right > size(img1_c, 3))
            right_weight = 0.
        end
        if (top < 1) | (top > size(img1_c, 2))
            top_weight = 0.
        end

        right_top_weight = right_weight * top_weight
        right_bottom_weight = right_weight * bottom_weight
        left_top_weight = left_weight * top_weight
        left_bottom_weight = left_weight * bottom_weight

        for (w, v, c_di, c_dj) in zip(
                          [right_top_weight, right_bottom_weight, left_top_weight, left_bottom_weight],
                          [[top, right], [bottom, right], [top, left], [bottom, left]],
                          [right_weight, -right_weight, left_weight, -left_weight],
                          [top_weight, bottom_weight, -top_weight, -bottom_weight])
            if w > 0
                ov = img1_c[:, v[1], v[2]]
                img2_t[:, i, j] += w * ov
                bt = BackTracker(v[1], v[2], w, ov * c_di, ov * c_dj)
                push!(back_tracker_vec_mat[i,j], bt)
            end
        end
    end
    img2_t, back_tracker_vec_mat
end

function ∇apply_flow_1(grad, back_tracker_mat)
    grad_i = zero(grad) |> collect
    grad_flow = zeros(eltype(grad), 2, size(grad, 2), size(grad, 3))
    
    for i in 1:size(grad, 2), j in 1:size(grad, 3)
        tracker_list = back_tracker_mat[i, j]
        for tracker in tracker_list
            grad_i[:, tracker.i, tracker.j] += grad[:, i, j] * tracker.w
            # flow di grad
            grad_flow[1, i, j] += sum(grad[:, i, j] .* tracker.di) 
            # flow dj grad
            grad_flow[2, i, j] += sum(grad[:, i, j] .* tracker.dj) 
        end
    end
    grad_i, grad_flow
end

apply_flow_1(img1_c, flow) = apply_flow_2(img1_c, flow)[1]

@adjoint function apply_flow_1(img1_arr, flow_arr)
    img2_t, back_tracker_mat = apply_flow_2(img1_arr, flow_arr)
    back(grad) = ∇apply_flow_1(grad, back_tracker_mat)
    return img2_t, back
end


function apply_flow_batch_2(img1_tensor, flow_tensor)
    batch_size = size(img1_tensor, 4)

    img2_t_arr = similar(img1_tensor)
    back_tracker_vec_mat_list = Vector{Matrix{Vector{BackTracker}}}(undef, batch_size)

    for i in 1:batch_size
        img1_arr = permutedims(img1_tensor[:,:,:,i], [3,2,1])
        flow_arr = permutedims(flow_tensor[:,:,:,i], [3,2,1])
        img2_t, back_tracker_vec_mat = apply_flow_2(img1_arr, flow_arr)

        img2_t_arr[:,:,:,i] = permutedims(img2_t, [3,2,1])
        # @show typeof(back_tracker_vec_mat_list) size(back_tracker_vec_mat_list) typeof(back_tracker_vec_mat) size(back_tracker_vec_mat)
        back_tracker_vec_mat_list[i] = back_tracker_vec_mat
    end

    return img2_t_arr, back_tracker_vec_mat_list
end

apply_flow_batch(img1_tensor, flow_tensor) = apply_flow_batch_2(img1_tensor, flow_tensor)[1]

@adjoint function apply_flow_batch(img1_tensor, flow_tensor)
    batch_size = size(img1_tensor, 4)

    img2_t_arr, back_tracker_vec_mat_list = apply_flow_batch_2(img1_tensor, flow_tensor)

    back = function(grad_tensor)
        grad_i_tensor = similar(img1_tensor)
        grad_flow_tensor = similar(flow_tensor)

        for i in 1:batch_size
            grad = permutedims(grad_tensor[:,:,:,i], [3,2,1])
            grad_i, grad_flow = ∇apply_flow_1(grad, back_tracker_vec_mat_list[i])
            # @show size(flow_tensor) typeof(flow_tensor) size(grad_flow) typeof(grad_flow)
            grad_i_tensor[:,:,:,i] = permutedims(grad_i, [3,2,1])
            grad_flow_tensor[:,:,:,i] = permutedims(grad_flow, [3,2,1])
        end
        return grad_i_tensor, grad_flow_tensor
    end

    return img2_t_arr, back
end

