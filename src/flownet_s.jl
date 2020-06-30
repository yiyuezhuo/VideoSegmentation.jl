
Base.@kwdef struct FlowNetS # Simple, used for key frame segmentation 
    conv1::Chain
    conv2::Chain
    conv3::Chain
    conv3_1::Chain
    conv4::Chain
    conv4_1::Chain
    conv5::Chain
    conv5_1::Chain
    conv6::Chain
    conv6_1::Chain

    deconv5::Chain
    deconv4::Chain
    deconv3::Chain
    deconv2::Chain

    predict_flow6::Conv
    predict_flow5::Conv
    predict_flow4::Conv
    predict_flow3::Conv
    predict_flow2::Conv
    
    upsampled_flow6_to_5::ConvTranspose
    upsampled_flow5_to_4::ConvTranspose
    upsampled_flow4_to_3::ConvTranspose
    upsampled_flow3_to_2::ConvTranspose
end

function flownet_s(input_dims::Int, out_dims::Int)
    FlowNetS(
        conv1 = conv(input_dims, 64, kernel_size=7, stride=2),
        conv2 = conv(64, 128, kernel_size=5, stride=2),
        conv3 = conv(128, 256, kernel_size=5, stride=2),
        conv3_1 = conv(256, 256),
        conv4 = conv(256, 512, stride=2),
        conv4_1 = conv(512, 512),
        conv5 = conv(512, 512, stride=2),
        conv5_1 = conv(512, 512),
        conv6 = conv(512, 1024, stride=2),
        conv6_1 = conv(1024, 1024),

        deconv5 = deconv(1024, 512),
        deconv4 = deconv(1024 + out_dims, 256),
        deconv3 = deconv(768 + out_dims, 128),
        deconv2 = deconv(384 + out_dims, 64),

        predict_flow6 = predict_flow(1024, out_dims),
        predict_flow5 = predict_flow(1024 + out_dims, out_dims),
        predict_flow4 = predict_flow(768 + out_dims, out_dims),
        predict_flow3 = predict_flow(384 + out_dims, out_dims),
        predict_flow2 = predict_flow(192 + out_dims, out_dims),

        upsampled_flow6_to_5 = ConvTranspose((4,4), out_dims => out_dims, stride=2, pad=1),
        upsampled_flow5_to_4 = ConvTranspose((4,4), out_dims => out_dims, stride=2, pad=1),
        upsampled_flow4_to_3 = ConvTranspose((4,4), out_dims => out_dims, stride=2, pad=1),
        upsampled_flow3_to_2 = ConvTranspose((4,4), out_dims => out_dims, stride=2, pad=1)
    )
end

function (m::FlowNetS)(x)
    out_conv2 = x |> m.conv1 |> m.conv2
    out_conv3 = out_conv2 |> m.conv3 |> m.conv3_1
    out_conv4 = out_conv3 |> m.conv4 |> m.conv4_1
    out_conv5 = out_conv4 |> m.conv5 |> m.conv5_1
    out_conv6 = out_conv5 |> m.conv6 |> m.conv6_1

    flow6 = out_conv6 |> m.predict_flow6
    flow6_up = crop_like(m.upsampled_flow6_to_5(flow6), out_conv5)
    out_deconv5 = crop_like(m.deconv5(out_conv6), out_conv5)

    concat5 = cat(out_conv5, out_deconv5, flow6_up, dims=3)
    flow5 = m.predict_flow5(concat5)
    flow5_up = crop_like(m.upsampled_flow5_to_4(flow5), out_conv4)
    out_deconv4 = crop_like(m.deconv4(concat5), out_conv4)

    concat4 = cat(out_conv4, out_deconv4, flow5_up, dims=3)
    flow4 = m.predict_flow4(concat4)
    flow4_up = crop_like(m.upsampled_flow4_to_3(flow4), out_conv3)
    out_deconv3 = crop_like(m.deconv3(concat4), out_conv3)

    concat3 = cat(out_conv3, out_deconv3, flow4_up, dims=3)
    flow3 = m.predict_flow3(concat3)
    flow3_up = crop_like(m.upsampled_flow3_to_2(flow3), out_conv2)
    out_deconv2 = crop_like(m.deconv2(concat3), out_conv2)

    concat2 = cat(out_conv2, out_deconv2, flow3_up, dims=3)
    flow2 = m.predict_flow2(concat2)

    return flow2
end

@functor FlowNetS