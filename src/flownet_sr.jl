
Base.@kwdef struct FlowNetSR # Simple and Reduced, used as true flow.
    conv1::Chain
    conv2::Chain
    conv3::Chain
    conv3_1::Chain
    conv4::Chain
    conv4_1::Chain

    deconv3::Chain
    deconv2::Chain

    predict_flow4::Conv
    predict_flow3::Conv
    predict_flow2::Conv
    
    upsampled_flow4_to_3::ConvTranspose
    upsampled_flow3_to_2::ConvTranspose
end

function flownet_sr(out_dims::Int)
    FlowNetSR(
        conv1 = conv(6, 64, kernel_size=7, stride=2),
        conv2 = conv(64, 128, kernel_size=5, stride=2),
        conv3 = conv(128, 256, kernel_size=5, stride=2),
        conv3_1 = conv(256, 256),
        conv4 = conv(256, 512, stride=2),
        conv4_1 = conv(512, 512),

        deconv3 = deconv(512, 128),
        deconv2 = deconv(128 + 256 + out_dims, 64),

        predict_flow4 = predict_flow(512, out_dims),
        predict_flow3 = predict_flow(128 + 256 + out_dims, out_dims),
        predict_flow2 = predict_flow(128 + 64 + out_dims, out_dims),

        upsampled_flow4_to_3 = ConvTranspose((4,4), out_dims => out_dims, stride=2, pad=1),
        upsampled_flow3_to_2 = ConvTranspose((4,4), out_dims => out_dims, stride=2, pad=1)
    )
end

function (m::FlowNetSR)(x)
    out_conv2 = x |> m.conv1 |> m.conv2
    out_conv3 = out_conv2 |> m.conv3 |> m.conv3_1
    out_conv4 = out_conv3 |> m.conv4 |> m.conv4_1

    flow4 = out_conv4 |> m.predict_flow4
    flow4_up = crop_like(m.upsampled_flow4_to_3(flow4), out_conv3)
    out_deconv3 = crop_like(m.deconv3(out_conv4), out_conv3)

    concat3 = cat(out_conv3, out_deconv3, flow4_up, dims=3)
    flow3 = m.predict_flow3(concat3)
    flow3_up = crop_like(m.upsampled_flow3_to_2(flow3), out_conv2)
    out_deconv2 = crop_like(m.deconv2(concat3), out_conv2)

    concat2 = cat(out_conv2, out_deconv2, flow3_up, dims=3)
    flow2 = m.predict_flow2(concat2)

    return flow2
end
