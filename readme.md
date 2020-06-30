# Segmentation prototype

My goal is to implement [Deep Feature Flow for Video Recognition](https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhu_Deep_Feature_Flow_CVPR_2017_paper.pdf) in Julia. 

## Setup

Unfortunately, Julia's ecosystem is not ready for such work. So some non-standard setup are required, though which is not that hard compared to setting up an old-fashion DL framework such as MXNet used by [original author](https://github.com/msracver/Deep-Feature-Flow).

[VideoLoader.jl](https://github.com/yiyuezhuo/VideoLoader.jl) use some fix which is merged but not be release by [VideoIO](https://github.com/JuliaIO/VideoIO.jl) yet. See `readme.md` in `VideoLoader.jl` for the related setup process.

[NNlib.jl] provided a wrong gradient for `maxpool`. A fix is released by `NNlib`, but not be used by `Flux`. Use `]dev NNlib` to get dev version for `NNlib`. By the way this bug make almost no effect on training.

## `apply_flow`

My main motivation to use Julia instead of PyTorch for this task, is that I can write this module easily, not bothering to obscure C extension (then I found Julia failed on some other strange aspects, leading terrible time wasting, so now I recommend DL-guy to still use PyTorch for such task.). The implementation includes a step transferring data to CPU from GPU to run a parallel unfriendly operation. Maybe some smart one can find a way to avoid it, but I can't figure out how to do it.

(I think a similar function has been implemented by PyTorch official or contributor, but I haven't check it yet.)

## Dummy dataset

I think clip based method, against frame based method, will not be beneficial to specified task, so I just create a dummy dataset and don't invest any time to label dense label data, as some dumb one suggesting this expected.

Dummy dataset is implicitly represented by classifying normed intensity > 0.1 as object class, other as background class.

## Usage

See following two gists (Jupyter notebooks):

* [Key frame model training](https://gist.github.com/yiyuezhuo/e6f915d1fcec5f484a34f0417a47de3d)
* [Flow model training (detached from key frame model)](https://gist.github.com/yiyuezhuo/3c8a42843021fce77d09b3e8c933735c)