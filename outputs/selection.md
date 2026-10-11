# Final selection (full val benchmark, 7772 cases)

Candidates: both U-Nets, every diffusion setting (refine-from-one-shot at t₀, DDIM 50) and the **hybrids** (`hybrid_refineT`: the matched U-Net's prediction → one-hot → noised to t₀ → stochastic DDIM with the constraint fine-tune, known-region reset). Selection axes: tolerant mIoU and architectural validity (both higher = better). The knee = Pareto-front point farthest from the chord between the front's end points, axes min-max normalised over all candidates. **All choices are made on val; the val numbers of the chosen setting are therefore slightly optimistic — the test set gives the final numbers.**

## Recommendation

**Recommended final configuration: size-adaptive hybrid** — sketch with the matched U-Net, then finish with the constraint fine-tune (stochastic DDIM 50, eta 1, known-region reset), re-noising further for bigger holes: small holes → `hybrid_refine400`, medium holes → `hybrid_refine600`, large holes → `hybrid_refine700`. The size bin comes from the mask area relative to the building, both visible in the input.

| configuration | tol. mIoU | validity % | conn. err | room leak | room count err |
|---|---|---|---|---|---|
| size_adaptive | 0.842 | 88.0 | 0.011 | 0.013 | 0.126 |
| hybrid_refine400 | 0.844 | 86.3 | 0.030 | 0.025 | 0.159 |
| hybrid_refine600 | 0.842 | 87.4 | 0.023 | 0.019 | 0.141 |
| unet_matched | 0.846 | 80.4 | 0.092 | 0.039 | 0.175 |
| ft_constraints_refine600 | 0.801 | 83.6 | 0.039 | 0.026 | 0.202 |

Why:
- **9 of the 9 pure diffusion settings are dominated by a hybrid** (a hybrid is at least as good on both tolerant mIoU and validity, better on one); the overall Pareto front is `unet_matched`, `hybrid_refine400`, `hybrid_refine600`, `hybrid_refine700`, `hybrid_refine800`. The U-Net sketch fixes diffusion's weak point (committing to a different but plausible layout); the diffusion finish fixes the U-Net's (broken, dangling structure).
- **Against the matched U-Net (the best pixel-accuracy model)**, the chosen configuration gives up 0.004 tolerant mIoU (statistically clear but tiny) for +7.6 points of validity, 8× lower connectivity error, 3.0× lower room leak and a lower room count error (0.126 vs 0.175).
- Bigger holes need more re-noising: the knees are t₀ = 400 overall, but t₀ = 700 for room holes (`hybrid_refine700`) and large holes (`hybrid_refine700`).
- **Size-adaptive vs the single knee `hybrid_refine400`**: clearly better validity (+1.6 points) and connectivity error (0.011 vs 0.030) for 0.002 tolerant mIoU, below our tolerance of 0.005 (half a point of tolerant mIoU for clear structural gains — a judgement call we set after seeing these results).
- Simpler alternative with one setting: `hybrid_refine600` (0.842, 87.4%, conn. err 0.023); see the size-adaptive − best hybrid bootstrap below.
- Cost: one U-Net pass plus about 20–35 DDIM steps (the steps at or below t₀ = 400–700 of the 50-step schedule over 1,000 levels): 0.22–0.37 s per plan measured on a T4, against 0.53 s for full DDIM 50.
- Caveat: the t₀ values were chosen on val, so these val numbers are slightly optimistic; report the test benchmark as the final result.

## Pareto analysis

### overall — knee: **hybrid_refine400** (max distance to the chord)  ·  figure `pareto.png`

| setting | tmIoU | arch_validity_pct | family | on_front | knee |
|---|---|---|---|---|---|
| ft_constraints_ddim50 | 0.761 | 86.696 | diffusion | False | False |
| unet | 0.775 | 67.872 | U-Net | False | False |
| ft_constraints_refine900 | 0.778 | 87.738 | diffusion | False | False |
| run1_refine600 | 0.784 | 81.794 | diffusion | False | False |
| ft_constraints_refine800 | 0.792 | 86.310 | diffusion | False | False |
| ft_constraints_refine700 | 0.798 | 84.238 | diffusion | False | False |
| hybrid_refine900 | 0.799 | 88.767 | hybrid (U-Net sketch + diffusion) | False | False |
| ft_constraints_refine200 | 0.801 | 81.433 | diffusion | False | False |
| ft_constraints_refine600 | 0.801 | 83.621 | diffusion | False | False |
| ft_control_refine600 | 0.801 | 83.685 | diffusion | False | False |
| ft_constraints_refine400 | 0.803 | 81.845 | diffusion | False | False |
| hybrid_refine800 | 0.827 | 88.883 | hybrid (U-Net sketch + diffusion) | True | False |
| hybrid_refine700 | 0.838 | 88.291 | hybrid (U-Net sketch + diffusion) | True | False |
| hybrid_refine600 | 0.842 | 87.416 | hybrid (U-Net sketch + diffusion) | True | False |
| hybrid_refine400 | 0.844 | 86.336 | hybrid (U-Net sketch + diffusion) | True | True |
| unet_matched | 0.846 | 80.378 | U-Net | True | False |

### type = room — knee: **hybrid_refine700** (max distance to the chord)  ·  figure `pareto_room.png`

| setting | tmIoU | arch_validity_pct | family | on_front | knee |
|---|---|---|---|---|---|
| ft_constraints_ddim50 | 0.566 | 69.003 | diffusion | False | False |
| unet | 0.582 | 28.939 | U-Net | False | False |
| run1_refine600 | 0.584 | 49.518 | diffusion | False | False |
| ft_constraints_refine900 | 0.587 | 70.096 | diffusion | False | False |
| ft_constraints_refine800 | 0.603 | 63.473 | diffusion | False | False |
| ft_constraints_refine600 | 0.606 | 56.785 | diffusion | False | False |
| ft_constraints_refine700 | 0.606 | 57.814 | diffusion | False | False |
| ft_control_refine600 | 0.608 | 57.299 | diffusion | False | False |
| ft_constraints_refine400 | 0.609 | 53.826 | diffusion | False | False |
| ft_constraints_refine200 | 0.609 | 53.891 | diffusion | False | False |
| hybrid_refine900 | 0.628 | 72.733 | hybrid (U-Net sketch + diffusion) | True | False |
| hybrid_refine800 | 0.660 | 69.775 | hybrid (U-Net sketch + diffusion) | True | False |
| hybrid_refine700 | 0.668 | 67.203 | hybrid (U-Net sketch + diffusion) | True | True |
| hybrid_refine600 | 0.672 | 64.695 | hybrid (U-Net sketch + diffusion) | True | False |
| hybrid_refine400 | 0.673 | 62.058 | hybrid (U-Net sketch + diffusion) | True | False |
| unet_matched | 0.680 | 50.868 | U-Net | True | False |

### size = large — knee: **hybrid_refine700** (max distance to the chord)  ·  figure `pareto_large.png`

| setting | tmIoU | arch_validity_pct | family | on_front | knee |
|---|---|---|---|---|---|
| ft_constraints_ddim50 | 0.583 | 68.820 | diffusion | False | False |
| unet | 0.601 | 33.810 | U-Net | False | False |
| run1_refine600 | 0.605 | 54.662 | diffusion | False | False |
| ft_constraints_refine900 | 0.609 | 70.213 | diffusion | False | False |
| ft_constraints_refine800 | 0.625 | 65.145 | diffusion | False | False |
| ft_constraints_refine700 | 0.628 | 61.238 | diffusion | False | False |
| ft_constraints_refine600 | 0.629 | 59.265 | diffusion | False | False |
| ft_constraints_refine200 | 0.629 | 56.673 | diffusion | False | False |
| ft_constraints_refine400 | 0.629 | 56.402 | diffusion | False | False |
| ft_control_refine600 | 0.630 | 60.077 | diffusion | False | False |
| hybrid_refine900 | 0.652 | 73.810 | hybrid (U-Net sketch + diffusion) | True | False |
| hybrid_refine800 | 0.684 | 72.147 | hybrid (U-Net sketch + diffusion) | True | False |
| hybrid_refine700 | 0.692 | 69.787 | hybrid (U-Net sketch + diffusion) | True | True |
| hybrid_refine600 | 0.694 | 67.660 | hybrid (U-Net sketch + diffusion) | True | False |
| hybrid_refine400 | 0.696 | 65.261 | hybrid (U-Net sketch + diffusion) | True | False |
| unet_matched | 0.701 | 53.308 | U-Net | True | False |

## Statistics (paired bootstrap, 1,000 resamples)

- **knee - refine t=600** (A = hybrid_refine400):
  - overall: A is better on tolerant mIoU, connectivity error, architectural validity %; worse on nothing.
  - type = room: A is better on tolerant mIoU, architectural validity %; worse on nothing; no clear difference on connectivity error.
  - size = large: A is better on tolerant mIoU, connectivity error, architectural validity %; worse on nothing.
- **knee - matched U-Net** (A = hybrid_refine400):
  - overall: A is better on connectivity error, architectural validity %; worse on tolerant mIoU.
  - type = room: A is better on connectivity error, architectural validity %; worse on tolerant mIoU.
  - size = large: A is better on connectivity error, architectural validity %; worse on tolerant mIoU.
- **best hybrid - matched U-Net** (A = hybrid_refine600):
  - overall: A is better on connectivity error, architectural validity %; worse on tolerant mIoU.
  - type = room: A is better on connectivity error, architectural validity %; worse on tolerant mIoU.
  - size = large: A is better on connectivity error, architectural validity %; worse on tolerant mIoU.
- **size-adaptive - knee** (A = size_adaptive):
  - overall: A is better on connectivity error, architectural validity %; worse on tolerant mIoU.
  - type = room: A is better on connectivity error, architectural validity %; worse on tolerant mIoU.
  - size = large: A is better on connectivity error, architectural validity %; worse on tolerant mIoU.
- **size-adaptive - best hybrid** (A = size_adaptive):
  - overall: A is better on connectivity error, architectural validity %; worse on nothing; no clear difference on tolerant mIoU.
  - type = room: A is better on connectivity error, architectural validity %; worse on tolerant mIoU.
  - size = large: A is better on connectivity error, architectural validity %; worse on tolerant mIoU.
- **size-adaptive - matched U-Net** (A = size_adaptive):
  - overall: A is better on connectivity error, architectural validity %; worse on tolerant mIoU.
  - type = room: A is better on connectivity error, architectural validity %; worse on tolerant mIoU.
  - size = large: A is better on connectivity error, architectural validity %; worse on tolerant mIoU.

### knee - refine t=600  (A = hybrid_refine400, B = ft_constraints_refine600)

| subset | metric | A − B | 95% interval | verdict |
|---|---|---|---|---|
| overall (n=7772) | tolerant mIoU | +0.0432 | [+0.0406, +0.0459] | clear: A better |
| overall (n=7772) | connectivity error | -0.0085 | [-0.0143, -0.0027] | clear: A better |
| overall (n=7772) | architectural validity % | +2.7149 | [+1.9043, +3.5641] | clear: A better |
| type = room (n=1555) | tolerant mIoU | +0.0670 | [+0.0614, +0.0725] | clear: A better |
| type = room (n=1555) | connectivity error | -0.0167 | [-0.0392, +0.0045] | not clear (interval includes 0) |
| type = room (n=1555) | architectural validity % | +5.2733 | [+2.5723, +8.0402] | clear: A better |
| size = large (n=2585) | tolerant mIoU | +0.0668 | [+0.0619, +0.0716] | clear: A better |
| size = large (n=2585) | connectivity error | -0.0244 | [-0.0426, -0.0062] | clear: A better |
| size = large (n=2585) | architectural validity % | +5.9961 | [+3.9836, +8.0077] | clear: A better |

### knee - matched U-Net  (A = hybrid_refine400, B = unet_matched)

| subset | metric | A − B | 95% interval | verdict |
|---|---|---|---|---|
| overall (n=7772) | tolerant mIoU | -0.0018 | [-0.0026, -0.0009] | clear: A worse |
| overall (n=7772) | connectivity error | -0.0611 | [-0.0685, -0.0533] | clear: A better |
| overall (n=7772) | architectural validity % | +5.9573 | [+5.3139, +6.5881] | clear: A better |
| type = room (n=1555) | tolerant mIoU | -0.0071 | [-0.0087, -0.0054] | clear: A worse |
| type = room (n=1555) | connectivity error | -0.1627 | [-0.1891, -0.1344] | clear: A better |
| type = room (n=1555) | architectural validity % | +11.1897 | [+9.2605, +13.3762] | clear: A better |
| size = large (n=2585) | tolerant mIoU | -0.0056 | [-0.0072, -0.0040] | clear: A worse |
| size = large (n=2585) | connectivity error | -0.1721 | [-0.1938, -0.1497] | clear: A better |
| size = large (n=2585) | architectural validity % | +11.9536 | [+10.3665, +13.5019] | clear: A better |

### best hybrid - matched U-Net  (A = hybrid_refine600, B = unet_matched)

| subset | metric | A − B | 95% interval | verdict |
|---|---|---|---|---|
| overall (n=7772) | tolerant mIoU | -0.0043 | [-0.0053, -0.0032] | clear: A worse |
| overall (n=7772) | connectivity error | -0.0686 | [-0.0776, -0.0597] | clear: A better |
| overall (n=7772) | architectural validity % | +7.0381 | [+6.3301, +7.7072] | clear: A better |
| type = room (n=1555) | tolerant mIoU | -0.0081 | [-0.0102, -0.0061] | clear: A worse |
| type = room (n=1555) | connectivity error | -0.1936 | [-0.2251, -0.1646] | clear: A better |
| type = room (n=1555) | architectural validity % | +13.8264 | [+11.7685, +16.0129] | clear: A better |
| size = large (n=2585) | tolerant mIoU | -0.0068 | [-0.0087, -0.0050] | clear: A worse |
| size = large (n=2585) | connectivity error | -0.1923 | [-0.2166, -0.1663] | clear: A better |
| size = large (n=2585) | architectural validity % | +14.3520 | [+12.5716, +16.0551] | clear: A better |

### size-adaptive - knee  (A = size_adaptive, B = hybrid_refine400)

| subset | metric | A − B | 95% interval | verdict |
|---|---|---|---|---|
| overall (n=7772) | tolerant mIoU | -0.0021 | [-0.0030, -0.0012] | clear: A worse |
| overall (n=7772) | connectivity error | -0.0194 | [-0.0237, -0.0156] | clear: A better |
| overall (n=7772) | architectural validity % | +1.6341 | [+1.1448, +2.1230] | clear: A better |
| type = room (n=1555) | tolerant mIoU | -0.0046 | [-0.0069, -0.0021] | clear: A worse |
| type = room (n=1555) | connectivity error | -0.0547 | [-0.0695, -0.0418] | clear: A better |
| type = room (n=1555) | architectural validity % | +5.3376 | [+3.6013, +7.2042] | clear: A better |
| size = large (n=2585) | tolerant mIoU | -0.0035 | [-0.0055, -0.0015] | clear: A worse |
| size = large (n=2585) | connectivity error | -0.0565 | [-0.0681, -0.0441] | clear: A better |
| size = large (n=2585) | architectural validity % | +4.5261 | [+3.1335, +5.8801] | clear: A better |

### size-adaptive - best hybrid  (A = size_adaptive, B = hybrid_refine600)

| subset | metric | A − B | 95% interval | verdict |
|---|---|---|---|---|
| overall (n=7772) | tolerant mIoU | +0.0004 | [-0.0004, +0.0013] | not clear (interval includes 0) |
| overall (n=7772) | connectivity error | -0.0120 | [-0.0158, -0.0085] | clear: A better |
| overall (n=7772) | architectural validity % | +0.5533 | [+0.0515, +0.9911] | clear: A better |
| type = room (n=1555) | tolerant mIoU | -0.0036 | [-0.0059, -0.0011] | clear: A worse |
| type = room (n=1555) | connectivity error | -0.0238 | [-0.0367, -0.0122] | clear: A better |
| type = room (n=1555) | architectural validity % | +2.7010 | [+1.0289, +4.6302] | clear: A better |
| size = large (n=2585) | tolerant mIoU | -0.0023 | [-0.0046, -0.0001] | clear: A worse |
| size = large (n=2585) | connectivity error | -0.0364 | [-0.0464, -0.0259] | clear: A better |
| size = large (n=2585) | architectural validity % | +2.1277 | [+0.7727, +3.4439] | clear: A better |

### size-adaptive - matched U-Net  (A = size_adaptive, B = unet_matched)

| subset | metric | A − B | 95% interval | verdict |
|---|---|---|---|---|
| overall (n=7772) | tolerant mIoU | -0.0038 | [-0.0048, -0.0028] | clear: A worse |
| overall (n=7772) | connectivity error | -0.0805 | [-0.0896, -0.0718] | clear: A better |
| overall (n=7772) | architectural validity % | +7.5914 | [+6.9352, +8.2990] | clear: A better |
| type = room (n=1555) | tolerant mIoU | -0.0117 | [-0.0143, -0.0092] | clear: A worse |
| type = room (n=1555) | connectivity error | -0.2174 | [-0.2521, -0.1871] | clear: A better |
| type = room (n=1555) | architectural validity % | +16.5273 | [+14.4051, +18.8424] | clear: A better |
| size = large (n=2585) | tolerant mIoU | -0.0091 | [-0.0112, -0.0069] | clear: A worse |
| size = large (n=2585) | connectivity error | -0.2286 | [-0.2538, -0.2019] | clear: A better |
| size = large (n=2585) | architectural validity % | +16.4797 | [+14.6228, +18.2215] | clear: A better |

## Size-adaptive option

For each size bin (mask area as % of the building; known at inference time because the mask is given) the knee of that bin's own Pareto front is used for its cases:

| size bin | knee setting | its tmIoU | its validity % |
|---|---|---|---|
| small | hybrid_refine400 | 0.929 | 97.4 |
| medium | hybrid_refine600 | 0.905 | 96.6 |
| large | hybrid_refine700 | 0.692 | 69.8 |

Combined (`size_adaptive`) vs the single settings:

| setting | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| unet_matched | 0.846 | 0.092 | 0.039 | 0.175 | 0.198 | 80.378 |
| ft_constraints_refine600 | 0.801 | 0.039 | 0.026 | 0.202 | 0.200 | 83.621 |
| hybrid_refine400 | 0.844 | 0.030 | 0.025 | 0.159 | 0.182 | 86.336 |
| hybrid_refine600 | 0.842 | 0.023 | 0.019 | 0.141 | 0.166 | 87.416 |
| size_adaptive | 0.842 | 0.011 | 0.013 | 0.126 | 0.155 | 87.970 |

## All candidates: overall (all metrics)

| setting | tmIoU | mIoU | pixel_acc | F1_door | F1_window | F1_room | conn_err | open_on_wall | room_leak | room_count_err | alignment | dead_ends_per_case | fragments_per_case | unreachable_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unet | 0.775 | 0.708 | 0.897 | 0.603 | 0.617 | 0.929 | 0.178 | 0.974 | 0.065 | 0.301 | 0.990 | 0.295 | 0.127 | 0.112 | 67.872 |
| unet_matched | 0.846 | 0.784 | 0.919 | 0.692 | 0.693 | 0.964 | 0.092 | 0.985 | 0.039 | 0.175 | 0.994 | 0.198 | 0.064 | 0.054 | 80.378 |
| run1_refine600 | 0.784 | 0.728 | 0.893 | 0.681 | 0.669 | 0.943 | 0.039 | 0.999 | 0.027 | 0.231 | 0.999 | 0.212 | 0.013 | 0.088 | 81.794 |
| ft_control_refine600 | 0.801 | 0.744 | 0.899 | 0.693 | 0.682 | 0.951 | 0.045 | 1.000 | 0.029 | 0.215 | 0.999 | 0.206 | 0.017 | 0.074 | 83.685 |
| ft_constraints_refine200 | 0.801 | 0.744 | 0.899 | 0.683 | 0.680 | 0.951 | 0.047 | 0.998 | 0.033 | 0.222 | 0.998 | 0.217 | 0.015 | 0.070 | 81.433 |
| ft_constraints_refine400 | 0.803 | 0.745 | 0.899 | 0.686 | 0.682 | 0.950 | 0.055 | 0.999 | 0.031 | 0.216 | 0.998 | 0.224 | 0.020 | 0.073 | 81.845 |
| ft_constraints_refine600 | 0.801 | 0.744 | 0.898 | 0.693 | 0.684 | 0.953 | 0.039 | 1.000 | 0.026 | 0.202 | 0.999 | 0.200 | 0.015 | 0.080 | 83.621 |
| ft_constraints_refine700 | 0.798 | 0.740 | 0.898 | 0.691 | 0.676 | 0.954 | 0.023 | 1.000 | 0.016 | 0.173 | 0.999 | 0.178 | 0.007 | 0.087 | 84.238 |
| ft_constraints_refine800 | 0.792 | 0.734 | 0.896 | 0.681 | 0.664 | 0.957 | 0.008 | 1.000 | 0.008 | 0.147 | 1.000 | 0.163 | 0.003 | 0.079 | 86.310 |
| ft_constraints_refine900 | 0.778 | 0.719 | 0.887 | 0.648 | 0.637 | 0.960 | 0.004 | 1.000 | 0.005 | 0.121 | 1.000 | 0.145 | 0.001 | 0.062 | 87.738 |
| ft_constraints_ddim50 | 0.761 | 0.702 | 0.869 | 0.623 | 0.604 | 0.956 | 0.005 | 1.000 | 0.005 | 0.120 | 1.000 | 0.146 | 0.001 | 0.067 | 86.696 |
| hybrid_refine400 | 0.844 | 0.787 | 0.920 | 0.734 | 0.721 | 0.967 | 0.030 | 0.999 | 0.025 | 0.159 | 0.999 | 0.182 | 0.011 | 0.060 | 86.336 |
| hybrid_refine600 | 0.842 | 0.783 | 0.919 | 0.736 | 0.719 | 0.969 | 0.023 | 1.000 | 0.019 | 0.141 | 0.999 | 0.166 | 0.010 | 0.066 | 87.416 |
| hybrid_refine700 | 0.838 | 0.779 | 0.918 | 0.735 | 0.719 | 0.970 | 0.011 | 1.000 | 0.013 | 0.125 | 0.999 | 0.154 | 0.004 | 0.068 | 88.291 |
| hybrid_refine800 | 0.827 | 0.768 | 0.916 | 0.720 | 0.699 | 0.970 | 0.005 | 1.000 | 0.007 | 0.109 | 1.000 | 0.144 | 0.001 | 0.061 | 88.883 |
| hybrid_refine900 | 0.799 | 0.738 | 0.904 | 0.673 | 0.659 | 0.969 | 0.003 | 1.000 | 0.005 | 0.101 | 1.000 | 0.138 | 0.001 | 0.058 | 88.767 |
| size_adaptive | 0.842 | 0.784 | 0.919 | 0.736 | 0.720 | 0.970 | 0.011 | 0.999 | 0.013 | 0.126 | 0.999 | 0.155 | 0.004 | 0.068 | 87.970 |

## type=door

| setting | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| unet | 0.885 | 0.019 | 0.008 | 0.052 | 0.049 | 81.555 |
| unet_matched | 0.950 | 0.005 | 0.006 | 0.025 | 0.044 | 92.674 |
| run1_refine600 | 0.904 | 0.001 | 0.005 | 0.034 | 0.045 | 94.537 |
| ft_control_refine600 | 0.916 | 0.001 | 0.005 | 0.034 | 0.047 | 94.923 |
| ft_constraints_refine200 | 0.916 | 0.003 | 0.006 | 0.031 | 0.045 | 93.573 |
| ft_constraints_refine400 | 0.918 | 0.002 | 0.006 | 0.026 | 0.050 | 93.188 |
| ft_constraints_refine600 | 0.917 | 0.001 | 0.005 | 0.031 | 0.044 | 95.630 |
| ft_constraints_refine700 | 0.913 | 0.001 | 0.005 | 0.026 | 0.045 | 95.308 |
| ft_constraints_refine800 | 0.906 | 0.000 | 0.005 | 0.024 | 0.046 | 96.337 |
| ft_constraints_refine900 | 0.895 | 0.000 | 0.005 | 0.026 | 0.045 | 95.694 |
| ft_constraints_ddim50 | 0.893 | 0.000 | 0.005 | 0.031 | 0.044 | 96.080 |
| hybrid_refine400 | 0.948 | 0.001 | 0.005 | 0.023 | 0.050 | 95.823 |
| hybrid_refine600 | 0.944 | 0.000 | 0.005 | 0.020 | 0.050 | 96.465 |
| hybrid_refine700 | 0.940 | 0.000 | 0.005 | 0.021 | 0.050 | 96.787 |
| hybrid_refine800 | 0.934 | 0.000 | 0.005 | 0.022 | 0.050 | 96.722 |
| hybrid_refine900 | 0.907 | 0.001 | 0.005 | 0.024 | 0.048 | 95.694 |
| size_adaptive | 0.946 | 0.000 | 0.005 | 0.021 | 0.050 | 96.080 |

## type=region

| setting | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| unet | 0.686 | 0.352 | 0.140 | 0.603 | 0.650 | 52.314 |
| unet_matched | 0.776 | 0.204 | 0.089 | 0.392 | 0.474 | 66.645 |
| run1_refine600 | 0.698 | 0.115 | 0.068 | 0.499 | 0.463 | 70.437 |
| ft_control_refine600 | 0.719 | 0.099 | 0.063 | 0.447 | 0.463 | 71.851 |
| ft_constraints_refine200 | 0.714 | 0.105 | 0.075 | 0.479 | 0.506 | 68.895 |
| ft_constraints_refine400 | 0.716 | 0.122 | 0.072 | 0.467 | 0.485 | 69.023 |
| ft_constraints_refine600 | 0.719 | 0.098 | 0.060 | 0.444 | 0.454 | 71.208 |
| ft_constraints_refine700 | 0.719 | 0.057 | 0.037 | 0.380 | 0.398 | 73.522 |
| ft_constraints_refine800 | 0.715 | 0.020 | 0.012 | 0.339 | 0.365 | 75.129 |
| ft_constraints_refine900 | 0.701 | 0.011 | 0.007 | 0.299 | 0.330 | 77.057 |
| ft_constraints_ddim50 | 0.675 | 0.012 | 0.007 | 0.314 | 0.326 | 74.229 |
| hybrid_refine400 | 0.774 | 0.072 | 0.059 | 0.363 | 0.412 | 76.992 |
| hybrid_refine600 | 0.773 | 0.067 | 0.045 | 0.325 | 0.384 | 78.663 |
| hybrid_refine700 | 0.772 | 0.031 | 0.031 | 0.303 | 0.356 | 79.756 |
| hybrid_refine800 | 0.764 | 0.011 | 0.012 | 0.259 | 0.326 | 81.491 |
| hybrid_refine900 | 0.732 | 0.006 | 0.005 | 0.258 | 0.310 | 80.334 |
| size_adaptive | 0.773 | 0.031 | 0.031 | 0.305 | 0.359 | 79.434 |

## type=room

| setting | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| unet | 0.582 | 0.486 | 0.105 | 0.795 | 0.636 | 28.939 |
| unet_matched | 0.680 | 0.239 | 0.060 | 0.434 | 0.357 | 50.868 |
| run1_refine600 | 0.584 | 0.077 | 0.039 | 0.594 | 0.447 | 49.518 |
| ft_control_refine600 | 0.608 | 0.122 | 0.048 | 0.570 | 0.413 | 57.299 |
| ft_constraints_refine200 | 0.609 | 0.122 | 0.053 | 0.572 | 0.428 | 53.891 |
| ft_constraints_refine400 | 0.609 | 0.147 | 0.049 | 0.565 | 0.477 | 53.826 |
| ft_constraints_refine600 | 0.606 | 0.093 | 0.040 | 0.515 | 0.400 | 56.785 |
| ft_constraints_refine700 | 0.606 | 0.055 | 0.023 | 0.439 | 0.341 | 57.814 |
| ft_constraints_refine800 | 0.603 | 0.021 | 0.011 | 0.355 | 0.297 | 63.473 |
| ft_constraints_refine900 | 0.587 | 0.008 | 0.005 | 0.261 | 0.243 | 70.096 |
| ft_constraints_ddim50 | 0.566 | 0.009 | 0.005 | 0.239 | 0.253 | 69.003 |
| hybrid_refine400 | 0.673 | 0.077 | 0.037 | 0.392 | 0.336 | 62.058 |
| hybrid_refine600 | 0.672 | 0.046 | 0.026 | 0.341 | 0.287 | 64.695 |
| hybrid_refine700 | 0.668 | 0.023 | 0.016 | 0.288 | 0.253 | 67.203 |
| hybrid_refine800 | 0.660 | 0.011 | 0.007 | 0.246 | 0.232 | 69.775 |
| hybrid_refine900 | 0.628 | 0.006 | 0.005 | 0.203 | 0.217 | 72.733 |
| size_adaptive | 0.668 | 0.022 | 0.016 | 0.291 | 0.253 | 67.395 |

## type=wall

| setting | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| unet | 0.876 | 0.022 | 0.012 | 0.031 | 0.121 | 89.781 |
| unet_matched | 0.923 | 0.007 | 0.008 | 0.017 | 0.098 | 95.823 |
| run1_refine600 | 0.887 | 0.003 | 0.005 | 0.020 | 0.091 | 97.172 |
| ft_control_refine600 | 0.894 | 0.002 | 0.005 | 0.018 | 0.091 | 96.979 |
| ft_constraints_refine200 | 0.897 | 0.004 | 0.006 | 0.016 | 0.093 | 95.887 |
| ft_constraints_refine400 | 0.897 | 0.003 | 0.006 | 0.017 | 0.093 | 96.787 |
| ft_constraints_refine600 | 0.897 | 0.003 | 0.004 | 0.012 | 0.089 | 97.494 |
| ft_constraints_refine700 | 0.891 | 0.001 | 0.004 | 0.013 | 0.091 | 97.301 |
| ft_constraints_refine800 | 0.885 | 0.001 | 0.005 | 0.012 | 0.090 | 98.008 |
| ft_constraints_refine900 | 0.873 | 0.001 | 0.004 | 0.012 | 0.091 | 98.136 |
| ft_constraints_ddim50 | 0.855 | 0.001 | 0.004 | 0.012 | 0.095 | 96.594 |
| hybrid_refine400 | 0.925 | 0.003 | 0.005 | 0.013 | 0.096 | 98.393 |
| hybrid_refine600 | 0.922 | 0.003 | 0.005 | 0.012 | 0.096 | 98.586 |
| hybrid_refine700 | 0.915 | 0.002 | 0.004 | 0.011 | 0.095 | 98.779 |
| hybrid_refine800 | 0.905 | 0.001 | 0.004 | 0.011 | 0.096 | 98.393 |
| hybrid_refine900 | 0.881 | 0.001 | 0.005 | 0.014 | 0.095 | 97.751 |
| size_adaptive | 0.923 | 0.003 | 0.005 | 0.012 | 0.096 | 98.458 |

## type=window

| setting | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| unet | 0.844 | 0.009 | 0.024 | 0.023 | 0.015 | 86.830 |
| unet_matched | 0.903 | 0.002 | 0.007 | 0.007 | 0.014 | 95.933 |
| run1_refine600 | 0.848 | 0.000 | 0.004 | 0.007 | 0.014 | 97.353 |
| ft_control_refine600 | 0.869 | 0.000 | 0.004 | 0.006 | 0.014 | 97.418 |
| ft_constraints_refine200 | 0.870 | 0.000 | 0.006 | 0.008 | 0.015 | 94.964 |
| ft_constraints_refine400 | 0.873 | 0.000 | 0.005 | 0.005 | 0.014 | 96.449 |
| ft_constraints_refine600 | 0.867 | 0.000 | 0.003 | 0.005 | 0.014 | 97.030 |
| ft_constraints_refine700 | 0.860 | 0.000 | 0.003 | 0.006 | 0.014 | 97.289 |
| ft_constraints_refine800 | 0.853 | 0.000 | 0.003 | 0.004 | 0.014 | 98.644 |
| ft_constraints_refine900 | 0.833 | 0.000 | 0.003 | 0.006 | 0.015 | 97.740 |
| ft_constraints_ddim50 | 0.818 | 0.000 | 0.004 | 0.006 | 0.013 | 97.611 |
| hybrid_refine400 | 0.902 | 0.000 | 0.003 | 0.004 | 0.014 | 98.451 |
| hybrid_refine600 | 0.900 | 0.000 | 0.004 | 0.005 | 0.014 | 98.709 |
| hybrid_refine700 | 0.894 | 0.000 | 0.003 | 0.004 | 0.014 | 98.967 |
| hybrid_refine800 | 0.874 | 0.000 | 0.003 | 0.005 | 0.015 | 98.063 |
| hybrid_refine900 | 0.844 | 0.000 | 0.002 | 0.007 | 0.017 | 97.353 |
| size_adaptive | 0.902 | 0.000 | 0.004 | 0.005 | 0.014 | 98.515 |

## size_bin=large

| setting | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| unet | 0.601 | 0.483 | 0.128 | 0.805 | 0.735 | 33.810 |
| unet_matched | 0.701 | 0.260 | 0.078 | 0.481 | 0.472 | 53.308 |
| run1_refine600 | 0.605 | 0.113 | 0.056 | 0.634 | 0.521 | 54.662 |
| ft_control_refine600 | 0.630 | 0.131 | 0.059 | 0.592 | 0.497 | 60.077 |
| ft_constraints_refine200 | 0.629 | 0.132 | 0.068 | 0.610 | 0.532 | 56.673 |
| ft_constraints_refine400 | 0.629 | 0.155 | 0.064 | 0.599 | 0.550 | 56.402 |
| ft_constraints_refine600 | 0.629 | 0.112 | 0.053 | 0.556 | 0.487 | 59.265 |
| ft_constraints_refine700 | 0.628 | 0.065 | 0.030 | 0.474 | 0.420 | 61.238 |
| ft_constraints_refine800 | 0.625 | 0.023 | 0.011 | 0.399 | 0.376 | 65.145 |
| ft_constraints_refine900 | 0.609 | 0.010 | 0.005 | 0.318 | 0.327 | 70.213 |
| ft_constraints_ddim50 | 0.583 | 0.011 | 0.005 | 0.314 | 0.324 | 68.820 |
| hybrid_refine400 | 0.696 | 0.087 | 0.050 | 0.441 | 0.425 | 65.261 |
| hybrid_refine600 | 0.694 | 0.067 | 0.036 | 0.386 | 0.380 | 67.660 |
| hybrid_refine700 | 0.692 | 0.031 | 0.023 | 0.343 | 0.344 | 69.787 |
| hybrid_refine800 | 0.684 | 0.012 | 0.009 | 0.288 | 0.312 | 72.147 |
| hybrid_refine900 | 0.652 | 0.006 | 0.005 | 0.261 | 0.293 | 73.810 |
| size_adaptive | 0.692 | 0.031 | 0.023 | 0.343 | 0.344 | 69.787 |

## size_bin=medium

| setting | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| unet | 0.850 | 0.045 | 0.024 | 0.080 | 0.108 | 84.030 |
| unet_matched | 0.907 | 0.014 | 0.009 | 0.035 | 0.082 | 92.981 |
| run1_refine600 | 0.863 | 0.005 | 0.005 | 0.049 | 0.079 | 93.770 |
| ft_control_refine600 | 0.875 | 0.004 | 0.006 | 0.045 | 0.083 | 94.322 |
| ft_constraints_refine200 | 0.875 | 0.007 | 0.007 | 0.046 | 0.083 | 93.178 |
| ft_constraints_refine400 | 0.877 | 0.009 | 0.007 | 0.042 | 0.084 | 93.218 |
| ft_constraints_refine600 | 0.876 | 0.005 | 0.005 | 0.041 | 0.077 | 94.361 |
| ft_constraints_refine700 | 0.871 | 0.003 | 0.005 | 0.037 | 0.076 | 94.282 |
| ft_constraints_refine800 | 0.865 | 0.002 | 0.005 | 0.035 | 0.073 | 95.978 |
| ft_constraints_refine900 | 0.851 | 0.002 | 0.004 | 0.038 | 0.070 | 94.953 |
| ft_constraints_ddim50 | 0.836 | 0.002 | 0.005 | 0.037 | 0.078 | 93.651 |
| hybrid_refine400 | 0.907 | 0.004 | 0.005 | 0.029 | 0.080 | 96.254 |
| hybrid_refine600 | 0.905 | 0.002 | 0.005 | 0.029 | 0.080 | 96.648 |
| hybrid_refine700 | 0.900 | 0.002 | 0.005 | 0.026 | 0.079 | 96.609 |
| hybrid_refine800 | 0.889 | 0.002 | 0.004 | 0.030 | 0.079 | 96.767 |
| hybrid_refine900 | 0.860 | 0.002 | 0.004 | 0.035 | 0.079 | 94.677 |
| size_adaptive | 0.905 | 0.002 | 0.005 | 0.029 | 0.080 | 96.648 |

## size_bin=small

| setting | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| unet | 0.871 | 0.007 | 0.011 | 0.022 | 0.043 | 85.628 |
| unet_matched | 0.929 | 0.002 | 0.007 | 0.011 | 0.040 | 94.719 |
| run1_refine600 | 0.884 | 0.001 | 0.005 | 0.012 | 0.039 | 96.794 |
| ft_control_refine600 | 0.898 | 0.000 | 0.005 | 0.010 | 0.039 | 96.530 |
| ft_constraints_refine200 | 0.899 | 0.001 | 0.006 | 0.011 | 0.039 | 94.342 |
| ft_constraints_refine400 | 0.901 | 0.000 | 0.006 | 0.008 | 0.039 | 95.775 |
| ft_constraints_refine600 | 0.898 | 0.000 | 0.005 | 0.009 | 0.039 | 97.095 |
| ft_constraints_refine700 | 0.893 | 0.000 | 0.005 | 0.009 | 0.040 | 97.058 |
| ft_constraints_refine800 | 0.886 | 0.000 | 0.005 | 0.008 | 0.040 | 97.699 |
| ft_constraints_refine900 | 0.873 | 0.000 | 0.005 | 0.008 | 0.040 | 97.925 |
| ft_constraints_ddim50 | 0.864 | 0.000 | 0.005 | 0.011 | 0.039 | 97.473 |
| hybrid_refine400 | 0.929 | 0.000 | 0.005 | 0.009 | 0.041 | 97.397 |
| hybrid_refine600 | 0.926 | 0.000 | 0.005 | 0.008 | 0.041 | 97.850 |
| hybrid_refine700 | 0.921 | 0.000 | 0.005 | 0.008 | 0.040 | 98.378 |
| hybrid_refine800 | 0.908 | 0.000 | 0.005 | 0.010 | 0.042 | 97.661 |
| hybrid_refine900 | 0.882 | 0.000 | 0.005 | 0.009 | 0.041 | 97.699 |
| size_adaptive | 0.929 | 0.000 | 0.005 | 0.009 | 0.041 | 97.397 |
