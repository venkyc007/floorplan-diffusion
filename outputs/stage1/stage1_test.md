# Stage 1: final TEST evaluation (run once, frozen Day 7 configurations)

Test benchmark: 7774 cases (data/benchmarks/test_benchmark.npz). Methods exactly as chosen on val (configs/stage1/app.yaml): size-adaptive hybrid (t0 = 400 / 600 / 700 for small / medium / large holes), matched U-Net, hybrid t0 = 600, constraint fine-tune refine-600, nearest fill. Same metrics and paired bootstrap as on val (src/stage1/selection.py). Nothing was tuned on test.

## Val vs test (overall)

Differences in percentage points for shares / IoU / F1 (x100), raw for counts. **Bold** = test differs from val by more than 2 points.

| method | metric | val | test | test - val |
|---|---|---|---|---|
| size_adaptive | tmIoU | 0.8424 | 0.8457 | +0.33 |
| size_adaptive | conn_err | 0.0111 | 0.0112 | +0.00 |
| size_adaptive | room_leak | 0.0132 | 0.0155 | +0.23 |
| size_adaptive | room_count_err | 0.1265 | 0.1178 | -0.01 |
| size_adaptive | dead_ends_per_case | 0.1545 | 0.1475 | -0.01 |
| size_adaptive | arch_validity_pct | 87.9696 | 88.0499 | +0.08 |
| size_adaptive | mIoU | 0.7843 | 0.7889 | +0.46 |
| size_adaptive | F1_door | 0.7357 | 0.7516 | +1.59 |
| size_adaptive | F1_window | 0.7202 | 0.7312 | +1.11 |
| size_adaptive | F1_room | 0.9696 | 0.9722 | +0.26 |
| unet_matched | tmIoU | 0.8462 | 0.8498 | +0.36 |
| unet_matched | conn_err | 0.0916 | 0.0850 | -0.01 |
| unet_matched | room_leak | 0.0390 | 0.0406 | +0.16 |
| unet_matched | room_count_err | 0.1751 | 0.1631 | -0.01 |
| unet_matched | dead_ends_per_case | 0.1976 | 0.1787 | -0.02 |
| unet_matched | arch_validity_pct | 80.3783 | 80.9622 | +0.58 |
| unet_matched | mIoU | 0.7837 | 0.7883 | +0.46 |
| unet_matched | F1_door | 0.6923 | 0.7107 | +1.85 |
| unet_matched | F1_window | 0.6935 | 0.7065 | +1.30 |
| unet_matched | F1_room | 0.9636 | 0.9665 | +0.29 |
| hybrid_refine600 | tmIoU | 0.8419 | 0.8456 | +0.37 |
| hybrid_refine600 | conn_err | 0.0230 | 0.0221 | -0.00 |
| hybrid_refine600 | room_leak | 0.0188 | 0.0199 | +0.12 |
| hybrid_refine600 | room_count_err | 0.1405 | 0.1295 | -0.01 |
| hybrid_refine600 | dead_ends_per_case | 0.1665 | 0.1554 | -0.01 |
| hybrid_refine600 | arch_validity_pct | 87.4164 | 87.5740 | +0.16 |
| hybrid_refine600 | mIoU | 0.7834 | 0.7883 | +0.49 |
| hybrid_refine600 | F1_door | 0.7363 | 0.7522 | +1.59 |
| hybrid_refine600 | F1_window | 0.7186 | 0.7334 | +1.49 |
| hybrid_refine600 | F1_room | 0.9687 | 0.9717 | +0.30 |
| ft_constraints_refine600 | tmIoU | 0.8012 | 0.8042 | +0.30 |
| ft_constraints_refine600 | conn_err | 0.0390 | 0.0369 | -0.00 |
| ft_constraints_refine600 | room_leak | 0.0256 | 0.0252 | -0.04 |
| ft_constraints_refine600 | room_count_err | 0.2016 | 0.1845 | -0.02 |
| ft_constraints_refine600 | dead_ends_per_case | 0.2005 | 0.1870 | -0.01 |
| ft_constraints_refine600 | arch_validity_pct | 83.6207 | 84.1008 | +0.48 |
| ft_constraints_refine600 | mIoU | 0.7441 | 0.7479 | +0.38 |
| ft_constraints_refine600 | F1_door | 0.6934 | 0.7026 | +0.93 |
| ft_constraints_refine600 | F1_window | 0.6840 | 0.6936 | +0.97 |
| ft_constraints_refine600 | F1_room | 0.9527 | 0.9551 | +0.24 |
| nearest_fill | tmIoU | 0.4892 | 0.4930 | +0.38 |
| nearest_fill | conn_err | 0.0656 | 0.0643 | -0.00 |
| nearest_fill | room_leak | 0.6268 | 0.6237 | -0.31 |
| nearest_fill | room_count_err | 1.7321 | 1.7039 | -0.03 |
| nearest_fill | dead_ends_per_case | 0.5796 | 0.5498 | -0.03 |
| nearest_fill | arch_validity_pct | 35.9624 | 36.6092 | +0.65 |
| nearest_fill | mIoU | 0.4374 | 0.4405 | +0.31 |
| nearest_fill | F1_door | 0.4496 | 0.4558 | +0.62 |
| nearest_fill | F1_window | 0.4514 | 0.4661 | +1.48 |
| nearest_fill | F1_room | 0.6936 | 0.7006 | +0.71 |

## Test, overall

| setting | tmIoU | mIoU | pixel_acc | F1_door | F1_window | F1_room | conn_err | open_on_wall | room_leak | room_count_err | alignment | dead_ends_per_case | fragments_per_case | unreachable_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| size_adaptive | 0.846 | 0.789 | 0.920 | 0.752 | 0.731 | 0.972 | 0.011 | 1.000 | 0.016 | 0.118 | 0.999 | 0.148 | 0.004 | 0.069 | 88.050 |
| unet_matched | 0.850 | 0.788 | 0.921 | 0.711 | 0.706 | 0.966 | 0.085 | 0.984 | 0.041 | 0.163 | 0.994 | 0.179 | 0.059 | 0.056 | 80.962 |
| hybrid_refine600 | 0.846 | 0.788 | 0.921 | 0.752 | 0.733 | 0.972 | 0.022 | 0.999 | 0.020 | 0.130 | 0.999 | 0.155 | 0.010 | 0.067 | 87.574 |
| ft_constraints_refine600 | 0.804 | 0.748 | 0.900 | 0.703 | 0.694 | 0.955 | 0.037 | 0.999 | 0.025 | 0.184 | 0.999 | 0.187 | 0.012 | 0.078 | 84.101 |
| nearest_fill | 0.493 | 0.440 | 0.674 | 0.456 | 0.466 | 0.701 | 0.064 | 0.995 | 0.624 | 1.704 | 0.999 | 0.550 | 0.002 | 0.014 | 36.609 |

## Test, type=door

|  | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| size_adaptive | 0.948 | 0.000 | 0.006 | 0.012 | 0.053 | 96.144 |
| unet_matched | 0.949 | 0.008 | 0.006 | 0.019 | 0.053 | 93.445 |
| hybrid_refine600 | 0.946 | 0.000 | 0.006 | 0.011 | 0.053 | 96.979 |
| ft_constraints_refine600 | 0.915 | 0.002 | 0.006 | 0.019 | 0.052 | 94.859 |
| nearest_fill | 0.552 | 0.035 | 0.323 | 1.288 | 1.010 | 34.254 |

## Test, type=region

|  | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| size_adaptive | 0.784 | 0.035 | 0.039 | 0.292 | 0.351 | 81.298 |
| unet_matched | 0.789 | 0.190 | 0.103 | 0.380 | 0.416 | 69.409 |
| hybrid_refine600 | 0.788 | 0.062 | 0.051 | 0.311 | 0.361 | 80.270 |
| ft_constraints_refine600 | 0.734 | 0.081 | 0.058 | 0.402 | 0.414 | 74.100 |
| nearest_fill | 0.537 | 0.065 | 0.456 | 1.413 | 0.801 | 36.954 |

## Test, type=room

|  | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| size_adaptive | 0.675 | 0.019 | 0.017 | 0.269 | 0.219 | 66.238 |
| unet_matched | 0.687 | 0.220 | 0.055 | 0.394 | 0.307 | 50.289 |
| hybrid_refine600 | 0.677 | 0.047 | 0.024 | 0.310 | 0.248 | 63.473 |
| ft_constraints_refine600 | 0.615 | 0.098 | 0.038 | 0.479 | 0.352 | 57.685 |
| nearest_fill | 0.271 | 0.165 | 0.912 | 3.317 | 0.583 | 9.196 |

## Test, type=wall

|  | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| size_adaptive | 0.924 | 0.001 | 0.005 | 0.011 | 0.105 | 98.008 |
| unet_matched | 0.925 | 0.002 | 0.007 | 0.013 | 0.107 | 95.758 |
| hybrid_refine600 | 0.922 | 0.001 | 0.005 | 0.011 | 0.105 | 98.265 |
| ft_constraints_refine600 | 0.894 | 0.002 | 0.005 | 0.015 | 0.104 | 96.144 |
| nearest_fill | 0.592 | 0.043 | 0.620 | 1.346 | 0.267 | 24.229 |

## Test, type=window

|  | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| size_adaptive | 0.898 | 0.001 | 0.007 | 0.005 | 0.009 | 98.582 |
| unet_matched | 0.900 | 0.005 | 0.012 | 0.008 | 0.009 | 95.938 |
| hybrid_refine600 | 0.895 | 0.001 | 0.007 | 0.005 | 0.010 | 98.904 |
| ft_constraints_refine600 | 0.863 | 0.001 | 0.007 | 0.006 | 0.012 | 97.743 |
| nearest_fill | 0.513 | 0.014 | 0.826 | 1.154 | 0.086 | 78.530 |

## Test, size_bin=large

|  | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| size_adaptive | 0.700 | 0.031 | 0.028 | 0.325 | 0.317 | 70.667 |
| unet_matched | 0.710 | 0.236 | 0.080 | 0.450 | 0.405 | 54.678 |
| hybrid_refine600 | 0.703 | 0.064 | 0.038 | 0.361 | 0.341 | 68.367 |
| ft_constraints_refine600 | 0.639 | 0.105 | 0.050 | 0.511 | 0.436 | 61.273 |
| nearest_fill | 0.355 | 0.131 | 0.749 | 2.672 | 0.750 | 15.759 |

## Test, size_bin=medium

|  | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| size_adaptive | 0.907 | 0.001 | 0.008 | 0.020 | 0.084 | 96.131 |
| unet_matched | 0.910 | 0.015 | 0.014 | 0.029 | 0.088 | 93.059 |
| hybrid_refine600 | 0.907 | 0.001 | 0.008 | 0.020 | 0.084 | 96.131 |
| ft_constraints_refine600 | 0.873 | 0.004 | 0.009 | 0.034 | 0.083 | 94.216 |
| nearest_fill | 0.567 | 0.041 | 0.577 | 1.312 | 0.448 | 43.039 |

## Test, size_bin=small

|  | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| size_adaptive | 0.931 | 0.001 | 0.004 | 0.007 | 0.041 | 97.480 |
| unet_matched | 0.931 | 0.003 | 0.004 | 0.008 | 0.042 | 95.337 |
| hybrid_refine600 | 0.927 | 0.001 | 0.004 | 0.006 | 0.041 | 98.345 |
| ft_constraints_refine600 | 0.901 | 0.001 | 0.004 | 0.006 | 0.041 | 96.954 |
| nearest_fill | 0.559 | 0.021 | 0.534 | 1.123 | 0.449 | 50.997 |

## Paired bootstrap on test (size-adaptive hybrid against the comparison methods)

- **size-adaptive - matched U-Net** (A = size_adaptive):
  - overall: A is better on connectivity error, architectural validity %; worse on tolerant mIoU.
  - type = room: A is better on connectivity error, architectural validity %; worse on tolerant mIoU.
  - size = large: A is better on connectivity error, architectural validity %; worse on tolerant mIoU.
- **size-adaptive - refine 600** (A = size_adaptive):
  - overall: A is better on tolerant mIoU, connectivity error, architectural validity %; worse on nothing.
  - type = room: A is better on tolerant mIoU, connectivity error, architectural validity %; worse on nothing.
  - size = large: A is better on tolerant mIoU, connectivity error, architectural validity %; worse on nothing.
- **size-adaptive - hybrid 600** (A = size_adaptive):
  - overall: A is better on connectivity error; worse on nothing; no clear difference on tolerant mIoU, architectural validity %.
  - type = room: A is better on connectivity error, architectural validity %; worse on nothing; no clear difference on tolerant mIoU.
  - size = large: A is better on connectivity error, architectural validity %; worse on tolerant mIoU.

### size-adaptive - matched U-Net  (A = size_adaptive, B = unet_matched)

| subset | metric | A − B | 95% interval | verdict |
|---|---|---|---|---|
| overall (n=7774) | tolerant mIoU | -0.0042 | [-0.0051, -0.0032] | clear: A worse |
| overall (n=7774) | connectivity error | -0.0738 | [-0.0837, -0.0643] | clear: A better |
| overall (n=7774) | architectural validity % | +7.0877 | [+6.4185, +7.7566] | clear: A better |
| type = room (n=1555) | tolerant mIoU | -0.0114 | [-0.0140, -0.0086] | clear: A worse |
| type = room (n=1555) | connectivity error | -0.2006 | [-0.2380, -0.1653] | clear: A better |
| type = room (n=1555) | architectural validity % | +15.9486 | [+13.6977, +18.1994] | clear: A better |
| size = large (n=2608) | tolerant mIoU | -0.0102 | [-0.0123, -0.0081] | clear: A worse |
| size = large (n=2608) | connectivity error | -0.2048 | [-0.2312, -0.1798] | clear: A better |
| size = large (n=2608) | architectural validity % | +15.9893 | [+14.3788, +17.7531] | clear: A better |

### size-adaptive - refine 600  (A = size_adaptive, B = ft_constraints_refine600)

| subset | metric | A − B | 95% interval | verdict |
|---|---|---|---|---|
| overall (n=7774) | tolerant mIoU | +0.0415 | [+0.0389, +0.0441] | clear: A better |
| overall (n=7774) | connectivity error | -0.0257 | [-0.0315, -0.0206] | clear: A better |
| overall (n=7774) | architectural validity % | +3.9491 | [+3.1773, +4.7472] | clear: A better |
| type = room (n=1555) | tolerant mIoU | +0.0603 | [+0.0548, +0.0658] | clear: A better |
| type = room (n=1555) | connectivity error | -0.0791 | [-0.0990, -0.0611] | clear: A better |
| type = room (n=1555) | architectural validity % | +8.5531 | [+5.5305, +11.3183] | clear: A better |
| size = large (n=2608) | tolerant mIoU | +0.0601 | [+0.0558, +0.0653] | clear: A better |
| size = large (n=2608) | connectivity error | -0.0740 | [-0.0905, -0.0583] | clear: A better |
| size = large (n=2608) | architectural validity % | +9.3942 | [+7.3227, +11.4666] | clear: A better |

### size-adaptive - hybrid 600  (A = size_adaptive, B = hybrid_refine600)

| subset | metric | A − B | 95% interval | verdict |
|---|---|---|---|---|
| overall (n=7774) | tolerant mIoU | +0.0001 | [-0.0007, +0.0009] | not clear (interval includes 0) |
| overall (n=7774) | connectivity error | -0.0109 | [-0.0143, -0.0076] | clear: A better |
| overall (n=7774) | architectural validity % | +0.4759 | [-0.0132, +0.9133] | not clear (interval includes 0) |
| type = room (n=1555) | tolerant mIoU | -0.0019 | [-0.0043, +0.0006] | not clear (interval includes 0) |
| type = room (n=1555) | connectivity error | -0.0277 | [-0.0399, -0.0174] | clear: A better |
| type = room (n=1555) | architectural validity % | +2.7653 | [+0.9646, +4.4389] | clear: A better |
| size = large (n=2608) | tolerant mIoU | -0.0037 | [-0.0056, -0.0020] | clear: A worse |
| size = large (n=2608) | connectivity error | -0.0330 | [-0.0429, -0.0234] | clear: A better |
| size = large (n=2608) | architectural validity % | +2.3006 | [+0.9960, +3.6043] | clear: A better |

