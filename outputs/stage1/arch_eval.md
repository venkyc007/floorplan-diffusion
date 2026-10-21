# Architectural evaluation (full val benchmark, 7772 cases)

Day 2 metrics + architectural validity metrics (`src/stage1/eval/arch_metrics.py`); every architectural rate is shown next to the ground truth's own rate on the same cases (`gt_` columns). `+pp` = with post-processing (`src/stage1/postprocess.py`). Per-case results: `arch_eval/per_case/`.

## Post-processing: choice of N (on ft_constraints_refine600)

|  | tmIoU | conn_err | dead_ends_per_case | arch_validity_pct | gt_plans_changed_pct |
|---|---|---|---|---|---|
| ft_constraints_refine600+pp4 | 0.801 | 0.036 | 0.187 | 83.749 | 0.180 |
| ft_constraints_refine600+pp6 | 0.801 | 0.036 | 0.178 | 83.955 | 0.862 |
| ft_constraints_refine600+pp8 | 0.801 | 0.039 | 0.169 | 84.058 | 1.853 |

Chosen N = **6**: highest architectural validity among the N that alter at most 1.0% of REAL plans (the clean-up also removes legitimate short stubs); ties broken by tolerant mIoU.

## Overall: Day 2 metrics

| setting | tmIoU | mIoU | pixel_acc | F1_door | F1_window | F1_room | conn_err | open_on_wall | room_leak | room_count_err | alignment |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nearest_fill | 0.489 | 0.437 | 0.671 | 0.450 | 0.451 | 0.694 | 0.066 | 0.996 | 0.627 | 1.732 | 0.999 |
| unet | 0.775 | 0.708 | 0.897 | 0.603 | 0.617 | 0.929 | 0.178 | 0.974 | 0.065 | 0.301 | 0.990 |
| unet_matched | 0.846 | 0.784 | 0.919 | 0.692 | 0.693 | 0.964 | 0.092 | 0.985 | 0.039 | 0.175 | 0.994 |
| run1_refine600 | 0.784 | 0.728 | 0.893 | 0.681 | 0.669 | 0.943 | 0.039 | 0.999 | 0.027 | 0.231 | 0.999 |
| ft_constraints_refine600 | 0.801 | 0.744 | 0.898 | 0.693 | 0.684 | 0.953 | 0.039 | 1.000 | 0.026 | 0.202 | 0.999 |
| ft_control_refine600 | 0.801 | 0.744 | 0.899 | 0.693 | 0.682 | 0.951 | 0.045 | 1.000 | 0.029 | 0.215 | 0.999 |
| nearest_fill+pp | 0.488 | 0.437 | 0.670 | 0.450 | 0.451 | 0.700 | 0.064 | 0.996 | 0.616 | 1.710 | 0.999 |
| unet+pp | 0.774 | 0.708 | 0.897 | 0.603 | 0.617 | 0.931 | 0.128 | 0.974 | 0.064 | 0.290 | 0.991 |
| unet_matched+pp | 0.846 | 0.784 | 0.919 | 0.692 | 0.693 | 0.965 | 0.070 | 0.985 | 0.039 | 0.170 | 0.994 |
| run1_refine600+pp | 0.784 | 0.728 | 0.893 | 0.681 | 0.669 | 0.944 | 0.036 | 0.999 | 0.027 | 0.226 | 0.999 |
| ft_constraints_refine600+pp | 0.801 | 0.744 | 0.898 | 0.693 | 0.684 | 0.954 | 0.036 | 1.000 | 0.026 | 0.197 | 0.999 |
| ft_control_refine600+pp | 0.801 | 0.744 | 0.899 | 0.693 | 0.682 | 0.951 | 0.042 | 1.000 | 0.029 | 0.212 | 0.999 |

## Overall: architectural metrics (prediction vs ground truth)

| setting | dead_ends_per_case | gt_dead_ends_per_case | fragments_per_case | gt_fragments_per_case | opening_bridge_per_case | gt_opening_bridge_per_case | opening_fit_pct | gt_opening_fit_pct | bridges_pct | thickness_pct | aligned_pct | door_spaces_per_case | gt_door_spaces_per_case | window_out_per_case | gt_window_out_per_case | unreachable_per_case | gt_unreachable_per_case | slivers_per_case | gt_slivers_per_case | rectangularity | gt_rectangularity | arch_validity_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| nearest_fill | 0.580 | 0.132 | 0.002 | 0.000 | 0.639 | 0.038 | 28.357 | 82.004 | 43.885 | 33.055 | 28.515 | 0.039 | 0.016 | 0.018 | 0.009 | 0.012 | 0.012 | 0.015 | 0.044 | 0.763 | 0.864 | 35.962 |
| unet | 0.295 | 0.132 | 0.127 | 0.000 | 0.153 | 0.038 | 75.940 | 82.004 | 92.500 | 82.302 | 77.065 | 0.123 | 0.016 | 0.075 | 0.009 | 0.112 | 0.012 | 0.069 | 0.044 | 0.856 | 0.864 | 67.872 |
| unet_matched | 0.198 | 0.132 | 0.064 | 0.000 | 0.117 | 0.038 | 77.095 | 82.004 | 94.655 | 84.699 | 78.101 | 0.094 | 0.016 | 0.043 | 0.009 | 0.054 | 0.012 | 0.047 | 0.044 | 0.860 | 0.864 | 80.378 |
| run1_refine600 | 0.212 | 0.132 | 0.013 | 0.000 | 0.051 | 0.038 | 83.488 | 82.004 | 97.281 | 91.066 | 83.564 | 0.016 | 0.016 | 0.012 | 0.009 | 0.088 | 0.012 | 0.034 | 0.044 | 0.859 | 0.864 | 81.794 |
| ft_constraints_refine600 | 0.200 | 0.132 | 0.015 | 0.000 | 0.052 | 0.038 | 83.254 | 82.004 | 97.301 | 90.939 | 83.333 | 0.018 | 0.016 | 0.012 | 0.009 | 0.080 | 0.012 | 0.037 | 0.044 | 0.860 | 0.864 | 83.621 |
| ft_control_refine600 | 0.206 | 0.132 | 0.017 | 0.000 | 0.054 | 0.038 | 82.878 | 82.004 | 97.241 | 90.968 | 82.958 | 0.016 | 0.016 | 0.013 | 0.009 | 0.074 | 0.012 | 0.036 | 0.044 | 0.860 | 0.864 | 83.685 |
| nearest_fill+pp | 0.529 | 0.132 | 0.005 | 0.000 | 0.640 | 0.038 | 28.342 | 82.004 | 43.846 | 33.051 | 28.500 | 0.039 | 0.016 | 0.018 | 0.009 | 0.019 | 0.012 | 0.016 | 0.044 | 0.765 | 0.864 | 36.979 |
| unet+pp | 0.187 | 0.132 | 0.088 | 0.000 | 0.154 | 0.038 | 75.859 | 82.004 | 92.418 | 82.228 | 76.978 | 0.123 | 0.016 | 0.075 | 0.009 | 0.119 | 0.012 | 0.071 | 0.044 | 0.856 | 0.864 | 69.056 |
| unet_matched+pp | 0.146 | 0.132 | 0.047 | 0.000 | 0.118 | 0.038 | 77.047 | 82.004 | 94.619 | 84.668 | 78.052 | 0.094 | 0.016 | 0.043 | 0.009 | 0.056 | 0.012 | 0.048 | 0.044 | 0.860 | 0.864 | 80.932 |
| run1_refine600+pp | 0.185 | 0.132 | 0.016 | 0.000 | 0.052 | 0.038 | 83.426 | 82.004 | 97.206 | 91.011 | 83.502 | 0.016 | 0.016 | 0.012 | 0.009 | 0.090 | 0.012 | 0.035 | 0.044 | 0.859 | 0.864 | 82.347 |
| ft_constraints_refine600+pp | 0.178 | 0.132 | 0.017 | 0.000 | 0.053 | 0.038 | 83.227 | 82.004 | 97.274 | 90.919 | 83.307 | 0.018 | 0.016 | 0.012 | 0.009 | 0.082 | 0.012 | 0.037 | 0.044 | 0.860 | 0.864 | 83.955 |
| ft_control_refine600+pp | 0.181 | 0.132 | 0.019 | 0.000 | 0.054 | 0.038 | 82.812 | 82.004 | 97.195 | 90.915 | 82.891 | 0.016 | 0.016 | 0.013 | 0.009 | 0.076 | 0.012 | 0.037 | 0.044 | 0.860 | 0.864 | 84.020 |

## type=door

| setting | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| nearest_fill | 0.546 | 0.029 | 0.320 | 1.299 | 1.028 | 32.841 |
| unet | 0.885 | 0.019 | 0.008 | 0.052 | 0.049 | 81.555 |
| unet_matched | 0.950 | 0.005 | 0.006 | 0.025 | 0.044 | 92.674 |
| run1_refine600 | 0.904 | 0.001 | 0.005 | 0.034 | 0.045 | 94.537 |
| ft_constraints_refine600 | 0.917 | 0.001 | 0.005 | 0.031 | 0.044 | 95.630 |
| ft_control_refine600 | 0.916 | 0.001 | 0.005 | 0.034 | 0.047 | 94.923 |
| nearest_fill+pp | 0.543 | 0.024 | 0.305 | 1.240 | 0.897 | 35.026 |
| unet+pp | 0.885 | 0.017 | 0.008 | 0.051 | 0.043 | 81.748 |
| unet_matched+pp | 0.950 | 0.006 | 0.006 | 0.025 | 0.045 | 92.609 |
| run1_refine600+pp | 0.904 | 0.001 | 0.005 | 0.033 | 0.044 | 94.537 |
| ft_constraints_refine600+pp | 0.917 | 0.001 | 0.005 | 0.032 | 0.044 | 95.630 |
| ft_control_refine600+pp | 0.916 | 0.001 | 0.005 | 0.034 | 0.046 | 94.859 |

## type=region

| setting | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| nearest_fill | 0.524 | 0.071 | 0.467 | 1.487 | 0.841 | 36.375 |
| unet | 0.686 | 0.352 | 0.140 | 0.603 | 0.650 | 52.314 |
| unet_matched | 0.776 | 0.204 | 0.089 | 0.392 | 0.474 | 66.645 |
| run1_refine600 | 0.698 | 0.115 | 0.068 | 0.499 | 0.463 | 70.437 |
| ft_constraints_refine600 | 0.719 | 0.098 | 0.060 | 0.444 | 0.454 | 71.208 |
| ft_control_refine600 | 0.719 | 0.099 | 0.063 | 0.447 | 0.463 | 71.851 |
| nearest_fill+pp | 0.523 | 0.070 | 0.460 | 1.466 | 0.794 | 36.825 |
| unet+pp | 0.686 | 0.238 | 0.138 | 0.594 | 0.407 | 54.756 |
| unet_matched+pp | 0.776 | 0.148 | 0.088 | 0.384 | 0.344 | 67.995 |
| run1_refine600+pp | 0.698 | 0.105 | 0.068 | 0.494 | 0.409 | 71.530 |
| ft_constraints_refine600+pp | 0.719 | 0.094 | 0.060 | 0.435 | 0.400 | 72.044 |
| ft_control_refine600+pp | 0.719 | 0.096 | 0.062 | 0.444 | 0.398 | 72.622 |

## type=room

| setting | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| nearest_fill | 0.271 | 0.158 | 0.908 | 3.352 | 0.621 | 11.318 |
| unet | 0.582 | 0.486 | 0.105 | 0.795 | 0.636 | 28.939 |
| unet_matched | 0.680 | 0.239 | 0.060 | 0.434 | 0.357 | 50.868 |
| run1_refine600 | 0.584 | 0.077 | 0.039 | 0.594 | 0.447 | 49.518 |
| ft_constraints_refine600 | 0.606 | 0.093 | 0.040 | 0.515 | 0.400 | 56.785 |
| ft_control_refine600 | 0.608 | 0.122 | 0.048 | 0.570 | 0.413 | 57.299 |
| nearest_fill+pp | 0.271 | 0.157 | 0.902 | 3.343 | 0.596 | 11.833 |
| unet+pp | 0.581 | 0.361 | 0.104 | 0.749 | 0.372 | 31.576 |
| unet_matched+pp | 0.680 | 0.189 | 0.060 | 0.417 | 0.233 | 52.219 |
| run1_refine600+pp | 0.584 | 0.069 | 0.039 | 0.577 | 0.370 | 51.190 |
| ft_constraints_refine600+pp | 0.606 | 0.083 | 0.040 | 0.502 | 0.344 | 57.621 |
| ft_control_refine600+pp | 0.608 | 0.112 | 0.048 | 0.559 | 0.356 | 58.264 |

## type=wall

| setting | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| nearest_fill | 0.592 | 0.051 | 0.624 | 1.359 | 0.300 | 23.586 |
| unet | 0.876 | 0.022 | 0.012 | 0.031 | 0.121 | 89.781 |
| unet_matched | 0.923 | 0.007 | 0.008 | 0.017 | 0.098 | 95.823 |
| run1_refine600 | 0.887 | 0.003 | 0.005 | 0.020 | 0.091 | 97.172 |
| ft_constraints_refine600 | 0.897 | 0.003 | 0.004 | 0.012 | 0.089 | 97.494 |
| ft_control_refine600 | 0.894 | 0.002 | 0.005 | 0.018 | 0.091 | 96.979 |
| nearest_fill+pp | 0.591 | 0.049 | 0.620 | 1.352 | 0.266 | 24.293 |
| unet+pp | 0.876 | 0.013 | 0.012 | 0.029 | 0.099 | 90.424 |
| unet_matched+pp | 0.922 | 0.007 | 0.008 | 0.015 | 0.094 | 95.951 |
| run1_refine600+pp | 0.887 | 0.004 | 0.005 | 0.019 | 0.088 | 97.172 |
| ft_constraints_refine600+pp | 0.897 | 0.002 | 0.004 | 0.012 | 0.087 | 97.494 |
| ft_control_refine600+pp | 0.893 | 0.003 | 0.005 | 0.018 | 0.089 | 96.979 |

## type=window

| setting | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| nearest_fill | 0.513 | 0.019 | 0.828 | 1.163 | 0.106 | 75.855 |
| unet | 0.844 | 0.009 | 0.024 | 0.023 | 0.015 | 86.830 |
| unet_matched | 0.903 | 0.002 | 0.007 | 0.007 | 0.014 | 95.933 |
| run1_refine600 | 0.848 | 0.000 | 0.004 | 0.007 | 0.014 | 97.353 |
| ft_constraints_refine600 | 0.867 | 0.000 | 0.003 | 0.005 | 0.014 | 97.030 |
| ft_control_refine600 | 0.869 | 0.000 | 0.004 | 0.006 | 0.014 | 97.418 |
| nearest_fill+pp | 0.513 | 0.019 | 0.819 | 1.149 | 0.090 | 77.082 |
| unet+pp | 0.844 | 0.010 | 0.024 | 0.023 | 0.015 | 86.830 |
| unet_matched+pp | 0.903 | 0.002 | 0.007 | 0.006 | 0.013 | 95.933 |
| run1_refine600+pp | 0.848 | 0.000 | 0.004 | 0.007 | 0.014 | 97.353 |
| ft_constraints_refine600+pp | 0.867 | 0.000 | 0.003 | 0.005 | 0.014 | 97.030 |
| ft_control_refine600+pp | 0.869 | 0.000 | 0.004 | 0.006 | 0.014 | 97.418 |

## size_bin=large

| setting | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| nearest_fill | 0.346 | 0.132 | 0.755 | 2.744 | 0.790 | 16.634 |
| unet | 0.601 | 0.483 | 0.128 | 0.805 | 0.735 | 33.810 |
| unet_matched | 0.701 | 0.260 | 0.078 | 0.481 | 0.472 | 53.308 |
| run1_refine600 | 0.605 | 0.113 | 0.056 | 0.634 | 0.521 | 54.662 |
| ft_constraints_refine600 | 0.629 | 0.112 | 0.053 | 0.556 | 0.487 | 59.265 |
| ft_control_refine600 | 0.630 | 0.131 | 0.059 | 0.592 | 0.497 | 60.077 |
| nearest_fill+pp | 0.346 | 0.131 | 0.746 | 2.728 | 0.757 | 17.060 |
| unet+pp | 0.601 | 0.347 | 0.126 | 0.773 | 0.443 | 36.634 |
| unet_matched+pp | 0.701 | 0.198 | 0.078 | 0.464 | 0.323 | 54.855 |
| run1_refine600+pp | 0.605 | 0.103 | 0.056 | 0.623 | 0.446 | 56.209 |
| ft_constraints_refine600+pp | 0.629 | 0.104 | 0.053 | 0.544 | 0.423 | 60.271 |
| ft_control_refine600+pp | 0.630 | 0.124 | 0.059 | 0.584 | 0.428 | 61.006 |

## size_bin=medium

| setting | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| nearest_fill | 0.569 | 0.042 | 0.569 | 1.304 | 0.453 | 43.375 |
| unet | 0.850 | 0.045 | 0.024 | 0.080 | 0.108 | 84.030 |
| unet_matched | 0.907 | 0.014 | 0.009 | 0.035 | 0.082 | 92.981 |
| run1_refine600 | 0.863 | 0.005 | 0.005 | 0.049 | 0.079 | 93.770 |
| ft_constraints_refine600 | 0.876 | 0.005 | 0.005 | 0.041 | 0.077 | 94.361 |
| ft_control_refine600 | 0.875 | 0.004 | 0.006 | 0.045 | 0.083 | 94.322 |
| nearest_fill+pp | 0.568 | 0.040 | 0.561 | 1.284 | 0.403 | 44.282 |
| unet+pp | 0.850 | 0.031 | 0.024 | 0.077 | 0.080 | 84.740 |
| unet_matched+pp | 0.907 | 0.013 | 0.009 | 0.035 | 0.078 | 93.139 |
| run1_refine600+pp | 0.862 | 0.004 | 0.005 | 0.047 | 0.074 | 93.849 |
| ft_constraints_refine600+pp | 0.876 | 0.005 | 0.005 | 0.040 | 0.074 | 94.361 |
| ft_control_refine600+pp | 0.874 | 0.004 | 0.006 | 0.045 | 0.078 | 94.401 |

## size_bin=small

| setting | tmIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|
| nearest_fill | 0.552 | 0.024 | 0.546 | 1.156 | 0.496 | 47.718 |
| unet | 0.871 | 0.007 | 0.011 | 0.022 | 0.043 | 85.628 |
| unet_matched | 0.929 | 0.002 | 0.007 | 0.011 | 0.040 | 94.719 |
| run1_refine600 | 0.884 | 0.001 | 0.005 | 0.012 | 0.039 | 96.794 |
| ft_constraints_refine600 | 0.898 | 0.000 | 0.005 | 0.009 | 0.039 | 97.095 |
| ft_control_refine600 | 0.898 | 0.000 | 0.005 | 0.010 | 0.039 | 96.530 |
| nearest_fill+pp | 0.550 | 0.021 | 0.532 | 1.126 | 0.427 | 49.415 |
| unet+pp | 0.871 | 0.007 | 0.011 | 0.021 | 0.041 | 85.666 |
| unet_matched+pp | 0.929 | 0.002 | 0.007 | 0.011 | 0.038 | 94.681 |
| run1_refine600+pp | 0.884 | 0.001 | 0.005 | 0.011 | 0.037 | 96.831 |
| ft_constraints_refine600+pp | 0.898 | 0.000 | 0.005 | 0.009 | 0.038 | 97.095 |
| ft_control_refine600+pp | 0.898 | 0.000 | 0.005 | 0.011 | 0.038 | 96.530 |

## Refine-level sweep (constraint fine-tune)

| setting | tmIoU | mIoU | conn_err | room_leak | room_count_err | dead_ends_per_case | arch_validity_pct |
|---|---|---|---|---|---|---|---|
| ft_constraints_refine200 | 0.801 | 0.744 | 0.047 | 0.033 | 0.222 | 0.217 | 81.433 |
| ft_constraints_refine400 | 0.803 | 0.745 | 0.055 | 0.031 | 0.216 | 0.224 | 81.845 |
| ft_constraints_refine600 | 0.801 | 0.744 | 0.039 | 0.026 | 0.202 | 0.200 | 83.621 |
| ft_constraints_refine800 | 0.792 | 0.734 | 0.008 | 0.008 | 0.147 | 0.163 | 86.310 |
| ft_constraints_ddim50 | 0.761 | 0.702 | 0.005 | 0.005 | 0.120 | 0.146 | 86.696 |

Figure: `refine_tradeoff.png`.
