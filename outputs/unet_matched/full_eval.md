# Full val-benchmark evaluation (7772 cases)

## Overall

| setting | n | mIoU | tmIoU | IoU_wall | tIoU_wall | IoU_door | tIoU_door | IoU_window | tIoU_window | IoU_room | IoU_outdoor | pixel_acc | F1_door | F1_window | F1_room | conn_err | open_on_wall | room_leak | room_count_err | alignment | sec_per_case |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| unet_matched | 7772 | 0.784 | 0.846 | 0.692 | 0.813 | 0.556 | 0.657 | 0.497 | 0.593 | 0.933 | 0.772 | 0.919 | 0.692 | 0.693 | 0.964 | 0.092 | 0.985 | 0.039 | 0.175 | 0.994 | 0.011 |

## tmIoU by type / location / position / size

| group | unet_matched |
|---|---|
| overall | 0.846 |
| type=door | 0.950 |
| type=region | 0.776 |
| type=room | 0.680 |
| type=wall | 0.923 |
| type=window | 0.903 |
| location=exterior | 0.812 |
| location=interior | 0.942 |
| position=center | 0.848 |
| position=edge | 0.845 |
| size_bin=large | 0.701 |
| size_bin=medium | 0.907 |
| size_bin=small | 0.929 |

## mIoU by type / location / position / size

| group | unet_matched |
|---|---|
| overall | 0.784 |
| type=door | 0.889 |
| type=region | 0.736 |
| type=room | 0.599 |
| type=wall | 0.868 |
| type=window | 0.827 |
| location=exterior | 0.746 |
| location=interior | 0.890 |
| position=center | 0.786 |
| position=edge | 0.781 |
| size_bin=large | 0.635 |
| size_bin=medium | 0.851 |
| size_bin=small | 0.864 |

## F1_door by type / location / position / size

| group | unet_matched |
|---|---|
| overall | 0.692 |
| type=door | 0.905 |
| type=region | 0.604 |
| type=room | 0.529 |
| type=wall | 0.955 |
| type=window | 0.830 |
| location=exterior | 0.617 |
| location=interior | 0.865 |
| position=center | 0.698 |
| position=edge | 0.681 |
| size_bin=large | 0.543 |
| size_bin=medium | 0.878 |
| size_bin=small | 0.918 |

## F1_window by type / location / position / size

| group | unet_matched |
|---|---|
| overall | 0.693 |
| type=door | 0.764 |
| type=region | 0.584 |
| type=room | 0.526 |
| type=wall | 0.890 |
| type=window | 0.893 |
| location=exterior | 0.686 |
| location=interior | 0.834 |
| position=center | 0.646 |
| position=edge | 0.735 |
| size_bin=large | 0.538 |
| size_bin=medium | 0.847 |
| size_bin=small | 0.885 |

## F1_room by type / location / position / size

| group | unet_matched |
|---|---|
| overall | 0.964 |
| type=door | 0.995 |
| type=region | 0.905 |
| type=room | 0.951 |
| type=wall | 0.994 |
| type=window | 0.998 |
| location=exterior | 0.954 |
| location=interior | 0.986 |
| position=center | 0.960 |
| position=edge | 0.970 |
| size_bin=large | 0.928 |
| size_bin=medium | 0.990 |
| size_bin=small | 0.997 |

## conn_err by type / location / position / size

| group | unet_matched |
|---|---|
| overall | 0.092 |
| type=door | 0.005 |
| type=region | 0.204 |
| type=room | 0.239 |
| type=wall | 0.007 |
| type=window | 0.002 |
| location=exterior | 0.116 |
| location=interior | 0.022 |
| position=center | 0.116 |
| position=edge | 0.066 |
| size_bin=large | 0.260 |
| size_bin=medium | 0.014 |
| size_bin=small | 0.002 |

## room_leak by type / location / position / size

| group | unet_matched |
|---|---|
| overall | 0.039 |
| type=door | 0.006 |
| type=region | 0.089 |
| type=room | 0.060 |
| type=wall | 0.008 |
| type=window | 0.007 |
| location=exterior | 0.054 |
| location=interior | 0.004 |
| position=center | 0.035 |
| position=edge | 0.046 |
| size_bin=large | 0.078 |
| size_bin=medium | 0.009 |
| size_bin=small | 0.007 |

## room_count_err by type / location / position / size

| group | unet_matched |
|---|---|
| overall | 0.175 |
| type=door | 0.025 |
| type=region | 0.392 |
| type=room | 0.434 |
| type=wall | 0.017 |
| type=window | 0.007 |
| location=exterior | 0.215 |
| location=interior | 0.061 |
| position=center | 0.223 |
| position=edge | 0.124 |
| size_bin=large | 0.481 |
| size_bin=medium | 0.035 |
| size_bin=small | 0.011 |
