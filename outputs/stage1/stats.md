# Statistics: paired bootstrap over val cases

1000 resamples of the cases (with replacement, seed 0); both methods are scored on the same resampled cases each time. Columns: observed difference A − B and its 95% interval. A difference is **statistically clear** only if the interval excludes 0. For connectivity error, room leak and room count error, LOWER is better.

## Summary: which differences are statistically clear

- **constraints - control** (A = ft_constraints_refine600):
  - overall: A is better on connectivity error, room leak, room count error; worse on nothing; no clear difference on tolerant mIoU, architectural validity %.
  - type = room: A is better on connectivity error, room leak, room count error; worse on nothing; no clear difference on tolerant mIoU, architectural validity %.
  - size = large: A is better on connectivity error, room leak, room count error; worse on nothing; no clear difference on tolerant mIoU, architectural validity %.
- **best diffusion - U-Net** (A = ft_constraints_refine600):
  - overall: A is better on tolerant mIoU, connectivity error, room leak, room count error, architectural validity %; worse on nothing.
  - type = room: A is better on tolerant mIoU, connectivity error, room leak, room count error, architectural validity %; worse on nothing.
  - size = large: A is better on tolerant mIoU, connectivity error, room leak, room count error, architectural validity %; worse on nothing.
- **best diffusion - matched U-Net** (A = ft_constraints_refine600):
  - overall: A is better on connectivity error, room leak, architectural validity %; worse on tolerant mIoU, room count error.
  - type = room: A is better on connectivity error, room leak, architectural validity %; worse on tolerant mIoU, room count error.
  - size = large: A is better on connectivity error, room leak, architectural validity %; worse on tolerant mIoU, room count error.

## constraints - control  (A = ft_constraints_refine600, B = ft_control_refine600)

| subset | metric | A − B | 95% interval | verdict |
|---|---|---|---|---|
| overall (n=7772) | tolerant mIoU | -0.0000 | [-0.0016, +0.0017] | not clear (interval includes 0) |
| overall (n=7772) | connectivity error | -0.0058 | [-0.0111, -0.0004] | clear: A better |
| overall (n=7772) | room leak | -0.0031 | [-0.0051, -0.0011] | clear: A better |
| overall (n=7772) | room count error | -0.0136 | [-0.0205, -0.0068] | clear: A better |
| overall (n=7772) | architectural validity % | -0.0643 | [-0.7077, +0.5665] | not clear (interval includes 0) |
| type = room (n=1555) | tolerant mIoU | -0.0027 | [-0.0062, +0.0006] | not clear (interval includes 0) |
| type = room (n=1555) | connectivity error | -0.0283 | [-0.0489, -0.0077] | clear: A better |
| type = room (n=1555) | room leak | -0.0084 | [-0.0136, -0.0033] | clear: A better |
| type = room (n=1555) | room count error | -0.0553 | [-0.0823, -0.0296] | clear: A better |
| type = room (n=1555) | architectural validity % | -0.5145 | [-3.0225, +1.7379] | not clear (interval includes 0) |
| size = large (n=2585) | tolerant mIoU | -0.0014 | [-0.0041, +0.0010] | not clear (interval includes 0) |
| size = large (n=2585) | connectivity error | -0.0190 | [-0.0352, -0.0035] | clear: A better |
| size = large (n=2585) | room leak | -0.0062 | [-0.0103, -0.0020] | clear: A better |
| size = large (n=2585) | room count error | -0.0360 | [-0.0553, -0.0170] | clear: A better |
| size = large (n=2585) | architectural validity % | -0.8124 | [-2.4768, +1.0058] | not clear (interval includes 0) |

## best diffusion - U-Net  (A = ft_constraints_refine600, B = unet)

| subset | metric | A − B | 95% interval | verdict |
|---|---|---|---|---|
| overall (n=7772) | tolerant mIoU | +0.0267 | [+0.0242, +0.0292] | clear: A better |
| overall (n=7772) | connectivity error | -0.1386 | [-0.1512, -0.1253] | clear: A better |
| overall (n=7772) | room leak | -0.0392 | [-0.0427, -0.0357] | clear: A better |
| overall (n=7772) | room count error | -0.0993 | [-0.1098, -0.0888] | clear: A better |
| overall (n=7772) | architectural validity % | +15.7488 | [+14.7829, +16.7148] | clear: A better |
| type = room (n=1555) | tolerant mIoU | +0.0241 | [+0.0199, +0.0289] | clear: A better |
| type = room (n=1555) | connectivity error | -0.3923 | [-0.4412, -0.3486] | clear: A better |
| type = room (n=1555) | room leak | -0.0656 | [-0.0733, -0.0577] | clear: A better |
| type = room (n=1555) | room count error | -0.2797 | [-0.3132, -0.2431] | clear: A better |
| type = room (n=1555) | architectural validity % | +27.8457 | [+24.6302, +30.8055] | clear: A better |
| size = large (n=2585) | tolerant mIoU | +0.0275 | [+0.0234, +0.0314] | clear: A better |
| size = large (n=2585) | connectivity error | -0.3714 | [-0.4081, -0.3331] | clear: A better |
| size = large (n=2585) | room leak | -0.0752 | [-0.0831, -0.0676] | clear: A better |
| size = large (n=2585) | room count error | -0.2484 | [-0.2774, -0.2220] | clear: A better |
| size = large (n=2585) | architectural validity % | +25.4545 | [+23.2882, +27.5841] | clear: A better |

## best diffusion - matched U-Net  (A = ft_constraints_refine600, B = unet_matched)

| subset | metric | A − B | 95% interval | verdict |
|---|---|---|---|---|
| overall (n=7772) | tolerant mIoU | -0.0450 | [-0.0475, -0.0424] | clear: A worse |
| overall (n=7772) | connectivity error | -0.0526 | [-0.0627, -0.0427] | clear: A better |
| overall (n=7772) | room leak | -0.0134 | [-0.0163, -0.0107] | clear: A better |
| overall (n=7772) | room count error | +0.0265 | [+0.0170, +0.0349] | clear: A worse |
| overall (n=7772) | architectural validity % | +3.2424 | [+2.3417, +4.0916] | clear: A better |
| type = room (n=1555) | tolerant mIoU | -0.0741 | [-0.0796, -0.0688] | clear: A worse |
| type = room (n=1555) | connectivity error | -0.1460 | [-0.1814, -0.1151] | clear: A better |
| type = room (n=1555) | room leak | -0.0205 | [-0.0266, -0.0141] | clear: A better |
| type = room (n=1555) | room count error | +0.0810 | [+0.0489, +0.1151] | clear: A worse |
| type = room (n=1555) | architectural validity % | +5.9164 | [+3.0852, +8.9389] | clear: A better |
| size = large (n=2585) | tolerant mIoU | -0.0724 | [-0.0770, -0.0679] | clear: A worse |
| size = large (n=2585) | connectivity error | -0.1478 | [-0.1768, -0.1207] | clear: A better |
| size = large (n=2585) | room leak | -0.0254 | [-0.0314, -0.0187] | clear: A better |
| size = large (n=2585) | room count error | +0.0754 | [+0.0484, +0.1010] | clear: A worse |
| size = large (n=2585) | architectural validity % | +5.9574 | [+3.5957, +8.2408] | clear: A better |
