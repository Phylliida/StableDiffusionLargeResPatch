# StableDiffusionLargeResPatch
Patch for stable diffusion that let you run it at large resolutions (up to 2992x2992 on 8GB VRAM)

Using this with diffusers is simple:

```
import largeResPatch
with largeResPatch.LargeResPatch(pipe=pipe, width=width, height=height):
    res = pipe([prompt], width=width, height=height)
```

This does a few things:
1. Patches `diffusers.models.attention.CrossAttention.forward` to process the attention in blocks (see `attn_slice_x`, `attn_slice_y`, and `attn_shape` parameters below). This will give different results than the unpatched code because blocks can't see outside themselves, but it still ends up being cohesive. Note that this is only ran for the expensive attention (key dim[1] != 77)
2. Patches` diffusers.models.attention.BasicTransformerBlock.forward` so the `self.ff(...)` call is processed in pieces. This will give identical results to the unpatched code, it's just less memory intensive at the cost of taking a little longer. `ff_chunk_size` specifies the chunk size.
3. Patches `diffusers.models.vae.AutoencoderKL.decode` and  `diffusers.models.vae.AutoencoderKL.encode` to process in patches. Because group norms will be different for each patch, we run two passes: the first pass just stores the output of each of the group norms for each patch. The second pass uses the average statistics over all the patches for the group norms. While this isn't equivalent, in practice it seems to work alright. `vae_chunk_size_x` and `vae_chunk_size_y` can be used to specify how large the patches are.

The `attn_shape` parameter can be `brick` or `tile`. The `brick` mode alternates between these four tilings, with bricks of size `attn_slice_x*2, attn_slice_y` for horizontal and `attn_slice_x, attn_slice_y*2` for vertical. I chose these tilings because it ensures that the information can eventually trickle throughout the whole image.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="fff.png">
  <source media="(prefers-color-scheme: light)" srcset="fff.png">
  <img alt="Brick tiling patterns" src="fff.png">
</picture>

The `tile` mode is in progress and not recommended, right now it just does the trivial thing of making `attn_slice_x, attn_slice_y` patches. I'm going to make it offset them every other input, which should help with some artifacts.

Note, width and height need to be divisible by 16 (TODO: fix this).
