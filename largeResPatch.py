import diffusers
from einops import rearrange
from torch import einsum
import torch.nn.functional as F
import math
from diffusers.models.vae import DecoderOutput, AutoencoderKLOutput, DiagonalGaussianDistribution
import torch
from collections import defaultdict
import numpy as np
  

def callWithModifiedGroupNorm(model, func, input):
  '''
  This is a hack to let us chunk things and yet have the same group norm statistics
  for every chunk
  
  This calls func once, and each time a groupnorm is called it stores the mean and var
  (it's assumed that you are chunking, so groupnorms will be called multiple times)
  
  Then func is called again, and the mean and vars used in each group norm are averages
  of the results found in the previous call to func
  '''
  
  original_forward = torch.nn.GroupNorm.forward
  
  def patchedGroupNormForward(self, input, *args, **kwargs):
    storing, use_stored = hasattr(self, "store_outputs"), hasattr(self, 'use_stored_outputs')
    if storing or use_stored:
      group_size = self.num_channels//self.num_groups
      all_output = torch.zeros(input.size(), device=input.device, dtype=input.dtype)
      for i, group_start in enumerate(range(0, self.num_channels, group_size)):
        group_end = min(group_start+group_size, self.num_channels)
        group = input[:,group_start:group_end]
        if use_stored:
          mean = torch.tensor(np.mean(self.means[i]))
          var = torch.tensor(np.mean(self.vars[i]))
        else:
          mean = torch.mean(group)
          var = torch.var(group, unbiased=False)
          self.means[i].append(float(mean))
          self.vars[i].append(float(var))
        output = (group - mean)/torch.sqrt(var+self.eps)
        all_output[:,group_start:group_end] = output
        del output
        del group
      if self.affine:
        affine_output = torch.einsum('i j k l, j -> i j k l', all_output, self.weight) + self.bias.view(1,-1,1,1)
        del all_output
        all_output = affine_output
      return all_output
    else:
      return original_forward(self, input, *args, **kwargs)
  
  # patch group norm
  torch.nn.GroupNorm.forward = patchedGroupNormForward
  for m in model.modules():
    if type(m) is torch.nn.GroupNorm:
      m.store_outputs = True
      if hasattr(m, 'use_stored_outputs'):
        del m.use_stored_outputs
      m.means = defaultdict(lambda: [])
      m.vars = defaultdict(lambda: [])
  # run the first time and store group norm statistics
  output = func(input)
  del output
  for m in model.modules():
    if type(m) is torch.nn.GroupNorm:
      del m.store_outputs
      m.use_stored_outputs = True
  # run the second time, use group norm statistics
  output = func(input)
  for m in model.modules():
    if type(m) is torch.nn.GroupNorm:
      del m.use_stored_outputs
      del m.means
      del m.vars
  # revert group norm back to normal
  torch.nn.GroupNorm.forward = original_forward
  return output


def chunkedVaeDecode(vae, input, padding_mode='circular', chunk_size_x=64, chunk_size_y=64):
  '''
  Splits up vae decoding into (chunkSizeX,chunkSizeY) sized chunks
  Each chunk is ran through through the vae
  Normally this should be fine because convs are very local operations
  However, group norms will get different results for each chunk,
  causing saturation and etc. of different chunks to be different
  
  To address this, we call the vae twice:
  - the first time, each chunk stores its group norm statistics
  - the second time, each chunk uses the average statistics of all the chunks
  
  This is *not* equivalent to calling the vae without chunking, but in practice
  the outputs are good enough
  
  The proper way to do this would be to
  1. Run every chunk up to the first group norm, store statistics
  2. Run every chunk up to the first group norm again, use average statistics
  3. Continue along until the second group norm, store statistics
  4. Run every chunk up to the second norm again, use average statistics
  5. Continue along until the third group norm, store statistics
  5. ... for each group norm (there's 30 of them)
  That would be very slow, so this alternative method is a nice compromise option
  '''
  self = vae
  z = self.post_quant_conv(input)
  del input
  b,c,w,h = map(int, z.size())
  # no need to chunk, we are small enough
  if w <= chunk_size_x and h <= chunk_size_y and False:
    output = vae.decoder(z)
  else:
    def chunkedVae(input):
      mult = 8
      padding = 4
      # so we don't need to bother with boundary conditions
      # it's important to pad and include some of the neighbor patch so the boundaries line up
      padded_input = torch.nn.functional.pad(input, (padding,padding,padding,padding), padding_mode)
      del input
      output = torch.zeros([b,3,w*mult,h*mult],device=padded_input.device, dtype=padded_input.dtype)
      for x_start in range(0, w, chunk_size_x):
        for y_start in range(0, h, chunk_size_y):
          x_end = min(x_start+chunk_size_x, w)
          y_end = min(y_start+chunk_size_y, h)
          x_chunk_start = x_start
          y_chunk_start = y_start
          x_chunk_end = x_end + padding*2
          y_chunk_end = y_end + padding*2
          x_start_padding = padding
          y_start_padding = padding
          if padding_mode != 'circular':
            # if we are at the edge, don't include extra padded stuff
            # this prevents the brown colors seeping in because we have zero padding in our buffer
            if x_start == 0:
              x_chunk_start += padding
              x_start_padding = 0
            if y_start == 0:
              y_chunk_start += padding
              y_start_padding = 0
            if x_end == w: x_chunk_end = w+padding
            if y_end == h: y_chunk_end = h+padding
          chunk = padded_input[:,:,x_chunk_start:x_chunk_end,y_chunk_start:y_chunk_end]
          piece_output = vae.decoder(chunk)
          del chunk
          output_x_start = x_start_padding*mult
          output_y_start = y_start_padding*mult
          output_x_end = output_x_start + (x_end-x_start)*mult
          output_y_end = output_y_start + (y_end-y_start)*mult
          #print(output_x_start, output_x_end, output_y_start, output_y_end, pieceOutput.size(), store_x_start, store_x_end, store_y_start, store_y_end)
          output[:,:,x_start*mult:x_end*mult,y_start*mult:y_end*mult] = piece_output[:,:,output_x_start:output_x_end,output_y_start:output_y_end]
          del piece_output
      del padded_input
      return output
    output = callWithModifiedGroupNorm(vae, chunkedVae, z)
  del z
  return output
  
def chunkedVaeEncode(vae, input, padding_mode='circular', chunk_size_x=64*8, chunk_size_y=64*8):
  '''
  Splits up vae encoding into (chunkSizeX,chunkSizeY) sized chunks
  Each chunk is ran through through the vae
  Normally this should be fine because convs are very local operations
  However, group norms will get different results for each chunk,
  causing saturation and etc. of different chunks to be different
  
  To address this, we call the vae twice:
  - the first time, each chunk stores its group norm statistics
  - the second time, each chunk uses the average statistics of all the chunks
  
  This is *not* equivalent to calling the vae without chunking, but in practice
  the outputs are good enough
  
  The proper way to do this would be to
  1. Run every chunk up to the first group norm, store statistics
  2. Run every chunk up to the first group norm again, use average statistics
  3. Continue along until the second group norm, store statistics
  4. Run every chunk up to the second norm again, use average statistics
  5. Continue along until the third group norm, store statistics
  5. ... for each group norm (there's 30 of them)
  That would be very slow, so this alternative method is a nice compromise option
  '''
  self = vae
  z = input
  del input
  b,c,w,h = map(int, z.size())
  # no need to chunk, we are small enough
  if w <= chunk_size_x and h <= chunk_size_y and False:
    output = vae.decoder(z)
  else:
    def chunkedVae(input):
      mult = 8
      padding = 4
      # so we don't need to bother with boundary conditions
      # it's important to pad and include some of the neighbor patch so the boundaries line up
      padded_input = torch.nn.functional.pad(input, (padding,padding,padding,padding), padding_mode)
      del input
      output = torch.zeros([b,8,w//mult,h//mult],device=padded_input.device, dtype=padded_input.dtype)
      for x_start in range(0, w, chunk_size_x):
        for y_start in range(0, h, chunk_size_y):
          x_end = min(x_start+chunk_size_x, w)
          y_end = min(y_start+chunk_size_y, h)
          x_chunk_start = x_start
          y_chunk_start = y_start
          x_chunk_end = x_end + padding*2
          y_chunk_end = y_end + padding*2
          x_start_padding = padding
          y_start_padding = padding
          if padding_mode != 'circular':
            # if we are at the edge, don't include extra padded stuff
            # this prevents the brown colors seeping in because we have zero padding in our buffer
            if x_start == 0:
              x_chunk_start += padding
              x_start_padding = 0
            if y_start == 0:
              y_chunk_start += padding
              y_start_padding = 0
            if x_end == w: x_chunk_end = w+padding
            if y_end == h: y_chunk_end = h+padding
          chunk = padded_input[:,:,x_chunk_start:x_chunk_end,y_chunk_start:y_chunk_end]
          piece_output = vae.encoder(chunk)
          del chunk
          output_x_start = x_start_padding//mult
          output_y_start = y_start_padding//mult
          output_x_end = output_x_start + (x_end-x_start)//mult
          output_y_end = output_y_start + (y_end-y_start)//mult
          #print(output_x_start, output_x_end, output_y_start, output_y_end, pieceOutput.size(), store_x_start, store_x_end, store_y_start, store_y_end)
          output[:,:,x_start//mult:x_end//mult,y_start//mult:y_end//mult] = piece_output[:,:,output_x_start:output_x_end,output_y_start:output_y_end]
          del piece_output
      del padded_input
      return output
    output = callWithModifiedGroupNorm(vae, chunkedVae, z)
  del z
  return output
  
# this is useful to see what the bricks look like
# for example try viewBricks(256, 256, 32, 32, "brick")
def viewBricks(width, height, attn_slice_x, attn_slice_y, outPath):
  import PIL.Image
  import mimetypes
  if not '.' in outPath or not ('image/' in mimetypes.guess_type(outPath)[0]):
    outPath = outPath + ".png"
  verticalPadding = 20
  res = np.zeros([height*4+verticalPadding*4, width, 3], dtype=np.uint8)
  res[:] = 255
  for ind in range(4):
    output = np.zeros([height, width, 3])
    # checkboard to let us see if we missed something, from https://stackoverflow.com/a/51715491/2924421
    output[np.indices([height, width]).sum(axis=0)%2 == 0] = 1.0
    for (x_start, y_start, x_end, y_end) in getBricks(width, height, ind, attn_slice_x, attn_slice_y): 
      color = np.random.random([3])
      output[y_start:y_end,x_start:x_end] = color
    y_start = ind*height+ind*verticalPadding
    y_end = y_start + height
    res[y_start:y_end,:] = (output*255).astype(np.uint8)
  PIL.Image.fromarray(res).save(outPath)
      
      
# todo: interlacing squares (grak idea)
      
def getBricks(width, height, ind, attn_slice_x, attn_slice_y):
  ## TODO: make bricks equivalent area
  
  ind = (ind % 4)
  
  usingWide = (ind % 2 == 0)
  
  patch_width = attn_slice_x*2 if usingWide else attn_slice_x
  patch_height = attn_slice_y*2 if not usingWide else attn_slice_y
  
  
  # to prevent tiny slice regions (if it's not divisible),
  # instead we will split the remainder into two segments
  # and we will have two half patches + remainder
  
  patch_dim_size = width if usingWide else height
  patch_size = patch_width if usingWide else patch_height
  other_dim_size = height if usingWide else width
  other_size = patch_height if usingWide else patch_width
  num_rows = int(math.ceil(other_dim_size/float(other_size)))
  
  # if we only have one item, this is very simple
  if patch_size >= patch_dim_size:
    configurations = [[patch_dim_size]]
  # we have more than one item, need to do complicated stuff
  else:
    num_patches = patch_dim_size//patch_size
    extra_stuff = patch_dim_size % patch_size
    patches = [patch_size]*num_patches
    bonus_patches = []
    # if the remainder is smaller than half a patch,
    # we split one of our patches into two pieces
    # and add half the remainder to each
    if extra_stuff < patch_size // 2:
      patches.pop(0)
      extra_piece_1 = extra_stuff//2
      extra_piece_2 = extra_stuff - extra_piece_1
      extra_patch_2 = patch_size//2
      extra_patch_1 = patch_size-extra_patch_2
      bonus_patches.append(extra_piece_1+extra_patch_1)
      bonus_patches.append(extra_piece_2+extra_patch_2)
    # if the remainder is larger than half a patch,
    # simply split it into two pieces
    else:
      extra_piece_1 = extra_stuff//2
      extra_piece_2 = extra_stuff - extra_piece_1
      bonus_patches.append(extra_piece_1)
      bonus_patches.append(extra_piece_2)
      
    configurations = [[bonus_patches[0]] + patches + [bonus_patches[1]]]
    for i in range(1, len(patches)):
      configurations.append(patches[:i] + bonus_patches + patches[i:])
      if i != len(patches)-1:
        configurations.append([bonus_patches[0]] + patches + [bonus_patches[1]])
    
  # alternate between configurations
  rowConfigurations = []
  for r in range(num_rows):
    rowConfigurations.append(configurations[r % len(configurations)])
  
  
  for other_start in range(0, other_dim_size, other_size):
    other_end = min(other_start + other_size, other_dim_size)
    rowI = other_start // other_size
    configuration = rowConfigurations[rowI]
    cur_start = 0
    for patch_size in configuration:
      cur_end = cur_start + patch_size
      
      if usingWide:
        (x_start, y_start, x_end, y_end) = (cur_start, other_start, cur_end, other_end)
      else:
        (x_start, y_start, x_end, y_end) = (other_start, cur_start, other_end, cur_end)
      
      # num 2 and 3 get flipped to ensure the thin remaining line of bricks goes on a different side
      flip = ind // 2 == 1
      if flip:
        (x_start, y_start, x_end, y_end) = (width-x_end, height-y_end, width-x_start, height-y_start)
      
      yield (x_start, y_start, x_end, y_end)
      
      cur_start = cur_end
      
def getTiles(width, height, attn_slice_x, attn_slice_y):
  '''
  Breaks up (width, height) into tiles of size attn_slice_x, attn_slice_y
  Say we have width = 70
  And attn_slice_x = 32
  Instead of slices
  [0,32], [32,64], [64,70]
  We will do slices
  [0,32], [32,64], [38,70]
  Note these slices overlap, that is intentional
  '''
  for x_start in range(0, width, attn_slice_x):
    for y_start in range(0, height, attn_slice_y):
      x_end = min(x_start+attn_slice_x, width)
      y_end = min(y_start+attn_slice_y, height)
      # if we are clipped at the end, do a larger patch overlapping with
      # the previous patch instead. This will overwrite that previous patch's values
      # but that's ok
      if (x_end-x_start) < attn_slice_x:
        x_start = max(0, x_end - attn_slice_x)
      if (y_end-y_start) < attn_slice_y:
        y_start = max(0, y_end - attn_slice_y)
      yield(x_start, y_start, x_end, y_end) 
  
class LargeResPatch(object):
  def __init__(self, pipe, width, height, attn_slice_x=64, attn_slice_y=64, attn_shape='brick', ff_chunk_size=16384//4, padding_mode='constant', vae_chunk_size_x=64, vae_chunk_size_y=64):
    self.width = width
    self.height = height
    self.attn_slice_x = attn_slice_x
    self.attn_slice_y = attn_slice_y
    self.attn_shape = attn_shape # can be 'brick' or 'tile'
    self.ff_chunk_size = ff_chunk_size
    self.padding_mode = padding_mode
    self.vae_chunk_size_x = vae_chunk_size_x
    self.vae_chunk_size_y = vae_chunk_size_y
    self.pipe = pipe
    
    
    for m in pipe.unet.modules():
      if type(m) is diffusers.models.attention.CrossAttention:
        m.brickInd = 0
    
    
  def __enter__(self):
    width, height, attn_slice_x, attn_slice_y, attn_shape, ff_chunk_size, padding_mode, vae_chunk_size_x, vae_chunk_size_y = self.width, self.height, self.attn_slice_x, self.attn_slice_y, self.attn_shape, self.ff_chunk_size, self.padding_mode, self.vae_chunk_size_x, self.vae_chunk_size_y
    
     # modified from https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/master/modules/sd_hijack_optimizations.py
    def split_cross_attention_forward(self, x, context=None, mask=None):
      '''
      Splits the expensive attention calls into blocks of size
      ("expensive" means attn calls where key.size()[1] != 77)
      attn_slice_x
      attn_slice_y
      This is *not* equivalent to normal attention,
      because each block can't see outside itself.
      However in practice it seems the cheap attention calls are enough to keep image cohesion
      '''
      
      h = self.heads

      q_in = self.to_q(x)
      context = context if context is not None else x
      k_in = self.to_k(context)
      v_in = self.to_v(context)
      
      k_in *= self.scale

      del context, x
      
      def doAttn(q,k,v):
        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)

        stats = torch.cuda.memory_stats(q.device)
        mem_active = stats['active_bytes.all.current']
        mem_reserved = stats['reserved_bytes.all.current']
        mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch

        gb = 1024 ** 3
        tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
        modifier = 3 if q.element_size() == 2 else 2.5
        mem_required = tensor_size * modifier
        steps = 1

        if mem_required > mem_free_total:
            steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))
            # print(f"Expected tensor size:{tensor_size/gb:0.1f}GB, cuda free:{mem_free_cuda/gb:0.1f}GB "
            #       f"torch free:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} steps:{steps}")
        #print("have", steps, "steps")
        if steps > 256:
            max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
            raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                               f'Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free')

        slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
        for i in range(0, q.shape[1], slice_size):
            end = i + slice_size
            s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k)

            s2 = s1.softmax(dim=-1, dtype=q.dtype)
            del s1

            r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
            del s2
        return r1
        
      q1, q2, q3 = map(int, q_in.size())
      k1, k2, k3 = map(int, k_in.size())
      v1, v2, v3 = map(int, v_in.size())
      
      # final dim is 320, 640, or 1280
      # each time it doubles in size, we get w//2 x h//2
      # for example: [2, 16384, 320] = [2, 128x128, 320]
      #              [2, 4096,  640] = [2, 64x64,   640]
      #              [2, 1024, 1280] = [2, 32x32,  1280]
      if k2 == 77 or (width <= 1024 and height <= 1024): # very quick, just do it
        # q=[2, wXh, 320], k=[2, 77, 320], v=[2, 77, 320]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in
        r1 = doAttn(q,k,v)
        del q, k, v
      else:
        # ([2, 16384, 320]) torch.Size([2, 16384, 320]) torch.Size([2, 16384, 320]
        latentW = width//8
        latentH = height//8
        moreDiv = {320: 1, 640: 2, 1280: 4}[q3]
        if moreDiv == 4: # ez small
          q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))
          del q_in, k_in, v_in
          r1 = doAttn(q,k,v)
          del q, k, v
        else:
          finalRDim = 40*moreDiv
          curW = latentW//moreDiv
          curH = latentH//moreDiv
          blockQ = q_in.view(q1, curH, curW, q3)
          blockK = k_in.view(k1, curH, curW, k3)
          blockV = v_in.view(v1, curH, curW, v3)
          del q_in, k_in, v_in
          res = torch.zeros([h*q1, curH, curW, finalRDim], dtype=blockQ.dtype, device=blockQ.device)
          self.brickInd = (self.brickInd + 1) % 4
          
          if attn_shape == 'brick':
            attn_blocks = getBricks(curW, curH, self.brickInd, attn_slice_x, attn_slice_y)
          elif attn_shape == 'tile':
            attn_blocks = getTiles(curW, curH, attn_slice_x, attn_slice_y)
          else:
            raise Exception("Unknown attn_shape, should be either brick or tile")
          
          for (x_start, y_start, x_end, y_end) in attn_blocks:
              x = x_start
              endX = x_end
              y = y_start
              endY = y_end
              widthX = (endX-x)
              widthY = (endY-y)
              subQ = blockQ[:,y:endY,x:endX]
              subK = blockK[:,y:endY,x:endX]
              subV = blockV[:,y:endY,x:endX]
              subQFlat = subQ.reshape(q1, widthX*widthY, q3)
              subKFlat = subK.reshape(k1, widthX*widthY, k3)
              subVFlat = subV.reshape(v1, widthX*widthY, v3)
              del subQ, subK, subV
              q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (subQFlat, subKFlat, subVFlat))
              del subQFlat, subKFlat, subVFlat
              r1 = doAttn(q,k,v)
              del q, k, v
              r2d = r1.reshape(q1*h, widthY, widthX, -1)
              del r1
              res[:,y:endY,x:endX] = r2d
              del r2d
          
          del blockQ, blockK, blockV
          r1 = res.reshape(h*q1, curW*curH, finalRDim)
          del res

      r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
      del r1
      
      return self.to_out(r2)
    
    def split_transformer_block_forward(self, hidden_states, context=None):
      '''
      Splits up self.ff into chunks, because that gets too big
      Splitting it into chunks gets the exact same output as if we didn't split it up
      '''
      
      # we split these calls up so we can manually call del
      # probably not needed but idk
      ### hidden_states = self.attn1(self.norm1(hidden_states)) + hidden_states ###
      norm_out = self.norm1(hidden_states)
      attn_out = self.attn1(norm_out)
      del norm_out
      first_block_out = attn_out + hidden_states
      del attn_out, hidden_states
      
      ### hidden_states = self.attn2(self.norm2(hidden_states), context=context) + hidden_states ###
      norm_2_out = self.norm2(first_block_out)
      attn_2_out = self.attn2(norm_2_out, context=context)
      del norm_2_out
      second_block_out = attn_2_out + first_block_out
      del attn_2_out, first_block_out
      
      ### hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states ###
      norm_3_out = self.norm3(second_block_out) # norm3Out [2,1024, 1280]
      
      # split this part up
      ## feedForwardOut = self.ff(norm3Out) ##
      if width <= 1024 and height <= 1024: # no need to split up
        linear_output = self.ff(norm_3_out)
      else:
        d1, d2, d3 = map(int, norm_3_out.size())
        chunkSize = min(ff_chunk_size, d2)
        linear_output = torch.zeros(norm_3_out.size(), dtype=norm_3_out.dtype, device=norm_3_out.device)
        for c in range(0, d2, chunkSize):
          max_c = min(c+chunkSize, d2)
          chunk_input = norm_3_out[:,c:max_c]
          output_piece = self.ff(chunk_input)
          linear_output[:,c:max_c] = output_piece
          del chunk_input, output_piece
        del norm_3_out
      
      res = linear_output + second_block_out
      del linear_output, second_block_out
      return res
      
      
    def split_vae_decode(self, z: torch.FloatTensor, return_dict: bool = True):
      # we are going from [x,y] -> [x*8, y*8]
      dec = chunkedVaeDecode(self, z, padding_mode=padding_mode, chunk_size_x=vae_chunk_size_x, chunk_size_y=vae_chunk_size_y)
      del z

      if not return_dict:
          return (dec,)

      return DecoderOutput(sample=dec)
      
    def split_vae_encode(self, x: torch.FloatTensor, return_dict: bool = True) -> AutoencoderKLOutput:
        # we are going from [x,y] -> [x//8, y//8]
        h = chunkedVaeEncode(self, x, padding_mode=padding_mode, chunk_size_x=vae_chunk_size_x*8, chunk_size_y=vae_chunk_size_y*8)
        del x
        moments = self.quant_conv(h)
        del h
        posterior = DiagonalGaussianDistribution(moments)
        del moments

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)
    
      
    self.old_cross_attntion_forward = diffusers.models.attention.CrossAttention.forward
    diffusers.models.attention.CrossAttention.forward = split_cross_attention_forward
    
    self.old_basic_transformer_block_forward = diffusers.models.attention.BasicTransformerBlock.forward
    diffusers.models.attention.BasicTransformerBlock.forward = split_transformer_block_forward
    
    self.old_vae_decode = diffusers.models.vae.AutoencoderKL.decode
    diffusers.models.vae.AutoencoderKL.decode = split_vae_decode
    
    self.old_vae_encode = diffusers.models.vae.AutoencoderKL.encode
    diffusers.models.vae.AutoencoderKL.encode = split_vae_encode

  def __exit__(self, exc_type, exc_val, exc_tb):
    diffusers.models.attention.CrossAttention.forward = self.old_cross_attntion_forward
    diffusers.models.attention.BasicTransformerBlock.forward = self.old_basic_transformer_block_forward
    diffusers.models.vae.AutoencoderKL.decode = self.old_vae_decode
    diffusers.models.vae.AutoencoderKL.encode = self.old_vae_encode
    
    for m in self.pipe.unet.modules():
      if type(m) is diffusers.models.attention.CrossAttention:
        del m.brickInd
    
