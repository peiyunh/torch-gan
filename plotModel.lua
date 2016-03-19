require "torch"
require "cudnn"
require "nn"
require "cunn"
require "nngraph"

-- Use file:write, file:close
--   instead of io.write and io.close(file)
function model2dot(name, model, output)
   local file = io.open(output, "w")
   file:write(string.format("digraph %s {\n", name))
   file:write('rotate=90; \n\n')
   for i = 1, #model.modules do
      local module = model.modules[i]
      local mname = module.__typename
      mname = string.gsub(mname, "cudnn%.", "")
      mname = string.gsub(mname, "nn%.", "")
      if mname == "SpatialMaxPooling" then
	 label = string.format("%s\\n%dx%d-%dx%d", mname,
			       module.kW, module.kH,
			       module.dW, module.dH)
      elseif mname == "SpatialConvolution" then
	 sz = module.weight:size()
	 label = string.format("%s\\n%dx%dx%dx%d", mname,
			       sz[1],sz[2],sz[3],sz[4])
      else
	 label = mname
      end
      label = string.format("\"%s\"", label)
      -- sz = module.weight:size()
      if i > 1 then
	 file:write(string.format("node%d -> node%d;\n",i-1, i))
      end

      -- label = string.format("\"%s-%dx%dx%dx%d\"", mname,sz[1],sz[2],sz[3],sz[4])
      file:write(string.format("node%d [label=%s,shape=box];\n", i, label))
   end
   file:write("}\n")
   file:close()
end

model_D = nn.Sequential()
model_D:add(cudnn.SpatialConvolution(1, 32, 5, 5, 1, 1, 2, 2))
model_D:add(cudnn.SpatialMaxPooling(2,2))
model_D:add(cudnn.ReLU(true))
model_D:add(nn.SpatialDropout(0.2))
model_D:add(cudnn.SpatialConvolution(32, 64, 5, 5, 1, 1, 2, 2))
model_D:add(cudnn.SpatialMaxPooling(2,2))
model_D:add(cudnn.ReLU(true))
model_D:add(nn.SpatialDropout(0.2))
model_D:add(cudnn.SpatialConvolution(64, 96, 5, 5, 1, 1, 2, 2))
model_D:add(cudnn.ReLU(true))
model_D:add(cudnn.SpatialMaxPooling(2,2))
model_D:add(nn.SpatialDropout(0.2))
model_D:add(nn.Reshape(8*8*96))
model_D:add(nn.Linear(8*8*96, 1024))
model_D:add(cudnn.ReLU(true))
model_D:add(nn.Dropout())
model_D:add(nn.Linear(1024,1))
model_D:add(nn.Sigmoid())

model2dot('D', model_D, "model_D.dot")

x_input = nn.Identity()()
lg = nn.Linear(512, 128*8*8)(x_input)
lg = nn.Reshape(128, 8, 8)(lg)
lg = cudnn.ReLU(true)(lg)
lg = nn.SpatialUpSamplingNearest(2)(lg)
lg = cudnn.SpatialConvolution(128, 256, 5, 5, 1, 1, 2, 2)(lg)
lg = nn.SpatialBatchNormalization(256)(lg)
lg = cudnn.ReLU(true)(lg)
lg = nn.SpatialUpSamplingNearest(2)(lg)
lg = cudnn.SpatialConvolution(256, 256, 5, 5, 1, 1, 2, 2)(lg)
lg = nn.SpatialBatchNormalization(256)(lg)
lg = cudnn.ReLU(true)(lg)
lg = nn.SpatialUpSamplingNearest(2)(lg)
lg = cudnn.SpatialConvolution(256, 128, 5, 5, 1, 1, 2, 2)(lg)
lg = nn.SpatialBatchNormalization(128)(lg)
lg = cudnn.ReLU(true)(lg)
lg = cudnn.SpatialConvolution(128, 1, 3, 3, 1, 1, 1, 1)(lg)
model_G = nn.gModule({x_input}, {lg})

model2dot('G', model_G, "model_G.dot")
