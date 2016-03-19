# TODO FIXIT

# 
MODELDIR=logs512_mpii64
INPUTSIZE=[1,64,64]
# convert to Caffe
th fb-caffe-exts/torch2caffe/torch2caffe.lua \
   --input $MODELDIR/adversarial.net \
   --prototxt $MODELDIR/adversarial.prototxt \
   --output-caffemodel $MODELDIR/adversarial.caffemodel \ 
   --input-tensor $INPUTSIZE \
   --preprocessing --verify 
# plot using Caffe tool
python $CAFFE_ROOT/python/draw_net.py \
       $MODELDIR/adversarial.prototxt \
       $MODELDIR/adversarial_net.png
