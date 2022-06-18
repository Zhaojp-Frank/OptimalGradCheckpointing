# batch size config at net/model_factory.py

# issues:
# vgg** has no benefit but enlarge memory
# resnet50 batch=64~104 jit works; batch=112 128 jit failed, and mem leak

#nsys profile -t cuda,osrt,nvtx,cublas,cudnn --gpuctxsw=true -d 120 -o ckpt-batch112-resnet50 -f true -w true \

python benchmark.py \
	--arch resnet50 \
	--device cuda:0
