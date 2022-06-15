# batch size config at net/model_factory.py
#nsys profile -t cuda,osrt,nvtx,cublas,cudnn --gpuctxsw=true -d 120 -o ckpt-batch112-resnet50 -f true -w true \
python benchmark.py \
	--arch resnet50 \
	--device cuda:0
