To run the code execute the following command:

time THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main.py data/train-labels-idx1-ubyte data/train-images-idx3-ubyte data/t10k-labels-idx1-ubyte data/t10k-images-idx3-ubyte

- on a MAcBook Pro with CUDA Toolkit installed the execution will take less than 4min.
- It is seems that at the moment only computations with float32 can be accelerated.
