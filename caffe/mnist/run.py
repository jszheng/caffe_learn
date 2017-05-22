#

'''
Environment

~/caffe_learn/caffe/mnist$ module display caffe
-------------------------------------------------------------------
/opt/modulefiles/caffe:

conflict         caffe
setenv           CAFFE_ROOT /home/manager/caffe
setenv           CAFFE_INCLUDE /home/manager/caffe/include
setenv           CAFFE_LIB /home/manager/caffe/build/lib
prepend-path     PATH /home/manager/caffe/build/tools
prepend-path     PYTHONPATH /home/manager/caffe/python
prepend-path     LD_LIBRARY_PATH /opt/intel/mkl/lib/intel64
prepend-path     LD_LIBRARY_PATH /home/manager/caffe/build/lib
-------------------------------------------------------------------

ll $CAFFE_ROOT/data/mnist/
total 53676
-rwxrwxr-x 1 manager manager      408  5月 18 16:15 get_mnist.sh*
-rw-rw-r-- 1 manager manager  7840016  7月 22  2000 t10k-images-idx3-ubyte
-rw-rw-r-- 1 manager manager    10008  7月 22  2000 t10k-labels-idx1-ubyte
-rw-rw-r-- 1 manager manager 47040016  7月 22  2000 train-images-idx3-ubyte
-rw-rw-r-- 1 manager manager    60008  7月 22  2000 train-labels-idx1-ubyte

# create train database 60K images
$CAFFE_ROOT/.build_release/examples/mnist/convert_mnist_data.bin  \
    $CAFFE_ROOT/data/mnist/train-images-idx3-ubyte \
    $CAFFE_ROOT/data/mnist/train-labels-idx1-ubyte \
    mnist_train_lmdb \
    --backend=lmdb

I0519 17:05:02.932853  7989 db_lmdb.cpp:35] Opened lmdb mnist_train_lmdb
I0519 17:05:02.933167  7989 convert_mnist_data.cpp:88] A total of 60000 items.
I0519 17:05:02.933192  7989 convert_mnist_data.cpp:89] Rows: 28 Cols: 28
I0519 17:05:07.608922  7989 convert_mnist_data.cpp:108] Processed 60000 files.

# create test database 10k images
$CAFFE_ROOT/.build_release/examples/mnist/convert_mnist_data.bin  \
    $CAFFE_ROOT/data/mnist/t10k-images-idx3-ubyte \
    $CAFFE_ROOT/data/mnist/t10k-labels-idx1-ubyte \
    mnist_test_lmdb \
    --backend=lmdb

I0519 17:15:24.460626  8157 db_lmdb.cpp:35] Opened lmdb mnist_test_lmdb
I0519 17:15:24.486660  8157 convert_mnist_data.cpp:88] A total of 10000 items.
I0519 17:15:24.486698  8157 convert_mnist_data.cpp:89] Rows: 28 Cols: 28
I0519 17:15:25.237675  8157 convert_mnist_data.cpp:108] Processed 10000 files.

caffe train --solver=lenet_solver.prototxt
caffe train --solver=lenet_solver_adam.prototxt

'''

