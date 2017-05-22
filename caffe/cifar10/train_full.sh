#!/usr/bin/env sh
caffe train --solver=cifar10_full_solver.prototxt $@

# reduce learning rate by factor of 10
caffe train --solver=cifar10_full_solver_lr1.prototxt --snapshot=cifar10_full_iter_60000.solverstate $@

# reduce learning rate by factor of 10
caffe train --solver=cifar10_full_solver_lr2.prototxt --snapshot=cifar10_full_iter_65000.solverstate $@
