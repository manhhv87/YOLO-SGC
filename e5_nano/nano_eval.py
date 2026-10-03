# -*- coding: utf-8 -*-
"""nano_eval.py - run the TensorRT engines on the 500 test images on the Jetson Nano; predictions are scored on a PC.

Uses the system Python 3.6 of JetPack 4.6 (python3) and needs tensorrt (package python3-libnvinfer,
shipped with JetPack), numpy and cv2 (also shipped). No pycuda or torch: GPU memory is allocated through
ctypes + libcudart.

    python3 nano_eval.py --selftest            # one engine, 3 images
    python3 nano_eval.py                       # every engine in engines/ -> results_nano/pred_<name>.npz

Pre- and post-processing come from trt_common.py, identical to the PC version, so an accuracy
difference from the FP32 ONNX model comes only from the engine arithmetic. Per-image times: image read,
pre-processing, inference (copy to GPU + run + copy back) and NMS.
"""
import argparse, ctypes, gc, glob, os, sys, time

import numpy as np
import cv2
import tensorrt as trt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import trt_common as tc


def load_cudart():
    cands = ["libcudart.so", "/usr/local/cuda/lib64/libcudart.so"] + sorted(glob.glob("/usr/local/cuda*/lib64/libcudart.so*"))
    for c in cands:
        try:
            return ctypes.CDLL(c)
        except OSError:
            pass
    sys.exit("libcudart.so not found (try: export LD_LIBRARY_PATH=/usr/local/cuda/lib64)")


class Cuda(object):
    H2D, D2H = 1, 2

    def __init__(self):
        rt = load_cudart()
        rt.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        rt.cudaFree.argtypes = [ctypes.c_void_p]
        rt.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.rt = rt

    @staticmethod
    def check(rc, what):
        if rc != 0:
            raise RuntimeError("%s: CUDA error code %d" % (what, rc))

    def malloc(self, nbytes):
        p = ctypes.c_void_p()
        self.check(self.rt.cudaMalloc(ctypes.byref(p), nbytes), "cudaMalloc")
        return p

    def h2d(self, dptr, arr):
        self.check(self.rt.cudaMemcpy(dptr, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes, self.H2D), "cudaMemcpy H2D")

    def d2h(self, arr, dptr):
        self.check(self.rt.cudaMemcpy(arr.ctypes.data_as(ctypes.c_void_p), dptr, arr.nbytes, self.D2H), "cudaMemcpy D2H")

    def free(self, dptr):
        self.rt.cudaFree(dptr)


class Engine(object):
    def __init__(self, path, cuda):
        self.cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        self.runtime = trt.Runtime(logger)              # keep the runtime alive with the engine
        with open(path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("cannot load engine %s (built with another TensorRT version?)" % path)
        self.ctx = self.engine.create_execution_context()
        self.bindings, self.outs, self.inp = [], [], None
        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            arr = np.empty(shape, dtype=trt.nptype(self.engine.get_binding_dtype(i)))
            d = self.cuda.malloc(arr.nbytes)
            self.bindings.append(int(d.value))
            if self.engine.binding_is_input(i):
                self.inp = (arr, d)
            else:
                self.outs.append((arr, d))

    def __call__(self, x):
        arr, d = self.inp
        self.cuda.h2d(d, np.ascontiguousarray(x, dtype=arr.dtype))
        if not self.ctx.execute_v2(self.bindings):      # synchronous: returns when the GPU has finished
            raise RuntimeError("execute_v2 failed")
        for a, dd in self.outs:
            self.cuda.d2h(a, dd)
        return [a for a, _ in self.outs]

    def close(self):
        for _, d in [self.inp] + self.outs:
            self.cuda.free(d)
        del self.ctx, self.engine, self.runtime


def run(path, imgs, cuda, out_dir):
    name = os.path.basename(path)[:-len(".engine")]
    eng = Engine(path, cuda)
    x0 = np.zeros(eng.inp[0].shape, np.float32)
    for _ in range(20):                                  # warm-up
        eng(x0)
    dets, idx, times = [], [], []
    for k, p in enumerate(imgs):
        t0 = time.perf_counter()
        bgr = cv2.imread(p)
        t1 = time.perf_counter()
        x, r, pad = tc.preprocess(bgr)
        t2 = time.perf_counter()
        out = eng(x)[0]
        t3 = time.perf_counter()
        d = tc.postprocess(out, bgr.shape[:2], r, pad)
        t4 = time.perf_counter()
        dets.append(d)
        idx.append(np.full(len(d), k, np.int32))
        times.append((t1 - t0, t2 - t1, t3 - t2, t4 - t3))
    eng.close()
    t = np.array(times, np.float64) * 1000.0
    if out_dir:
        np.savez_compressed(os.path.join(out_dir, "pred_%s.npz" % name),
                            dets=np.concatenate(dets).astype(np.float32), img=np.concatenate(idx),
                            names=np.array([os.path.basename(p) for p in imgs]),
                            times_ms=t, trt_version=np.array(trt.__version__))
    print("%-34s %4d images | ms/image: read %.1f  pre %.1f  inference %.1f  NMS %.1f | %d boxes" % (
        name, len(imgs), t[:, 0].mean(), t[:, 1].mean(), t[:, 2].mean(), t[:, 3].mean(), sum(len(d) for d in dets)))
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="*", default=None, help="default: engines/*.engine")
    ap.add_argument("--images", default=os.path.join(HERE, "test_images"))
    ap.add_argument("--out", default=os.path.join(HERE, "results_nano"))
    ap.add_argument("--selftest", action="store_true", help="one engine, 3 images, nothing written")
    args = ap.parse_args()

    engines = args.engines or sorted(glob.glob(os.path.join(HERE, "engines", "*.engine")))
    imgs = sorted(glob.glob(os.path.join(args.images, "*.jpg")) + glob.glob(os.path.join(args.images, "*.png")))
    if not engines:
        sys.exit("No engine in engines/; run nano_bench.sh first.")
    if not imgs:
        sys.exit("No images in %s" % args.images)
    print("TensorRT %s | %d engines | %d images" % (trt.__version__, len(engines), len(imgs)))
    cuda = Cuda()
    if args.selftest:
        run(engines[0], imgs[:3], cuda, None)
        return
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    for e in engines:
        name = os.path.basename(e)[:-len(".engine")]
        if os.path.exists(os.path.join(args.out, "pred_%s.npz" % name)):
            print("%-34s exists, skipped" % name)
            continue
        run(e, imgs, cuda, args.out)
        gc.collect()


if __name__ == "__main__":
    main()
