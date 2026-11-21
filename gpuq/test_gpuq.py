"""gpuq/test_gpuq.py (trench-gguf) - the GPU quantizer (libgpuq.so) against ggml_quantize_chunk, byte for byte.

    test_gpuq.py <gguf shard> <imatrix.gguf> [--tensor=<regex>] [--experts=N] [--types=iq1_s,iq1_m,...,q4_k,q8_0]
    test_gpuq.py --selftest [--types=...]      (no model files: random weights, a coarse 4-bit-like copy with many
                                                equal values, and a random imatrix)

The k-quants (Q2_K .. Q6_K) are checked with the imatrix and without one (ggml's two code paths), Q8_0 without
(ggml ignores the imatrix for it). Also checks a batch of all N matrices in one call (gpuq_quantize_batch) and reading
the source on the GPU (gpuq_requantize: F16 / BF16 / Q8_0 / MXFP4 bytes -> IQ2_XXS).
Takes the first tensor matching --tensor (an expert tensor, ne2 experts), dequantizes N experts with ggml's to_float,
quantizes each to every type on the CPU (ggml) and on the GPU, and prints the share of identical blocks and the time.
"""
import ctypes
import os
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from requant_experts import _env, LIBGGML, TO_FLOAT, TypeTraits, read_imatrix, gguf, np, Q  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    flags = {a.split("=", 1)[0]: (a.split("=", 1) + [""])[1] for a in sys.argv[1:] if a.startswith("--")}
    selftest = "--selftest" in flags
    if not selftest:
        shard, imat_path = [a for a in sys.argv[1:] if not a.startswith("--")]
    pat = re.compile(flags.get("--tensor", r"ffn_gate_exps"))
    n_exp = int(flags.get("--experts", "4"))
    types = flags.get("--types", "iq1_s,iq1_m,iq2_xxs,iq2_xs,iq2_s,iq3_xxs,iq3_s,iq4_xs,iq4_nl,"
                                 "q2_k,q3_k,q4_k,q5_k,q6_k,q8_0").split(",")

    lib = ctypes.CDLL(LIBGGML)
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_void_p]
    lib.ggml_get_type_traits.restype = ctypes.POINTER(TypeTraits)
    lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
    lib.ggml_quantize_init.argtypes = [ctypes.c_int]
    gq = ctypes.CDLL(os.path.join(HERE, _env("GPUQ_LIB") or "libgpuq.so"))
    gq.gpuq_init.argtypes = [ctypes.c_char_p, ctypes.c_int]
    gq.gpuq_quantize.restype = ctypes.c_size_t
    gq.gpuq_quantize.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                 ctypes.c_void_p]
    gq.gpuq_block_bytes.restype = ctypes.c_size_t
    gq.gpuq_quantize_batch.restype = ctypes.c_size_t
    gq.gpuq_quantize_batch.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                       ctypes.c_int64, ctypes.c_void_p]
    gq.gpuq_requantize.restype = ctypes.c_size_t
    gq.gpuq_requantize.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int64,
                                   ctypes.c_int64, ctypes.c_void_p]
    rc = gq.gpuq_init(os.path.join(HERE, "gpuq_tables.bin").encode(), int(os.environ.get("GPUQ_DEVICE", "0")))
    assert rc == 0, "gpuq_init %d" % rc

    if selftest:
        rng = np.random.default_rng(1234)
        ne0, ne1 = 2304, 256
        g = rng.standard_normal((ne1, ne0)).astype(np.float32) * 0.02
        coarse = (np.round(g / 0.005) * 0.005).astype(np.float32)          # ties, as MXFP4 / k-quant sources have
        heavy = (rng.standard_t(3, (ne1, ne0)) * 0.01).astype(np.float32)  # outliers
        edge = g.copy()                     # all-zero / constant / one-signed / tiny sub-blocks (the scale guards)
        edge[0:8] = 0
        edge[8:16, :1024] = 0.01
        edge[16:32] = np.abs(edge[16:32])
        edge[32:48] = -np.abs(edge[32:48])
        edge[48:64] *= 1e-20
        edge[64:80, ::16] = 0
        xs = [g.reshape(-1), coarse.reshape(-1), heavy.reshape(-1), edge.reshape(-1)]
        imv = [rng.gamma(1.0, 1.0, ne0).astype(np.float32) for _ in xs]
        n_exp = len(xs)
        print("selftest: %d random matrices [%d x %d]" % (n_exp, ne0, ne1))
    else:
        imat = read_imatrix(imat_path)
        r = gguf.GGUFReader(shard)
        t = next((t for t in r.tensors if pat.search(t.name) and len(t.shape) == 3), None)
        if t is None:
            sys.exit("no 3-D expert tensor matching --tensor in %s (a split model keeps its experts in some shards only:\n"
                     "pass a shard that holds them)" % shard)
        ne0, ne1, ne2 = (int(x) for x in t.shape)
        print("%s %s [%d x %d x %d], %d experts" % (t.name, t.tensor_type.name, ne0, ne1, ne2, n_exp))
        to_float = TO_FLOAT(lib.ggml_get_type_traits(int(t.tensor_type)).contents.to_float)
        raw = np.ascontiguousarray(np.asarray(t.data).reshape(ne2, -1))
        im = imat[t.name]
        xs = []
        for e in range(n_exp):
            f32 = np.empty(ne1 * ne0, dtype=np.float32)
            to_float(raw[e].ctypes.data, f32.ctypes.data, ne1 * ne0)
            xs.append(f32)
        imv = [im[e] for e in range(n_exp)]
    bad = 0
    for name in types:
        typ = Q[name.upper()]
        bb = gq.gpuq_block_bytes(int(typ))
        blk = gguf.GGML_QUANT_SIZES[typ][0]
        lib.ggml_quantize_init(int(typ))
        kq = name.startswith("q") or name in ("iq4_nl", "iq4_xs")   # (also quantized without an imatrix)
        modes = ([] if name == "q8_0" else [True]) + ([False] if kq else [])
        for with_im in modes:
            same = blocks = 0
            tc = tg = 0.0
            cpu_all = []
            for e in range(n_exp):
                nbytes = ne1 * (ne0 // blk) * bb
                a = np.empty(nbytes, dtype=np.uint8)
                b = np.empty(nbytes, dtype=np.uint8)
                w = imv[e].ctypes.data if with_im else None
                t0 = time.time()
                got = lib.ggml_quantize_chunk(int(typ), xs[e].ctypes.data, a.ctypes.data, 0, ne1, ne0, w)
                t1 = time.time()
                got2 = gq.gpuq_quantize(int(typ), xs[e].ctypes.data, b.ctypes.data, ne1, ne0, w)
                t2 = time.time()
                assert got == nbytes and got2 == nbytes, (name, got, got2, nbytes)
                tc += t1 - t0
                tg += t2 - t1
                cpu_all.append(a)
                ab, bb_ = a.reshape(-1, bb), b.reshape(-1, bb)
                eq = np.all(ab == bb_, axis=1)
                same += int(eq.sum())
                blocks += len(eq)
                if not eq.all() and e == 0:
                    i = int(np.argmin(eq))
                    print("   first differing block %d:\n   cpu %s\n   gpu %s" % (i, ab[i].tobytes().hex(), bb_[i].tobytes().hex()))
            # all matrices in one call (each its own imatrix row)
            xb = np.ascontiguousarray(np.stack(xs[:n_exp]))
            ib = np.ascontiguousarray(np.stack(imv[:n_exp]).astype(np.float32))
            ob = np.empty(n_exp * len(cpu_all[0]), dtype=np.uint8)
            t3 = time.time()
            got3 = gq.gpuq_quantize_batch(int(typ), xb.ctypes.data, ob.ctypes.data, n_exp, ne1, ne0,
                                          ib.ctypes.data if with_im else None)
            t4 = time.time()
            bdiff = len(ob) // bb if got3 != len(ob) else int(
                (ob.reshape(-1, bb) != np.concatenate(cpu_all).reshape(-1, bb)).any(axis=1).sum())
            label = name + ("" if not kq else " +im" if with_im else " noim")
            print("%-12s identical blocks %d / %d (%.4f%%), batch of %d: %s   cpu 1 thread %.3f s/matrix, gpu %.4f s/matrix"
                  " (x%.0f), batch %.4f s/matrix" %
                  (label, same, blocks, 100.0 * same / blocks, n_exp, "identical" if bdiff == 0 else "%d differ" % bdiff,
                   tc / n_exp, tg / n_exp, tc / max(tg, 1e-9), (t4 - t3) / n_exp), flush=True)
            bad += blocks - same + bdiff
    # reading the source on the GPU (gpuq_requantize): the source bytes -> GPU floats -> IQ2_XXS, against ggml's
    # to_float -> ggml_quantize_chunk. Self-test: the first matrix stored as each source type; a model: its own tensor.
    if selftest:
        srcs = []
        for st in ("F16", "BF16", "Q8_0", "MXFP4"):
            typ = Q[st]
            tr = lib.ggml_get_type_traits(int(typ)).contents
            enc = np.empty(ne1 * ne0 // tr.blck_size * tr.type_size, dtype=np.uint8)
            lib.ggml_quantize_chunk(int(typ), xs[0].ctypes.data, enc.ctypes.data, 0, ne1, ne0, None)
            srcs.append((st, typ, enc))
    elif gq.gpuq_can_dequantize(int(t.tensor_type)) == 1:
        srcs = [(t.tensor_type.name, t.tensor_type, raw[0])]
    else:
        srcs = []
        print("%s: read on the CPU (gpuq_requantize does not take this type)" % t.tensor_type.name)
    tq = Q.IQ2_XXS
    bbq = gq.gpuq_block_bytes(int(tq))
    for st, typ, enc in srcs:
        f32 = np.empty(ne1 * ne0, dtype=np.float32)
        TO_FLOAT(lib.ggml_get_type_traits(int(typ)).contents.to_float)(enc.ctypes.data, f32.ctypes.data, ne1 * ne0)
        a = np.empty(ne1 * ne0 // 256 * bbq, dtype=np.uint8)
        b = np.empty_like(a)
        lib.ggml_quantize_chunk(int(tq), f32.ctypes.data, a.ctypes.data, 0, ne1, ne0, imv[0].ctypes.data)
        got = gq.gpuq_requantize(int(typ), np.ascontiguousarray(enc).ctypes.data, int(tq), b.ctypes.data, ne1, ne0,
                                 imv[0].ctypes.data)
        diff = int((a.reshape(-1, bbq) != b.reshape(-1, bbq)).any(axis=1).sum()) if got == a.size else len(a) // bbq
        print("%-8s read on the GPU -> iq2_xxs: %s" % (st.lower(), "identical" if diff == 0 else "%d blocks differ" % diff))
        bad += diff
    print("OK: byte-identical to ggml" if bad == 0 else "MISMATCH: %d blocks differ" % bad)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
