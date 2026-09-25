import sys
import time
import runpy
import inspect
import functools
import traceback
from pathlib import Path
from collections import defaultdict


# ============================================================
# CONFIG
# ============================================================

TARGET = "/app/fill_pytorch_compile2.py"

# We only care about functions taking meaningful time.
MIN_TOTAL = 0.100       # 100 ms cumulative
MIN_CALL = 0.050        # 50 ms individual call

# We stop the original script immediately after FIRST generate_long().
STOP_AFTER_FIRST_GENERATION = True


# ============================================================
# STORAGE
# ============================================================

stats = defaultdict(
    lambda: {
        "calls": 0,
        "total": 0.0,
        "max": 0.0,
        "module": "",
        "qualname": "",
    }
)

active = set()
wrapped = []


# ============================================================
# TIMER WRAPPER
# ============================================================

def make_wrapper(func, module_name, qualname):

    key = f"{module_name}:{qualname}"

    # Avoid wrapping the same function multiple times.
    if getattr(func, "__mega_timer_wrapped__", False):
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        # Recursive/self calls are ignored to avoid double counting
        # the same function recursively.
        if key in active:
            return func(*args, **kwargs)

        active.add(key)

        t0 = time.perf_counter()

        try:
            return func(*args, **kwargs)

        finally:
            dt = time.perf_counter() - t0

            active.discard(key)

            s = stats[key]
            s["calls"] += 1
            s["total"] += dt
            s["max"] = max(s["max"], dt)
            s["module"] = module_name
            s["qualname"] = qualname

            # Immediate notification for really long individual calls.
            if dt >= 1.0:
                print(
                    f"[MEGA-LONG] "
                    f"{dt:9.3f}s "
                    f"{module_name}.{qualname}",
                    flush=True,
                )

    wrapper.__mega_timer_wrapped__ = True

    wrapped.append((module_name, qualname))

    return wrapper


# ============================================================
# MODULES TO INSTRUMENT
# ============================================================

MODULE_NAMES = [
    # Main Inductor compilation machinery
    "torch._inductor.compile_fx",
    "torch._inductor.codecache",
    "torch._inductor.async_compile",

    # Dynamo / compiler front-end
    "torch._dynamo",
    "torch._dynamo.convert_frame",
    "torch._dynamo.eval_frame",
    "torch._dynamo.output_graph",
    "torch._dynamo.backends.registry",

    # Inductor runtime
    "torch._inductor.runtime.autotune_cache",
    "torch._inductor.runtime.triton_heuristics",
    "torch._inductor.runtime.compile_tasks",
    "torch._inductor.runtime.triton_bundler",
    "torch._inductor.runtime.runtime_utils",
    "torch._inductor.runtime.caching",
    "torch._inductor.runtime.benchmarking",
    "torch._inductor.runtime.hints",
    "torch._inductor.runtime.triton_helpers",

    # Triton integration
    "torch._inductor.triton_heuristics",
    "torch._inductor.triton_bundler",

    # CUDA / generated-code loading
    "torch._inductor.runtime.static_cuda_launcher",

    # PyTorch compiler API
    "torch.compiler",
]


# ============================================================
# FUNCTIONS WE DO NOT WANT
# ============================================================

EXCLUDE_NAMES = {
    "__getattr__",
    "__getattribute__",
    "__setattr__",
    "__delattr__",
    "__repr__",
    "__str__",
    "__hash__",
    "__eq__",
    "__ne__",
}


# ============================================================
# KEYWORDS
# ============================================================

# We don't blindly wrap every Python function in all of Dynamo.
# We focus on functions whose names can plausibly account for
# compilation/cache/autotune/loading time.

KEYWORDS = (
    "compile",
    "code",
    "cache",
    "load",
    "deserialize",
    "serialize",
    "precompile",
    "autotune",
    "tune",
    "benchmark",
    "bench",
    "launcher",
    "kernel",
    "triton",
    "async",
    "wait",
    "compile_fx",
    "inductor",
    "graph",
    "codegen",
    "lower",
    "schedule",
    "artifact",
    "bundle",
    "cubin",
    "ptx",
    "python",
    "module",
)


def interesting_name(name):
    lname = name.lower()

    if lname in EXCLUDE_NAMES:
        return False

    return any(k in lname for k in KEYWORDS)


# ============================================================
# IMPORT + WRAP
# ============================================================

def import_and_wrap():

    print()
    print("=" * 100)
    print("[MEGA] IMPORTING PYTORCH / INDUCTOR MODULES")
    print("=" * 100)

    import importlib

    total_wrapped = 0

    for module_name in MODULE_NAMES:

        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            print(
                f"[MEGA] SKIP {module_name}: {type(e).__name__}: {e}",
                flush=True,
            )
            continue

        count = 0

        try:
            members = inspect.getmembers(module)
        except Exception:
            continue

        for name, obj in members:

            if not interesting_name(name):
                continue

            if inspect.isfunction(obj):

                # Only functions actually defined in this module.
                # This prevents wrapping imported functions over and
                # over again.
                if getattr(obj, "__module__", None) != module_name:
                    continue

                new_obj = make_wrapper(
                    obj,
                    module_name,
                    name,
                )

                try:
                    setattr(module, name, new_obj)
                    count += 1
                except Exception:
                    pass

        if count:
            print(
                f"[MEGA] {module_name}: wrapped {count} functions",
                flush=True,
            )

        total_wrapped += count

    print()
    print(f"[MEGA] TOTAL WRAPPED FUNCTIONS: {total_wrapped}")
    print("=" * 100)
    print()

    return total_wrapped


# ============================================================
# SPECIAL: GENERATE_LONG
# ============================================================

_generate_count = 0


def install_generate_long_stop():

    import fish_speech.models.text2semantic.inference as inference

    original = inference.generate_long

    global _generate_count

    def generate_long_wrapper(*args, **kwargs):

        global _generate_count

        _generate_count += 1

        current = _generate_count

        print()
        print(
            f"[MEGA] >>> generate_long #{current}",
            flush=True,
        )

        t0 = time.perf_counter()

        try:

            # generate_long is a generator.
            # The actual work happens during iteration.
            for item in original(*args, **kwargs):
                yield item

        finally:

            dt = time.perf_counter() - t0

            print(
                f"[MEGA] <<< generate_long #{current}: "
                f"{dt:.3f}s",
                flush=True,
            )

            if (
                STOP_AFTER_FIRST_GENERATION
                and current == 1
            ):
                print()
                print("=" * 100)
                print(
                    "[MEGA] FIRST GENERATION COMPLETE - "
                    "STOPPING ORIGINAL SCRIPT"
                )
                print("=" * 100)
                print()

                # Give the generator a chance to finish its own cleanup,
                # then stop before generation 2/3.
                raise SystemExit(0)

    inference.generate_long = generate_long_wrapper


# ============================================================
# RESULTS
# ============================================================

def print_results():

    print()
    print()
    print("=" * 110)
    print("[MEGA] FINAL FUNCTION TIMINGS")
    print("=" * 110)
    print()

    rows = []

    for key, s in stats.items():

        if s["total"] < MIN_TOTAL:
            continue

        rows.append(
            (
                s["total"],
                s["max"],
                s["calls"],
                s["module"],
                s["qualname"],
            )
        )

    rows.sort(reverse=True)

    print(
        f"{'TOTAL':>12} "
        f"{'MAX':>12} "
        f"{'CALLS':>8}  "
        f"FUNCTION"
    )

    print("-" * 110)

    for total, max_dt, calls, module, qualname in rows:

        print(
            f"{total:12.3f} "
            f"{max_dt:12.3f} "
            f"{calls:8d}  "
            f"{module}.{qualname}"
        )

    print()
    print("=" * 110)
    print(
        f"[MEGA] Functions with cumulative time >= "
        f"{MIN_TOTAL:.3f}s: {len(rows)}"
    )
    print("=" * 110)
    print()


# ============================================================
# LONGEST SINGLE CALLS
# ============================================================

def print_summary():

    print()
    print("=" * 110)
    print("[MEGA] INTERPRETATION SUMMARY")
    print("=" * 110)
    print()

    rows = []

    for key, s in stats.items():

        if s["total"] >= MIN_TOTAL:
            rows.append(
                (
                    s["total"],
                    s["max"],
                    s["calls"],
                    f"{s['module']}.{s['qualname']}",
                )
            )

    rows.sort(reverse=True)

    print("[MEGA] TOP 30 BY CUMULATIVE TIME")
    print()

    for total, max_dt, calls, name in rows[:30]:

        print(
            f"{total:9.3f}s total | "
            f"{max_dt:9.3f}s max | "
            f"{calls:5d} calls | "
            f"{name}"
        )

    print()
    print("[MEGA] TOP 30 BY SINGLE CALL")
    print()

    rows2 = sorted(
        rows,
        key=lambda x: x[1],
        reverse=True,
    )

    for total, max_dt, calls, name in rows2[:30]:

        print(
            f"{max_dt:9.3f}s max | "
            f"{total:9.3f}s total | "
            f"{calls:5d} calls | "
            f"{name}"
        )

    print()
    print("=" * 110)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("#" * 110)
    print("# MEGA TIMER - FISH SPEECH / PYTORCH / INDUCTOR")
    print("#" * 110)
    print()

    t0 = time.perf_counter()

    # IMPORTANT:
    # Import and wrap before executing the Fish Speech script.
    import_and_wrap()

    # Patch generate_long after importing its module.
    install_generate_long_stop()

    print()
    print("=" * 100)
    print("[MEGA] STARTING ORIGINAL FISH SPEECH SCRIPT")
    print("[MEGA] ONLY FIRST GENERATION")
    print("[MEGA] NO CACHE DELETION")
    print("[MEGA] NO MEGACACHE REWRITE")
    print("=" * 100)
    print()

    try:

        runpy.run_path(
            TARGET,
            run_name="__main__",
        )

    except SystemExit:
        pass

    except Exception:
        print()
        print("[MEGA] ORIGINAL SCRIPT FAILED:")
        traceback.print_exc()

    total_runtime = time.perf_counter() - t0

    print()
    print("=" * 100)
    print(
        f"[MEGA] TOTAL TEST PROCESS TIME: "
        f"{total_runtime:.3f}s"
    )
    print("=" * 100)

    print_results()
    print_summary()


if __name__ == "__main__":
    main()