import time, atexit
import ctf
import numpy as np

TIME_DICT = {}
NUM_CALLS_DICT = {}


def get_tag(tag_names, tag_inputs, args):
    tag_list = []
    for i in range(len(tag_inputs)):
        var = args[tag_inputs[i]]
        if isinstance(var, np.ndarray) or isinstance(var, ctf.core.tensor):
            tag_list.append(f"{tag_names[i]}:{var.shape}")
        else:
            tag_list.append(f"{tag_names[i]}:{var}")
    return tag_list


def backend_profiler(timeit=True, tag_names=[], tag_inputs=[]):
    assert len(tag_names) == len(tag_inputs)

    def decorator(f):
        def f_timer(*args, **kwargs):
            start = time.time()
            result = f(*args, **kwargs)
            end = time.time()

            tag_list = get_tag(tag_names, tag_inputs, args)
            print(f"{f.__name__} with {tag_list} took {end - start} s")

            if f.__name__ in NUM_CALLS_DICT.keys():
                NUM_CALLS_DICT[f.__name__] += 1
                TIME_DICT[f.__name__] += end - start
            else:
                NUM_CALLS_DICT[f.__name__] = 1
                TIME_DICT[f.__name__] = end - start

            return result

        if timeit:
            return f_timer
        else:
            return f

    return decorator


def exit_handler():
    if not TIME_DICT == {}:
        print(f"---profiling info---")
        for funcname, time in TIME_DICT.items():
            print(
                f"Calling {funcname} {NUM_CALLS_DICT[funcname]} times, overall time {time}."
            )


atexit.register(exit_handler)
