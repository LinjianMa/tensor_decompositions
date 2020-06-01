import time, atexit

TIME_DICT = {}
NUM_CALLS_DICT = {}


def backend_profiler(timeit=True, tag_names=[], tag_inputs=[]):
    assert len(tag_names) == len(tag_inputs)

    def decorator(f):
        def f_timer(*args, **kwargs):
            start = time.time()
            result = f(*args, **kwargs)
            end = time.time()

            tag_list = [
                f"{tag_names[i]}:{args[tag_inputs[i]]}"
                for i in range(len(tag_inputs))
            ]
            print(f"{f.__name__} with {tag_list} took {end - start} time")

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
