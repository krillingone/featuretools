import time
from functools import wraps
from inspect import signature

import pkg_resources


def entry_point(name):
    def inner_function(func):
        @wraps(func)
        def function_wrapper(*args, **kwargs):
            """function_wrapper of greeting"""
            # add positional args as named kwargs
            # args是不定长参数，kwargs是字典类型参数，这里把args的参数转成kwargs
            # 这个在python是需要处理的，如果函数参数只搞了部分，则根据顺序赋值给前面几个
            on_call_kwargs = kwargs.copy()
            sig = signature(func)
            for arg, parameter in zip(args, sig.parameters):
                on_call_kwargs[parameter] = arg

            # collect and initialize all registered entry points
            entry_points = []
            for entry_point in pkg_resources.iter_entry_points(name):
                entry_point = entry_point.load()
                entry_points.append(entry_point())

            # send arguments before function is called
            for ep in entry_points:
                ep.on_call(on_call_kwargs)

            try:
                # call function
                start = time.time()
                return_value = func(*args, **kwargs)
                runtime = time.time() - start
            except Exception as e:
                runtime = time.time() - start
                # send error
                for ep in entry_points:
                    ep.on_error(error=e, runtime=runtime)
                raise e

            # send return value
            for ep in entry_points:
                ep.on_return(return_value=return_value, runtime=runtime)

            return return_value

        return function_wrapper

    return inner_function
