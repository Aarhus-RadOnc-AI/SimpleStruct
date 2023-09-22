def njit(*args, **kwargs):
    def wrapper(f):
        return f
    if len(args) == 0:
        # @njit()
        return wrapper
    else:
        # @njit
        return args[0]