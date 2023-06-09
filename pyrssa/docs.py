import os
doc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extdoc')


def add_doc(function=None, filename=None):
    """
    Decorator to load detailed documentation from external files.

    Parameters
    ----------
        function
            If decorator is used alone, function will be the actual function which it is decorating.
            If decorator is called with arguments, function will be None.
        filename
            the name of the file from which to load the documentation. If None, decorated function name is used.
    """
    def decorator(f):
        final_name = f.__name__ if filename is None else filename
        file_path = os.path.join(doc_path, f"{final_name}.rst")
        file_contents = ''.join(['    ' + line for line in open(file_path).readlines()])
        f.__doc__ += '\n\n' + file_contents

        return f
    if function:
        return decorator(function)
    return decorator
