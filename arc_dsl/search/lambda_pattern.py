from arc_dsl.interpreter import Symbol

from .types import Pattern, ArgPattern


def get_lambda_pattern(lambda_obj, patterns):
    # Step 1: Check if the Lambda calls another variable (assumed to be a Lambda)
    if _calls_unknown_variable(lambda_obj.body, patterns, lambda_obj.params):
        # Give up if it calls an unknown variable
        return None

    # Step 2: Determine the return type from the outermost call
    return_type = _get_return_type(lambda_obj.body, patterns)
    if return_type is None:
        # Could not determine the return type
        return None

    # Step 3: Determine the input types for each parameter
    param_types = {}
    for param in lambda_obj.params:
        first_usage = _find_first_call_with_param(lambda_obj.body, param)
        if first_usage is not None:
            function_symbol, arg_index = first_usage
            # Find the pattern for the function_symbol
            pattern = next((p for p in patterns if p.symbol == function_symbol), None)
            if pattern and arg_index < len(pattern.arg_classes):
                arg_pattern = pattern.arg_classes[arg_index]
                param_types[param] = arg_pattern.type
            else:
                # Could not find a matching pattern or invalid argument index
                param_types[param] = None
        else:
            # Parameter is not used; type is unknown
            param_types[param] = None

    # Construct the Pattern object representing the Lambda's type signature
    arg_patterns = [
        ArgPattern(type=param_types.get(param)) for param in lambda_obj.params
    ]
    return Pattern(symbol=None, arg_classes=arg_patterns, type=return_type)


def _calls_unknown_variable(form, patterns, params):
    if isinstance(form, list):
        # Function call
        fun_symbol = form[0]
        if isinstance(fun_symbol, Symbol):
            if not any(p.symbol == fun_symbol for p in patterns):
                if fun_symbol in params:
                    # Function symbol is a parameter (higher-order function); give up
                    return True
                else:
                    # Unknown function symbol; give up
                    return True
        else:
            # Function symbol is not a symbol (e.g., another Lambda); give up
            return True
        # Recursively check arguments
        for arg in form[1:]:
            if _calls_unknown_variable(arg, patterns, params):
                return True
    elif isinstance(form, Symbol):
        # It's a variable; nothing to check
        pass
    # Literal values or other types
    return False


def _get_return_type(form, patterns):
    if isinstance(form, list):
        fun_symbol = form[0]
        if isinstance(fun_symbol, Symbol):
            pattern = next((p for p in patterns if p.symbol == fun_symbol), None)
            if pattern:
                return pattern.type
    elif isinstance(form, int):
        return int
    return None  # Return type could not be determined


def _find_first_call_with_param(form, param):
    if isinstance(form, list):
        fun_symbol = form[0]
        args = form[1:]
        # Check if param is directly in args
        for i, arg in enumerate(args):
            if arg == param:
                return (fun_symbol, i)
        # Recursively check arguments
        for arg in args:
            result = _find_first_call_with_param(arg, param)
            if result:
                return result
    elif isinstance(form, Symbol):
        # Direct usage of the parameter; no function call
        pass
    return None


def _contains_param(form, param):
    if isinstance(form, list):
        return any(_contains_param(elem, param) for elem in form)
    elif isinstance(form, Symbol):
        return form == param
    return False
