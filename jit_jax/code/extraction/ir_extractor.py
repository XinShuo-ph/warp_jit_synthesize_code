import jax

def extract_ir(fn, *args, **kwargs):
    """
    Extracts JAXPR and HLO/StableHLO from a JAX-compatible function.
    
    Args:
        fn: Python function or jax.jit-decorated function.
        *args: Example arguments to trace with.
        **kwargs: Example keyword arguments to trace with.
        
    Returns:
        dict: containing 'jaxpr' (str) and 'hlo' (str).
    """
    try:
        # 1. Get Jaxpr
        # jax.make_jaxpr returns a function that produces the ClosedJaxpr when called
        jaxpr_obj = jax.make_jaxpr(fn)(*args, **kwargs)
        
        # 2. Get HLO
        # We need to jit it to lower it. 
        # Note: If fn is already jitted, re-jitting is usually fine/idempotent-ish for this purpose
        # but strictly speaking jax.jit(jax.jit(f)) might add overhead.
        # However, .lower() is a method on the Stage object returned by jit(f).
        
        # If the function is not already a Jitted object, we wrap it.
        # But even if it is, calling jit(fn) works.
        jit_fn = jax.jit(fn)
        lowered = jit_fn.lower(*args, **kwargs)
        hlo_text = lowered.as_text()
        
        return {
            "jaxpr": str(jaxpr_obj),
            "hlo": hlo_text,
            "success": True,
            "error": None
        }
        
    except Exception as e:
        return {
            "jaxpr": None,
            "hlo": None,
            "success": False,
            "error": str(e)
        }
