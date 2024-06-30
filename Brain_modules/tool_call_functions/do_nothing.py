def do_nothing(progress_callback=None):
    if progress_callback:
        progress_callback("Executing do nothing tool")
    return {"result": "Nothing was done as requested this round."}
