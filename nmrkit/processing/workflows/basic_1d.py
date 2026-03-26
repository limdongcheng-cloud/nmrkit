import nmrkit as nk


def process(data, **kwargs):
    """Basic 1D NMR data processing workflow.

    Parameters
    ----------
    data : nmrkit.Data
        Input NMR data.
    **kwargs
        Additional processing parameters.

        em_lb : float, optional
            Line broadening parameter for exponential multiplication.
        ph0 : float, optional
            Zero order phase correction.
        ph1 : float, optional
            First order phase correction.
        pivot : int, optional
            Pivot point for phase correction.
        baseline : bool, optional
            Enable baseline correction (default: False). When True, uses
            AsLS algorithm. Disabled by default to avoid introducing
            subjective bias into automated processing.
        bc_method : str, optional
            Baseline correction method: "asls", "airpls", or "polynomial".
            Only used when baseline=True.
        bc_lambda : float, optional
            Smoothness parameter for AsLS/airPLS. Only used when baseline=True.
        bc_p : float, optional
            Asymmetry parameter for AsLS. Only used when baseline=True.
        bc_order : int, optional
            Polynomial order (for method="polynomial"). Only used when
            baseline=True.

    Returns
    -------
    data : nmrkit.Data
        Processed NMR data.
    """
    # Remove digital filter in time domain for JEOL data (before FT)
    if data.source_format == "delta":
        data = nk.remove_digital_filter(data)

    em_lb = kwargs.get("em_lb", 1)
    data = nk.em(data, lb=em_lb)

    data = nk.zf(data)

    data = nk.ft(data)

    # Frequency-domain digital filter correction for Bruker data
    if data.source_format != "delta":
        data = nk.correct_digital_filter_phase(data)

    ph0 = kwargs.get("ph0", None)
    ph1 = kwargs.get("ph1", None)
    pivot = kwargs.get("pivot", None)

    if ph0 is not None and ph1 is not None and pivot is not None:
        data = nk.phase(data, ph0=ph0, ph1=ph1, pivot=pivot)
    else:
        data = nk.autophase(data)

    # Baseline correction (opt-in to avoid subjective bias in automation)
    if kwargs.get("baseline", False):
        bc_method = kwargs.get("bc_method", "asls")
        bc_kwargs = {}
        if "bc_lambda" in kwargs:
            bc_kwargs["lambda_"] = kwargs["bc_lambda"]
        if "bc_p" in kwargs:
            bc_kwargs["p"] = kwargs["bc_p"]
        if "bc_order" in kwargs:
            bc_kwargs["order"] = kwargs["bc_order"]
        data = nk.baseline_correct(data, method=bc_method, **bc_kwargs)

    return data
