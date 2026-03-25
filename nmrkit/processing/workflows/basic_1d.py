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

    return data
