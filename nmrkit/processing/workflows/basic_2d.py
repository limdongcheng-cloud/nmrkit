import nmrkit as nk


def process(data, **kwargs):
    """Basic 2D NMR data processing workflow.

    Parameters
    ----------
    data : nmrkit.NMRData
        Input NMR data.
    **kwargs
        Additional processing parameters.

        zf_size_dim1 : int, optional
            Zero filling size for dimension 1.
        zf_size_dim2 : int, optional
            Zero filling size for dimension 2.

    Returns
    -------
    data : nmrkit.NMRData
        Processed NMR data.
    """
    # Direct dimension processing
    data = nk.zf(data, dim=0)
    data = nk.ft(data, dim=0)
    data = nk.correct_digital_filter_phase(data)
    data = nk.autophase(data, dim=0)
    if kwargs.get("baseline", False):
        data = nk.baseline_correct(data, dim=0)

    # Indirect dimension processing
    data = nk.complexify_indirect(data)

    data = nk.zf(data, dim=1, size=2048)
    data = nk.ft(data, dim=1)
    data = nk.autophase(data, dim=1)
    if kwargs.get("baseline", False):
        data = nk.baseline_correct(data, dim=1)

    return data
