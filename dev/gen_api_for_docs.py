

def gen_api_for_docs():
    """
    import sys, ubelt
    sys.path.append(ubelt.expandpath('~/code/ubelt/dev'))
    from gen_api_for_docs import *  # NOQA
    """
    from count_usage_freq import count_usage
    usage = count_usage()

    modname = 'kwarray'

    gaurd = ('=' * 64 + ' ' + '=' * 16)
    print(gaurd)
    print('{:<64} {:>8}'.format(' Function name ', 'Usefulness'))
    print(gaurd)
    for key, value in usage.items():
        print('{:<64} {:>16}'.format(':func:`' + modname + '.' + key + '`', value))
    print(gaurd)

    import ubelt as ub
    module = ub.import_module_from_name(modname)

    attrnames = module.__all__

    if hasattr(module, '__protected__'):
        # Hack for lazy imports
        for subattr in module.__protected__:
            submod = ub.import_module_from_name(modname + '.' + subattr)
            setattr(module, subattr, submod)
        attrnames += module.__protected__

    for attrname in attrnames:
        member = getattr(module, attrname)

        submembers = getattr(member, '__all__', None)

        # if attrname.startswith('util_'):
        if not submembers:
            from mkinit.static_mkinit import _extract_attributes
            try:
                submembers = _extract_attributes(member.__file__)
            except AttributeError:
                pass

        if submembers:
            print('\n:mod:`{}.{}`'.format(modname, attrname))
            print('-------------')
            for subname in submembers:
                if not subname.startswith('_'):
                    print(':func:`{}.{}`'.format(modname, subname))
            submembers = dir(member)


if __name__ == '__main__':
    """
    CommandLine:
        cd ~/code/kwarray/dev
        python gen_api_for_docs.py
        python gen_api_for_docs.py --extra_modnames=bioharn,

        # Paste output into
        ~/code/kwarray/docs/source/index.rst
    """
    gen_api_for_docs()
