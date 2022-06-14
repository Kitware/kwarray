

import scriptconfig as scfg


class UsageConfig(scfg.Config):
    default = {
        'print_packages': False,
        'remove_zeros': True,
        # 'hardcoded_ubelt_hack': True,
        'extra_modnames': [],
    }


def count_usage(cmdline=True):

    config = UsageConfig(cmdline=cmdline)

    import ubelt as ub
    import glob
    from os.path import join
    names = [
        'netharn', 'ndsampler', 'kwimage', 'kwplot',
    ] + config['extra_modnames']

    all_fpaths = []
    for name in names:
        if name:
            repo_fpath = ub.expandpath(join('~/code', name))
            fpaths = glob.glob(join(repo_fpath, '**', '*.py'), recursive=True)
            for fpath in fpaths:
                all_fpaths.append((name, fpath))

    print('names = {}'.format(ub.repr2(names)))

    import re

    import ubelt as ub
    modname = 'kwarray'
    module = ub.import_module_from_name(modname)

    package_name = module.__name__
    package_allvar = module.__all__

    pat = re.compile(r'\b' + package_name + r'\.(?P<attr>[a-zA-Z_][A-Za-z_0-9]*)\b')

    pkg_to_hist = ub.ddict(lambda: ub.ddict(int))
    for name, fpath in ub.ProgIter(all_fpaths):
        # print('fpath = {!r}'.format(fpath))
        text = ub.readfrom(fpath, verbose=0)
        # text = open(fpath, 'r').read()
        for match in pat.finditer(text):
            attr = match.groupdict()['attr']
            if attr in package_allvar:
                pkg_to_hist[name][attr] += 1

    hist_iter = iter(pkg_to_hist.values())
    usage = next(hist_iter).copy()
    for other in hist_iter:
        for k, v in other.items():
            usage[k] += v
    for attr in package_allvar:
        usage[attr] += 0

    for name in pkg_to_hist.keys():
        pkg_to_hist[name] = ub.odict(sorted(pkg_to_hist[name].items(), key=lambda t: t[1])[::-1])

    usage = ub.odict(sorted(usage.items(), key=lambda t: t[1])[::-1])

    if config['print_packages']:
        print(ub.repr2(pkg_to_hist, nl=2))

    if config['remove_zeros']:
        for k, v in list(usage.items()):
            if v == 0:
                usage.pop(k)

    # if config['hardcoded_ubelt_hack']:
    #     for k in list(usage):
    #         if k.startswith('util_'):
    #             usage.pop(k)
    #         if k.startswith('_util_'):
    #             usage.pop(k)
    #         # ub._util_deprecated
    #         from ubelt import _util_deprecated
    #         if k in dir(_util_deprecated):
    #             usage.pop(k)

    print(ub.repr2(usage, nl=1))
    return usage


if __name__ == '__main__':
    """
    CommandLine:
        python ~/code/kwarray/dev/count_usage_freq.py --help
        python ~/code/kwarray/dev/count_usage_freq.py --extra_modnames=bioharn,watch --print_packages=True

    """
    count_usage()
