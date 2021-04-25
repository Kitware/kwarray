#!/usr/bin/env python
# -*- coding: utf-8 -*-
if __name__ == '__main__':
    import pytest
    import sys
    package_name = 'kwarray'
    pytest_args = [
        '-p', 'no:doctest',
        '--cov-config', '.coveragerc',
        '--cov-report', 'html',
        '--cov-report', 'term',
        '--xdoctest',
        '--cov=' + package_name,
        package_name,
    ]
    pytest_args = pytest_args + sys.argv[1:]
    ret = pytest.main(pytest_args)
    sys.exit(ret)
