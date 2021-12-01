# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import sys

# MathSymbolsAscii should always be available
from kwarray.math_symbols.math_symbols_ascii import MathSymbolsAscii

# MathSymbolsUnicode might not be available
try:
    if sys.version_info[0] == 2:
        raise Exception
    from kwarray.math_symbols.math_symbols_unicode import MathSymbolsUnicode
    from kwarray.math_symbols.math_symbols_unicode import MathSymbolsUnicode as MathSymbols
except Exception:
    from kwarray.math_symbols.math_symbols_ascii import MathSymbolsAscii as MathSymbols
    MathSymbolsUnicode = None

__all__ = ['MathSymbolsAscii', 'MathSymbolsUnicode', 'MathSymbols']
