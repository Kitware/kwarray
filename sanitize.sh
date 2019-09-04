#!/bin/bash 
__heredoc__="""
chmod +x sanitize.sh
"""

SAGING_DPATH=$HOME/temp/kwarray-stage
SOURCE_DPATH=.
mkdir -p $STAGING_DPATH

rsync -avrP --exclude=".git" $SOURCE_DPATH $STAGING_DPATH
