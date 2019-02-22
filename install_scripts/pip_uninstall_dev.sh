#!/bin/sh

# echo "- - - - - - - - uninstalling pastedeploy"
# pip uninstall --yes pastedeploy
# echo "- - - - - - - - uninstalling eventlet"
# pip uninstall --yes eventlet
# echo "- - - - - - - - uninstalling greenlet"
# pip uninstall --yes greenlet
# echo "- - - - - - - - uninstalling netifaces"
# pip uninstall --yes netifaces
echo "- - - - - - - - uninstalling simplejson"
pip uninstall --yes simplejson
# echo "- - - - - - - - uninstalling setuptools"
# pip uninstall setuptools
# echo "- - - - - - - - uninstalling six"
# pip uninstall --yes six
echo "- - - - - - - - uninstalling pyopenssl"
pip uninstall --yes pyopenssl
# echo "- - - - - - - - uninstalling cryptography"
# pip uninstall --yes cryptography
# echo "- - - - - - - - uninstalling dnspython"
# pip uninstall --yes dnspython
echo "- - - - - - - - deleting python3-dev residue (config-3.6m-x86_64-linux-gnu)"
rm -rf /opt/usr/local/lib/python3.6/config-3.6m-x86_64-linux-gnu/
