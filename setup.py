#-*- coding:utf-8 _*-
"""
@author:charlesXu
@file: setup.py
@desc: 设置
@time: 2019/05/23
"""

from setuptools import setup, find_packages, convert_path

def _version():
    ns = {}
    with open(convert_path("bert4tf/version.py"), "r") as fh:
        exec(fh.read(), ns)
    return ns['__version__']

__version = _version()

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name="bert4tf",
    version=__version,
    keywords=("bert4tf", "tensorflow2"),
    description="bert for tensorflow2",
    long_description="text/x-rst",
    license="MIT Licence",
    url="https://github.com/charlesXu86/Bert4tf",
    author="xu",
    author_email="charlesxu86@163.com",
    packages={'bert4tf': 'bert4tf'},
    package_data={"": ["*.txt", "*.rst"]},
    include_package_data=True,
    platforms="any",
    install_requires=['tensorflow'],
    zip_safe=False,
    classifiers=[
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.6'
        ]
)
print("Welcome to Bert4TF")