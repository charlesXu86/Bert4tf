#-*- coding:utf-8 _*-
"""
@author:charlesXu
@file: setup.py
@desc: 设置
@time: 2019/05/23
"""

from setuptools import setup

setup(
    name="bert4tf",
    version="1.0.3",
    keywords=("bert4tf", "tensorflow"),
    description="bert for tensorflow",
    long_description="...",
    license="MIT Licence",
    url="https://github.com/charlesXu86/Time_Convert",
    author="xu",
    author_email="charlesxu86@163.com",
    packages={'bert4tf': 'bert4tf'},
    package_data={},
    include_package_data=True,
    platforms="any",
    install_requires=['tensorflow_gpu==1.14.0'],
    zip_safe=False,
    classifiers=[
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.6'
        ]
)
print("Welcome to Bert4TF")