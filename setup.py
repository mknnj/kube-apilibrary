from setuptools import find_packages, setup
setup(
    name='kubernetesdlprofile',
    packages=find_packages(include=['kubernetesdlprofile']),
    version='0.1.1',
    description='Simple API to send DL profiling data to a kubernetes scheduler',
    author='Michele',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
)
