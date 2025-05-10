from setuptools import setup, find_packages

setup(
    name="blood_analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'opencv-python>=4.8.0',
        'numpy>=1.24.0',
        'scikit-image>=0.21.0',
        'tensorflow>=2.13.0',
        'scikit-learn>=1.3.0'
    ]
)
