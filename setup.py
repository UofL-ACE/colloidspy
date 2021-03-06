import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def get_reqs():
    import sys
    import importlib.util
    requirements = [
        'numpy>=1.18',
        'pandas>=1.2',
        'scipy>=1.5',
        'scikit-image>=0.16',
        'tqdm>=4.30',
        'matplotlib>=3.3',
        'PyQt5'
        ]
    
    if importlib.util.find_spec('cv2') is not None:
        if sys.platform == "linux":
            requirements.append('opencv-python-headless')
        else:

            requirements.append('opencv-python>=4')

    return requirements

setuptools.setup(
    name="colloidspy",
    version="2021.05",
    author="Adam Cecil",
    author_email="ajcecil64@gmail.com",
    description="2D colloidal image analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UofL-ACE/colloidspy",
    license='LICENSE.txt',
    packages=setuptools.find_packages(),
    install_requires=get_reqs(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

