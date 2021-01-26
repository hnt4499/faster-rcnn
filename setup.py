from setuptools import setup, find_packages


setup(
    name="faster_rcnn",
    version="1.0.0",
    author="Hoang Nghia Tuyen",
    author_email="hnt4499@gmail.com",
    url="https://github.com/hnt4499/faster-rcnn",
    install_requires=[],
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "": ["LICENSE", "README.md"],
    },
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    description="Full Faster R-CNN implementation and experimental results",
    keywords=["deep learning", "computer vision", "faster_rcnn"]
)
