from setuptools import find_packages, setup

setup(
    name="siameseFastTraining",
    version="0.1.3",
    description="Library for fast Siamese Network Training - ULTRADATA",
    author="Michel Davidovich",
    author_email="micheldc55@gmail.com",
    packages=find_packages(),
    install_requires=["numpy", "torch", "efficientnet_pytorch", "tqdm", "torchvision", "Pillow"],
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.9",
)
