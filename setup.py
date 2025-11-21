
'''
The stup.py file is an essentioal part of packaging and distributing Python projects.
It is used by stuptools, a widely used library for managing Python packages to define the configuration of the project, such as its metaadata, 
dependencies, and more'''

from setuptools import setup, find_packages
from typing import List

##from typing import Generator

def get_requirements()->List[str]:
    """Reads the requirements.txt file and returns a list of dependencies."""
    requirements_list:List[str] = []
    try:
        with open("requirements.txt", "r") as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != '-e .':
                    requirements_list.append(requirement)
    except FileNotFoundError:
        print("requirements.txt file not found.")
    return requirements_list
setup(
    name="network_security_project",
    version="0.1.0",
    author="Raju Subba",
    packages=find_packages(),
    
)