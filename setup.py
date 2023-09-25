from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''Lee los requisitos del archivo de texto y devuelve una lista de requisitos'''
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements] 
        if '-e .' in requirements:
            requirements.remove('-e .')

    return requirements
    

setup(
    name='proyectoml1',
    version='0.1.0',
    author='Fernando',
    author_email='altffdez@gmail.com',
    packages=find_packages(),
    install_requeires=get_requirements('requirements.txt')
)