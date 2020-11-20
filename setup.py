from setuptools import setup

with open("requirements.txt", "r") as f:
    REQUIRED_PACKAGES = f.read().splitlines()

setup(
    name="pylithics",
    version="0.0.1",
    description="A lithics study using computer vision.",
    url="https://github.com/alan-turing-institute/Palaeoanalytics",
    author="Jason Gellis, Camila Rangel Smith",
    license="MIT",
    include_package_data=True,
    packages=["pylithics",
              "pylithics.src"],
    install_requires=REQUIRED_PACKAGES,
    # we will need this later, i'll leave it commented as a reminder.
    #entry_points={"console_scripts": [
     #   "pyveg_calc_EC=pyveg.scripts.calc_euler_characteristic:main",

    #]},
)