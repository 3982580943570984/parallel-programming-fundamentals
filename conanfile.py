from conan import ConanFile
from conan.tools.cmake import cmake_layout


class Recipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    def requirements(self):
        self.requires("boost/1.86.0")
        self.requires("nlohmann_json/3.11.3")
        self.requires("openmpi/4.1.6")

    def layout(self):
        cmake_layout(self)
