# location of the Python header files
PYTHON_VERSION = 3.11
PYTHON_DIR = /Users/akalanka/miniconda3/envs/roller
# PYTHON_DIR = /usr/local/pkg/python/$(PYTHON_VERSION)
PYTHON_INCLUDE = $(PYTHON_DIR)/include/python$(PYTHON_VERSION)
PYTHON_LIB = /Users/akalanka/miniconda3/envs/roller/lib/python3.11/config-3.11-darwin

# location of the Boost Python include files and library
BOOST_INC = /Users/akalanka/miniconda3/envs/roller/include/boost
BOOST_LIB = /Users/akalanka/miniconda3/envs/roller/lib

EIGEN_INC = /Users/akalanka/miniconda3/envs/roller/include/eigen3

# Location of Fortran libraries,
# uncomment to use the Fast Multipole Metho library dstfmmlib3d-1.2
# FORTRAN_LIB = --link-fopenmp -L/path/to/stfmmlib3d-1.2/ -lstfmm3d 

CXX_FLAGS = -D__FILENAME__='"$(shell pwd)"'

# 
TARGET = Lubrication_Class

all: $(TARGET).so

$(TARGET).so: $(TARGET).o
	g++ -O3 -shared  -dynamiclib $(TARGET).o -L$(BOOST_LIB)  -lboost_python311 -lboost_numpy311 -L$(PYTHON_LIB) -undefined dynamic_lookup -o $(TARGET).so

$(TARGET).o: $(TARGET).cc
	g++ -O3 -std=c++11 -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -I$(EIGEN_INC) $(CXX_FLAGS) -fPIC -c $(TARGET).cc
clean: 
	rm -f $(TARGET).o $(TARGET).so
