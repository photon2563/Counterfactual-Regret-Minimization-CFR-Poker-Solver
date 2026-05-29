CXX = clang++
CXXFLAGS = -O3 -Wall -shared -fPIC -std=c++11

ifeq ($(shell uname -s),Darwin)
	TARGET = libcfr_engine.dylib
else
	TARGET = libcfr_engine.so
endif

all: $(TARGET)

$(TARGET): cfr_engine.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) cfr_engine.cpp

clean:
	rm -f *.so *.dylib
